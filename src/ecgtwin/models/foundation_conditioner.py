"""JEPA-style multimodal conditioner used for ECG personalization."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ecgtwin.models.base_vector import BaseVectorBottleneck
from ecgtwin.models.embedder import RoPEEmbedder


class FeedForward(nn.Module):
    """Two-layer MLP block used inside the conditioner transformer."""

    def __init__(self, embed_dim: int, ff_hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_size, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionerBlock(nn.Module):
    """Transformer block with self-attention and multimodal cross-attention."""

    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_size: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(embed_dim, ff_hidden_size, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        condition_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x_norm = self.norm1(x)
        x = x + self.dropout(self.self_attn(x_norm, x_norm, x_norm, need_weights=False)[0])

        x_norm = self.norm2(x)
        x = x + self.dropout(
            self.cross_attn(
                x_norm,
                condition,
                condition,
                key_padding_mask=condition_padding_mask,
                need_weights=False,
            )[0]
        )

        x = x + self.dropout(self.ff(self.norm3(x)))
        return x


class FoundationConditioner(nn.Module):
    """Runtime ECG conditioner exported from JEPA training."""

    def __init__(
        self,
        embed_dim: int = 256,
        in_channel: int = 4,
        num_heads: int = 8,
        ff_hidden_size: int = 1024,
        num_layers: int = 6,
        dropout: float = 0.0,
        text_embed_dim: int = 768,
        patient_info_size: int = 3,
        base_vector_mode: str = "standard",
        base_vector_bottleneck_dim: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patient_info_size = patient_info_size
        self.signal_embedding = nn.Linear(in_channel, embed_dim)
        self.rope = RoPEEmbedder(dim=embed_dim)
        self.text_projector = nn.Linear(text_embed_dim + patient_info_size, embed_dim)
        self.classfree_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.layers = nn.ModuleList(
            [ConditionerBlock(embed_dim, num_heads, ff_hidden_size, dropout) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        if base_vector_mode == "bottleneck":
            self.base_vector_adapter = BaseVectorBottleneck(embed_dim, base_vector_bottleneck_dim)
        else:
            self.base_vector_adapter = nn.Identity()

    def _build_condition(
        self,
        text_embed: torch.Tensor | None,
        text_mask: torch.Tensor | None,
        patient_info: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if text_embed is None:
            return self.classfree_emb.expand(batch_size, -1, -1), None

        expanded_patient = patient_info.unsqueeze(1).expand(-1, text_embed.shape[1], -1)
        condition = self.text_projector(torch.cat([text_embed, expanded_patient], dim=-1))
        padding_mask = None if text_mask is None else text_mask <= 0
        return condition, padding_mask

    def encode_tokens(
        self,
        x: torch.Tensor,
        text_embed: torch.Tensor | None,
        text_mask: torch.Tensor | None,
        patient_info: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.signal_embedding(x)
        if token_mask is not None:
            x = torch.where(token_mask.unsqueeze(-1), self.mask_token.expand_as(x), x)
        x = self.rope(x)
        condition, padding_mask = self._build_condition(text_embed, text_mask, patient_info, x.shape[0])
        for layer in self.layers:
            x = layer(x, condition, padding_mask)
        return self.output_norm(x)

    def pool_features(self, token_features: torch.Tensor) -> torch.Tensor:
        return self.base_vector_adapter(token_features.mean(dim=1))

    def extract_features(
        self,
        x: torch.Tensor,
        text_embed: torch.Tensor | None,
        mask: torch.Tensor | None,
        patient_info: torch.Tensor,
        reduce: bool = True,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_features = self.encode_tokens(x, text_embed, mask, patient_info, token_mask=token_mask)
        if reduce:
            return self.pool_features(token_features)
        return token_features


class PredictorHead(nn.Module):
    """Predictor that maps masked student states into teacher space."""

    def __init__(self, embed_dim: int, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sample_block_mask(
    batch_size: int,
    seq_len: int,
    mask_ratio: float,
    mask_span: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample a per-sequence boolean mask with short contiguous spans."""
    target_mask_count = max(1, int(math.ceil(seq_len * mask_ratio)))
    span = max(1, int(mask_span))
    mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)

    for batch_index in range(batch_size):
        remaining = target_mask_count
        while remaining > 0:
            current_span = min(span, remaining)
            max_start = max(seq_len - current_span, 0)
            start = int(torch.randint(0, max_start + 1, (1,), device=device).item())
            mask[batch_index, start : start + current_span] = True
            remaining = target_mask_count - int(mask[batch_index].sum().item())
            if int(mask[batch_index].sum().item()) >= target_mask_count:
                break

        if int(mask[batch_index].sum().item()) > target_mask_count:
            true_indices = torch.nonzero(mask[batch_index], as_tuple=False).squeeze(-1)
            keep = true_indices[torch.randperm(true_indices.numel(), device=device)[:target_mask_count]]
            mask[batch_index].zero_()
            mask[batch_index, keep] = True

        if int(mask[batch_index].sum().item()) == seq_len:
            mask[batch_index, torch.randint(0, seq_len, (1,), device=device)] = False

    return mask


def variance_regularization(features: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    """Penalize representation collapse by enforcing batch variance."""
    std = torch.sqrt(features.var(dim=0, unbiased=False) + 1.0e-4)
    return F.relu(target_std - std).mean()
