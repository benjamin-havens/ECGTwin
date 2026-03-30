"""Embedding layers used across ECGTwin transformer and UNet backbones."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PatchEmbed1D(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim, bias=True):
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias
        )
        self.patch_size = patch_size

    def forward(self, x):  # x: [B, C, L]
        x = self.proj(x)  # [B, embed_dim, L/P]
        x = x.transpose(1, 2)  # [B, N_patches, embed_dim]
        return x


def rotate_half(x):
    """Splits the last dimension in half and swaps the halves with negation."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)

class RoPEEmbedder(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    
    def forward(self, x):
        """Applies RoPE to the input tensor x."""
        seq_len = x.shape[1]
        device = x.device
        pos = torch.arange(seq_len, device=device, dtype=torch.float).unsqueeze(1)
        freqs = torch.einsum("nd,d->nd", pos, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        
        cos_emb = emb.cos().unsqueeze(0)  # Shape (1, seq_len, dim)
        sin_emb = emb.sin().unsqueeze(0)
        
        return x * cos_emb + rotate_half(x) * sin_emb


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PatientInfoEmbedder(nn.Module):
    def __init__(self, pat_info_length: int, dim: int, mlp_ratio: int=4):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(pat_info_length, dim))  # (L, Dim)
        self.norm = nn.Parameter(torch.ones(pat_info_length))
        self.bias = nn.Parameter(torch.zeros(pat_info_length))

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.SiLU(),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )

        nn.init.xavier_uniform_(self.embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L)
        # self.embedding: (L, Dim)

        x = x * self.norm + self.bias
        x =  x.unsqueeze(-1) * self.embedding.unsqueeze(0)  # (B, L, Dim)

        x = self.mlp(x)
        # x: (B, L, dim)
        return x


class PositionalEmbedder(nn.Module):
    def __init__(self, embed_size, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_encoding = nn.Parameter(pe.unsqueeze(0))  # Make it learnable
        # self.pos_encoding.requires_grad = False

    def forward(self, x):
        return x + self.pos_encoding[:, :x.size(1), :]
