"""Shared helpers for conditioning-vector ablations and bottlenecks."""

from __future__ import annotations

import torch
import torch.nn as nn


class BaseVectorBottleneck(nn.Module):
    """Reduce a base vector through a learned bottleneck and restore its width."""

    def __init__(self, embed_dim: int, bottleneck_dim: int):
        super().__init__()
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")

        if bottleneck_dim >= embed_dim:
            self.layers = nn.Identity()
        else:
            self.layers = nn.Sequential(
                nn.Linear(embed_dim, bottleneck_dim),
                nn.GELU(),
                nn.Linear(bottleneck_dim, embed_dim),
            )

    def forward(self, base_vector: torch.Tensor) -> torch.Tensor:
        """Apply the learned bottleneck while keeping the original tensor shape."""
        return self.layers(base_vector)


def apply_base_vector_ablation(
    base_vector: torch.Tensor,
    mode: str = "standard",
    noise_std: float = 0.0,
) -> torch.Tensor:
    """Apply the configured non-learned base-vector ablation."""
    normalized_mode = mode.lower()
    if normalized_mode in {"standard", "bottleneck"}:
        return base_vector
    if normalized_mode == "remove":
        return torch.zeros_like(base_vector)
    if normalized_mode == "noise":
        if noise_std <= 0:
            return base_vector
        return base_vector + torch.randn_like(base_vector) * noise_std
    raise ValueError(f"Unsupported base-vector mode: {mode}")


def apply_random_feature_mask(base_vector: torch.Tensor, mask_prob: float) -> torch.Tensor:
    """Mask the same feature coordinates across the batch, matching the legacy workflow."""
    if mask_prob <= 0:
        return base_vector
    keep_mask = (torch.rand(1, base_vector.shape[1], device=base_vector.device) > mask_prob).float()
    return base_vector * keep_mask
