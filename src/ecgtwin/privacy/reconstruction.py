"""Model-inversion-style latent reconstruction scoring helpers."""

from __future__ import annotations

import torch

from .black_box import compute_distances


def _flatten_candidates(candidate_latents: torch.Tensor) -> torch.Tensor:
    """Flatten a batch of latent candidates into feature vectors."""
    if candidate_latents.ndim < 2:
        raise ValueError("Reconstruction scoring requires a batch of candidate latents")
    return candidate_latents.reshape(candidate_latents.shape[0], -1).to(dtype=torch.float32)


def _flatten_query(query_latent: torch.Tensor) -> torch.Tensor:
    """Flatten one latent query into a feature vector."""
    return query_latent.reshape(-1).to(dtype=torch.float32)


def _cosine_similarities(query_feature: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between one query and all candidates."""
    query_norm = query_feature / query_feature.norm().clamp_min(1e-8)
    candidate_norm = candidate_features / candidate_features.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return (candidate_norm * query_norm.unsqueeze(0)).sum(dim=1)


def reconstruction_score(
    query_latent: torch.Tensor,
    candidate_latents: torch.Tensor,
    distance: str,
) -> dict[str, float]:
    """Score how well a synthetic pool reconstructs one target latent."""
    query_feature = _flatten_query(query_latent)
    candidate_features = _flatten_candidates(candidate_latents)
    distances = compute_distances(query_feature, candidate_features, metric=distance)
    best_index = int(torch.argmin(distances).item())
    cosine_scores = _cosine_similarities(query_feature, candidate_features)
    distance_min = float(distances[best_index].item())
    return {
        "score": -distance_min,
        "distance_min": distance_min,
        "cosine_max": float(cosine_scores.max().item()),
        "best_candidate_index": best_index,
    }
