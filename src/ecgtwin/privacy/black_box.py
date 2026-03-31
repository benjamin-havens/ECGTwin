"""Black-box membership scoring against synthetic ECG pools."""

from __future__ import annotations

import torch

from .features import extract_pool_features, extract_record_feature


def compute_distances(query: torch.Tensor, candidates: torch.Tensor, metric: str = "l2") -> torch.Tensor:
    """Compute distances from one query feature to a candidate feature matrix."""
    if metric == "l2":
        return torch.linalg.norm(candidates - query.unsqueeze(0), dim=1)
    if metric == "cosine":
        query_norm = query / query.norm().clamp_min(1e-8)
        candidate_norm = candidates / candidates.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return 1 - (candidate_norm * query_norm.unsqueeze(0)).sum(dim=1)
    raise ValueError(f"Unsupported distance metric: {metric}")


def black_box_score(query_feature: torch.Tensor, candidate_features: torch.Tensor, k: int, distance: str) -> dict[str, float]:
    """Score membership by how closely the generator reproduces the query."""
    distances = compute_distances(query_feature, candidate_features, metric=distance)
    k = min(max(k, 1), distances.shape[0])
    topk = torch.topk(distances, k=k, largest=False).values
    return {
        "score": float(-topk.mean().item()),
        "distance_min": float(distances.min().item()),
        "distance_mean_k": float(topk.mean().item()),
    }


def domias_score(
    query_feature: torch.Tensor,
    candidate_features: torch.Tensor,
    k: int,
    distance: str,
    reference_split: float,
) -> dict[str, float]:
    """Compute a synthetic-only DOMIAS-style density-ratio score."""
    num_candidates = candidate_features.shape[0]
    split_index = min(max(int(num_candidates * reference_split), 1), num_candidates - 1)
    support_features = candidate_features[:split_index]
    reference_features = candidate_features[split_index:]

    support_distances = compute_distances(query_feature, support_features, metric=distance)
    reference_distances = compute_distances(query_feature, reference_features, metric=distance)
    k_support = min(max(k, 1), support_distances.shape[0])
    k_reference = min(max(k, 1), reference_distances.shape[0])

    support_radius = torch.topk(support_distances, k=k_support, largest=False).values.mean().clamp_min(1e-8)
    reference_radius = torch.topk(reference_distances, k=k_reference, largest=False).values.mean().clamp_min(1e-8)
    support_density = 1.0 / support_radius
    reference_density = 1.0 / reference_radius
    score = torch.log(support_density) - torch.log(reference_density)
    return {
        "score": float(score.item()),
        "support_density": float(support_density.item()),
        "reference_density": float(reference_density.item()),
    }


def score_synthetic_pool(
    sample: dict,
    synthetic_latents: torch.Tensor,
    feature_space: str,
    ibe_model,
    device: torch.device,
    k: int,
    distance: str,
    reference_split: float,
    use_amp: bool = False,
) -> dict[str, dict[str, float]]:
    """Compute both black-box and DOMIAS scores for one synthetic pool."""
    query_feature = extract_record_feature(
        sample,
        feature_space=feature_space,
        ibe_model=ibe_model,
        device=device,
        use_amp=use_amp,
    )
    pool_features = extract_pool_features(
        synthetic_latents,
        conditioning_sample=sample,
        feature_space=feature_space,
        ibe_model=ibe_model,
        device=device,
        use_amp=use_amp,
    )
    return {
        "black_box": black_box_score(query_feature, pool_features, k=k, distance=distance),
        "domias": domias_score(
            query_feature,
            pool_features,
            k=k,
            distance=distance,
            reference_split=reference_split,
        ),
    }
