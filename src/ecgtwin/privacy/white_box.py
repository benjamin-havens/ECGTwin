"""White-box membership scoring for the IBE and diffusion stages."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ecgtwin.models.base_vector import apply_base_vector_ablation

from .features import sample_latent_batch, sample_patient_tensor, sample_text_tensors


def ibe_white_box_score(sample: dict, partner: dict, ibe_model, device: torch.device) -> dict[str, float]:
    """Score a record using internal IBE activations and gradients."""
    anchor_latent = sample_latent_batch(sample, device).transpose(2, 1).detach().clone().requires_grad_(True)
    partner_latent = sample_latent_batch(partner, device).transpose(2, 1)
    anchor_text, anchor_mask = sample_text_tensors(sample, device)
    partner_text, partner_mask = sample_text_tensors(partner, device)
    anchor_pat = sample_patient_tensor(sample, device)
    partner_pat = sample_patient_tensor(partner, device)

    anchor_feature = ibe_model.extract_features(anchor_latent, anchor_text, anchor_mask, anchor_pat, reduce=True)
    with torch.no_grad():
        partner_feature = ibe_model.extract_features(partner_latent, partner_text, partner_mask, partner_pat, reduce=True)
        classfree_feature = ibe_model.extract_features(anchor_latent.detach(), None, None, anchor_pat, reduce=True)

    anchor_feature_norm = anchor_feature / anchor_feature.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    partner_feature_norm = partner_feature / partner_feature.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    similarity = (anchor_feature_norm * partner_feature_norm).sum(dim=-1).mean()
    gradient = torch.autograd.grad(-similarity, anchor_latent)[0]
    conditioning_gap = (anchor_feature.detach() - classfree_feature).norm(dim=-1).mean()

    return {
        "score": float(similarity.item()),
        "similarity": float(similarity.item()),
        "gradient_norm": float(gradient.norm().item()),
        "conditioning_gap": float(conditioning_gap.item()),
    }


def diffusion_white_box_score(
    sample: dict,
    ibe_model,
    noise_predictor,
    diffused_model,
    device: torch.device,
    timesteps: list[int],
    base_vector_mode: str,
    base_vector_noise_std: float,
    seed: int,
) -> dict[str, float]:
    """Score a record using diffusion denoising loss and gradients."""
    latent = sample_latent_batch(sample, device).detach().clone().requires_grad_(True)
    text_embed, text_mask = sample_text_tensors(sample, device)
    pat_info = sample_patient_tensor(sample, device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    base_vector = ibe_model.extract_features(
        latent.transpose(2, 1),
        text_embed,
        text_mask,
        pat_info,
        reduce=True,
    )
    base_vector = apply_base_vector_ablation(
        base_vector,
        mode=base_vector_mode,
        noise_std=base_vector_noise_std,
    )

    losses = []
    residual_norms = []
    for timestep in timesteps:
        timestep_value = int(min(max(timestep, 1), diffused_model.config.num_train_timesteps - 1))
        timestep_tensor = torch.tensor([timestep_value], device=device, dtype=torch.long)
        noise = torch.randn(latent.shape, generator=generator, device=device)
        noised_latent = diffused_model.add_noise(latent, noise, timestep_tensor)
        predicted_noise = noise_predictor(noised_latent, timestep_tensor, text_embed, text_mask, pat_info, base_vector)
        losses.append(F.mse_loss(predicted_noise, noise))
        residual_norms.append((predicted_noise - noise).norm())

    mean_loss = torch.stack(losses).mean()
    mean_residual = torch.stack(residual_norms).mean()
    latent_gradient, base_gradient = torch.autograd.grad(mean_loss, [latent, base_vector], allow_unused=True)

    return {
        "score": float(-mean_loss.item()),
        "loss_mean": float(mean_loss.item()),
        "residual_norm_mean": float(mean_residual.item()),
        "latent_gradient_norm": float(0.0 if latent_gradient is None else latent_gradient.norm().item()),
        "base_gradient_norm": float(0.0 if base_gradient is None else base_gradient.norm().item()),
    }
