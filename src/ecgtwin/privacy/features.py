"""Feature extraction utilities shared by privacy attacks."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from ecgtwin.data.patient import build_patient_info_tensor, sex_to_binary


def sample_text_tensors(sample: dict, device: torch.device, repeat: int = 1):
    """Return the stored text embedding and optional mask for a sample."""
    text_embed = sample["label"].get("text_embed")
    if text_embed is None:
        raise ValueError("Privacy utilities require precomputed text embeddings in each record label")
    if text_embed.ndim == 2:
        text_embed = text_embed.unsqueeze(0)
    if repeat > 1:
        text_embed = text_embed.repeat(repeat, 1, 1)

    text_mask = sample["label"].get("text_embed_mask")
    if text_mask is not None and text_mask.ndim == 1:
        text_mask = text_mask.unsqueeze(0)
    if text_mask is not None and repeat > 1:
        text_mask = text_mask.repeat(repeat, 1)

    text_embed = text_embed.to(device)
    text_mask = None if text_mask is None else text_mask.to(device)
    return text_embed, text_mask


def sample_patient_tensor(sample: dict, device: torch.device, repeat: int = 1):
    """Build the normalized patient-information tensor for a serialized record."""
    sex_value = sample["label"]["sex"]
    if isinstance(sex_value, str):
        sex_value = sex_to_binary(sex_value)

    pat_info = build_patient_info_tensor(
        normalize=True,
        add_token=False,
        hr=torch.tensor([sample["label"]["hr"]], device=device),
        age=torch.tensor([sample["label"]["age"]], device=device),
        sex=torch.tensor([sex_value], device=device),
    )
    if repeat > 1:
        pat_info = pat_info.repeat(repeat, 1)
    return pat_info


def sample_latent_batch(sample: dict, device: torch.device, repeat: int = 1) -> torch.Tensor:
    """Return a sample's latent tensor in channel-first diffusion layout."""
    latent = sample["data"]
    if latent.ndim == 2:
        latent = latent.unsqueeze(0)
    if repeat > 1:
        latent = latent.repeat(repeat, 1, 1)
    return latent.to(device=device, dtype=torch.float32)


def extract_record_feature(
    sample: dict,
    feature_space: str,
    ibe_model,
    device: torch.device,
    use_amp: bool = False,
) -> torch.Tensor:
    """Extract either a flattened latent feature or an IBE feature for one record."""
    normalized_space = feature_space.lower()
    if normalized_space == "latent":
        return sample["data"].reshape(-1).to(dtype=torch.float32)
    if normalized_space != "ibe":
        raise ValueError(f"Unsupported privacy feature space: {feature_space}")

    latent = sample_latent_batch(sample, device).transpose(2, 1)
    text_embed, text_mask = sample_text_tensors(sample, device)
    pat_info = sample_patient_tensor(sample, device)
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp and device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad():
        with amp_context:
            feature = ibe_model.extract_features(latent, text_embed, text_mask, pat_info, reduce=True)
    return feature.squeeze(0).cpu()


def extract_pool_features(
    latents: torch.Tensor,
    conditioning_sample: dict,
    feature_space: str,
    ibe_model,
    device: torch.device,
    use_amp: bool = False,
) -> torch.Tensor:
    """Project a generated latent pool into the feature space used by an attack."""
    normalized_space = feature_space.lower()
    if normalized_space == "latent":
        return latents.reshape(latents.shape[0], -1).to(dtype=torch.float32)
    if normalized_space != "ibe":
        raise ValueError(f"Unsupported privacy feature space: {feature_space}")

    latents = latents.to(device=device, dtype=torch.float32)
    text_embed, text_mask = sample_text_tensors(conditioning_sample, device, repeat=latents.shape[0])
    pat_info = sample_patient_tensor(conditioning_sample, device, repeat=latents.shape[0])
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp and device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad():
        with amp_context:
            features = ibe_model.extract_features(latents.transpose(2, 1), text_embed, text_mask, pat_info, reduce=True)
    return features.cpu()
