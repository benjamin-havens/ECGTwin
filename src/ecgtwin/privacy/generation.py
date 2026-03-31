"""Synthetic-pool generation helpers for black-box privacy attacks."""

from __future__ import annotations

from pathlib import Path
from contextlib import nullcontext

import torch

from ecgtwin.inference.generation import ddpm_generation
from ecgtwin.inference.scheduler import build_inference_scheduler
from ecgtwin.models.base_vector import apply_base_vector_ablation
from ecgtwin.models.factory import build_noise_predictor
from ecgtwin.models.ib_extractor import IBExtractor

from .features import sample_latent_batch, sample_patient_tensor, sample_text_tensors


def _hyper_params(cfg):
    """Translate the config tree into the model-factory hyperparameter format."""
    return {
        "ddpm": {
            "num_train_steps": cfg.DIFFUSION.NUM_TRAIN_STEPS,
            "beta_start": cfg.DIFFUSION.BETA_START,
            "beta_end": cfg.DIFFUSION.BETA_END,
        },
        "dit": {
            "hidden_size": cfg.MODEL.DIT.HIDDEN_SIZE,
            "depth": cfg.MODEL.DIT.DEPTH,
            "num_heads": cfg.MODEL.DIT.NUM_HEADS,
            "patient_info_size": cfg.MODEL.DIT.PATIENT_INFO_SIZE,
        },
        "unet": {
            "kernel_size": cfg.MODEL.UNET.KERNEL_SIZE,
            "num_level": cfg.MODEL.UNET.NUM_LEVEL,
            "n_heads": cfg.MODEL.UNET.N_HEADS,
            "patient_info_size": cfg.MODEL.UNET.PATIENT_INFO_SIZE,
        },
        "ibe": {
            "embed_dim": cfg.MODEL.IBE.EMBED_DIM,
            "num_heads": cfg.MODEL.IBE.NUM_HEADS,
            "ff_hidden_size": cfg.MODEL.IBE.FF_HIDDEN_SIZE,
            "num_layers": cfg.MODEL.IBE.NUM_LAYERS,
            "text_embed_dim": cfg.MODEL.IBE.TEXT_EMBED_DIM,
            "patient_info_size": cfg.MODEL.IBE.PATIENT_INFO_SIZE,
        },
    }


def load_privacy_runtime(cfg):
    """Load the trained ECGTwin components needed for privacy scoring."""
    if not cfg.MODEL.USE_VAE_LATENT:
        raise NotImplementedError("Privacy-audit generation currently supports VAE-latent ECGTwin checkpoints only")

    device = torch.device(cfg.SYSTEM.DEVICE if torch.cuda.is_available() else "cpu")
    hyper_params = _hyper_params(cfg)

    noise_predictor = build_noise_predictor(cfg.MODEL.NAME, 4, hyper_params)
    noise_predictor.load_state_dict(torch.load(cfg.CHECKPOINTS.NOISE_PREDICTOR_PATH, map_location="cpu"))
    noise_predictor.to(device)
    noise_predictor.eval()

    diffused_model = build_inference_scheduler(cfg)

    ibe_model = IBExtractor(
        embed_dim=cfg.MODEL.IBE.EMBED_DIM,
        num_heads=cfg.MODEL.IBE.NUM_HEADS,
        ff_hidden_size=cfg.MODEL.IBE.FF_HIDDEN_SIZE,
        num_layers=cfg.MODEL.IBE.NUM_LAYERS,
        text_embed_dim=cfg.MODEL.IBE.TEXT_EMBED_DIM,
        patient_info_size=cfg.MODEL.IBE.PATIENT_INFO_SIZE,
        base_vector_mode=cfg.MODEL.BASE_VECTOR.MODE,
        base_vector_bottleneck_dim=cfg.MODEL.BASE_VECTOR.BOTTLENECK_DIM,
    )
    ibe_model.load_state_dict(torch.load(cfg.CHECKPOINTS.IBE_PATH, map_location="cpu"))
    ibe_model.to(device)
    ibe_model.eval()

    return {
        "device": device,
        "noise_predictor": noise_predictor,
        "diffused_model": diffused_model,
        "ibe_model": ibe_model,
    }


@torch.no_grad()
def generate_synthetic_latents(sample: dict, runtime: dict, cfg, num_samples: int | None = None) -> torch.Tensor:
    """Generate a synthetic latent pool conditioned on one candidate record."""
    device = runtime["device"]
    num_samples = num_samples or cfg.PRIVACY.SYNTHETIC_NUM_SAMPLES
    batch_size = cfg.PRIVACY.SYNTHETIC_BATCH_SIZE
    generated_batches = []
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if cfg.PRIVACY.USE_AMP and device.type == "cuda"
        else nullcontext()
    )

    for offset in range(0, num_samples, batch_size):
        current_batch = min(batch_size, num_samples - offset)
        with amp_context:
            ref_latent = sample_latent_batch(sample, device, repeat=current_batch).transpose(2, 1)
            ref_text, ref_mask = sample_text_tensors(sample, device, repeat=current_batch)
            ref_pat = sample_patient_tensor(sample, device, repeat=current_batch)
            base_vector = runtime["ibe_model"].extract_features(ref_latent, ref_text, ref_mask, ref_pat, reduce=True)
            base_vector = apply_base_vector_ablation(
                base_vector,
                mode=cfg.MODEL.BASE_VECTOR.MODE,
                noise_std=cfg.MODEL.BASE_VECTOR.NOISE_STD,
            )

            tar_text, tar_mask = sample_text_tensors(sample, device, repeat=current_batch)
            tar_pat = sample_patient_tensor(sample, device, repeat=current_batch)
            latent_batch = ddpm_generation(
                diffused_model=runtime["diffused_model"],
                noise_predictor=runtime["noise_predictor"],
                batch_size=current_batch,
                device=device,
                text_embed=tar_text,
                text_embed_mask=tar_mask,
                pat_info=tar_pat,
                base_vector=base_vector,
                progress_bar=False,
            )
        generated_batches.append(latent_batch.cpu())

    return torch.cat(generated_batches, dim=0)


def save_synthetic_pool(
    output_path: Path,
    sample: dict,
    record_id: str,
    cache_key: str,
    subset: str,
    latents: torch.Tensor,
) -> None:
    """Persist a synthetic latent pool and its metadata to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "record_id": record_id,
            "cache_key": cache_key,
            "subject_id": sample["label"].get("subject_id"),
            "subset": subset,
            "latent_gen": latents,
        },
        output_path,
    )
