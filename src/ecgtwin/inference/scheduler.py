"""Inference-time diffusion scheduler construction helpers."""

from __future__ import annotations

from diffusers import DDPMScheduler, DDIMScheduler


def build_inference_scheduler(cfg):
    """Construct the configured inference-time scheduler and set its sampling steps."""
    common_kwargs = {
        "num_train_timesteps": cfg.DIFFUSION.NUM_TRAIN_STEPS,
        "beta_start": cfg.DIFFUSION.BETA_START,
        "beta_end": cfg.DIFFUSION.BETA_END,
    }
    sampler = str(cfg.DIFFUSION.SAMPLER).lower()
    if sampler == "ddpm":
        scheduler = DDPMScheduler(**common_kwargs)
    elif sampler == "ddim":
        scheduler = DDIMScheduler(**common_kwargs, clip_sample=False)
    else:
        raise ValueError(f"Unsupported diffusion inference sampler: {cfg.DIFFUSION.SAMPLER}")

    scheduler.set_timesteps(cfg.DIFFUSION.INFERENCE_TIMESTEP)
    return scheduler
