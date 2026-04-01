"""Runtime loading helpers shared by evaluation workflows."""

from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer

from ecgtwin.config import load_config
from ecgtwin.inference.scheduler import build_inference_scheduler
from ecgtwin.models.clip_model import CLIP
from ecgtwin.models.conditioner import conditioner_hparams, load_conditioner
from ecgtwin.models.factory import build_noise_predictor
from ecgtwin.models.vae_model import VAE_Decoder


def diffusion_hyper_params(cfg):
    """Translate config into the noise-predictor hyperparameter format."""
    conditioner = conditioner_hparams(cfg)
    return {
        "epochs": cfg.TRAIN.EPOCHS,
        "lr": cfg.TRAIN.LR,
        "batch_size": cfg.TRAIN.BATCH_SIZE,
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
        "conditioner": {
            "embed_dim": conditioner["embed_dim"],
            "text_embed_dim": conditioner["text_embed_dim"],
            "patient_info_size": conditioner["patient_info_size"],
        },
    }


def resolve_runtime_device(cfg, device_override: str | torch.device | None = None) -> torch.device:
    """Resolve the torch device to use for inference runtimes."""
    if device_override is not None:
        return torch.device(device_override)
    return torch.device(cfg.SYSTEM.DEVICE if torch.cuda.is_available() else "cpu")


def load_text_embedding_runtime(device: torch.device):
    """Load the tokenizer and text-embedding backbone used for runtime prompting."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    embedding_model = AutoModel.from_pretrained(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        safe_serialization=True,
    )
    embedding_model.to(device)
    embedding_model.eval()
    return {"tokenizer": tokenizer, "embedding_model": embedding_model}


def load_generation_runtime(cfg, device_override: str | torch.device | None = None, include_decoder: bool = True, include_text_encoder: bool = False):
    """Load the trained modules required to generate ECG twins."""
    device = resolve_runtime_device(cfg, device_override=device_override)
    n_channels = 4 if cfg.MODEL.USE_VAE_LATENT else 12
    noise_predictor = build_noise_predictor(cfg.MODEL.NAME, n_channels, diffusion_hyper_params(cfg))
    noise_predictor.load_state_dict(torch.load(cfg.CHECKPOINTS.NOISE_PREDICTOR_PATH, map_location="cpu"))
    noise_predictor.to(device)
    noise_predictor.eval()

    conditioner_model = load_conditioner(cfg, map_location="cpu")
    conditioner_model.to(device)
    conditioner_model.eval()

    runtime = {
        "device": device,
        "noise_predictor": noise_predictor,
        "conditioner": conditioner_model,
        "scheduler": build_inference_scheduler(cfg),
    }
    if include_decoder:
        decoder = VAE_Decoder()
        vae_checkpoint = torch.load(cfg.CHECKPOINTS.VAE_PATH, map_location="cpu")
        decoder.load_state_dict(vae_checkpoint["decoder"])
        decoder.to(device)
        decoder.eval()
        runtime["decoder"] = decoder
    if include_text_encoder:
        runtime.update(load_text_embedding_runtime(device))
    return runtime


def load_clip_runtime(cfg, device_override: str | torch.device | None = None):
    """Load the CLIP feature extractor used by generation evaluation."""
    device = resolve_runtime_device(cfg, device_override=device_override)
    clip_model = CLIP(embed_dim=cfg.MODEL.CLIP.EMBED_DIM, text_embed_dim=cfg.MODEL.CLIP.TEXT_EMBED_DIM)
    clip_model.load_state_dict(torch.load(cfg.CHECKPOINTS.CLIP_PATH, map_location="cpu"))
    clip_model.to(device)
    clip_model.eval()
    return {"device": device, "clip_model": clip_model}


def load_config_and_runtime(config_path, overrides):
    """Convenience wrapper for loading config plus generation runtime."""
    cfg = load_config(config_path, overrides)
    return cfg, load_generation_runtime(cfg)
