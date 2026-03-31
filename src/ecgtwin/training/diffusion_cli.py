"""CLI-facing diffusion training workflow."""

from pathlib import Path

import torch
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader

from ecgtwin.config import load_config, resolve_serialized_data_path
from ecgtwin.core.logging import configure_logger
from ecgtwin.data.collate import paired_ecg_collate_fn
from ecgtwin.data.datasets import PairedECGDataset
from ecgtwin.models.factory import build_noise_predictor
from ecgtwin.models.ib_extractor import IBExtractor
from ecgtwin.models.vae_model import VAE_Decoder
from ecgtwin.training.diffusion import train_diffusion_model


def _hyper_params(cfg):
    """Translate the config tree into the legacy hyperparameter dict shape."""
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
        "ibe": {
            "embed_dim": cfg.MODEL.IBE.EMBED_DIM,
            "num_heads": cfg.MODEL.IBE.NUM_HEADS,
            "ff_hidden_size": cfg.MODEL.IBE.FF_HIDDEN_SIZE,
            "num_layers": cfg.MODEL.IBE.NUM_LAYERS,
            "text_embed_dim": cfg.MODEL.IBE.TEXT_EMBED_DIM,
            "patient_info_size": cfg.MODEL.IBE.PATIENT_INFO_SIZE,
        },
    }


def _meta(cfg):
    """Build the metadata structure consumed by the training loop."""
    return {
        "device": cfg.SYSTEM.DEVICE,
        "mix": cfg.MODEL.MIX_TEXT,
        "model_type": cfg.MODEL.NAME,
        "base_vector_mode": cfg.MODEL.BASE_VECTOR.MODE,
        "base_vector_noise_std": cfg.MODEL.BASE_VECTOR.NOISE_STD,
        "base_vector_mask_prob": cfg.MODEL.BASE_VECTOR.MASK_PROB,
    }


def _next_experiment_dir(root: Path, experiment_name: str) -> Path:
    """Allocate the next numbered experiment directory for a workflow."""
    root.mkdir(parents=True, exist_ok=True)
    existing_indices = []
    for item in root.iterdir():
        if item.is_dir() and item.name.startswith(f"{experiment_name}_"):
            try:
                existing_indices.append(int(item.name.split("_")[-1]))
            except ValueError:
                continue
    next_index = max(existing_indices, default=0) + 1
    return root / f"{experiment_name}_{next_index}"


def run(config_path, overrides):
    """Execute diffusion training from a config file and optional overrides."""
    cfg = load_config(config_path, overrides)
    hyper_params = _hyper_params(cfg)
    meta = _meta(cfg)

    save_dir = _next_experiment_dir(Path(cfg.PATHS.CHECKPOINTS_DIR), cfg.MODEL.EXP_NAME)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(cfg.MODEL.EXP_NAME, save_dir / "train.log")
    logger.info(meta)
    logger.info(hyper_params)

    train_dataset = PairedECGDataset(str(resolve_serialized_data_path(cfg, cfg.DATA.DATASET_PATH)))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.DATA.SHUFFLE,
        collate_fn=paired_ecg_collate_fn,
    )

    n_channels = 4 if cfg.MODEL.USE_VAE_LATENT else 12
    noise_predictor = build_noise_predictor(cfg.MODEL.NAME, n_channels, hyper_params)
    diffused_model = DDPMScheduler(
        num_train_timesteps=cfg.DIFFUSION.NUM_TRAIN_STEPS,
        beta_start=cfg.DIFFUSION.BETA_START,
        beta_end=cfg.DIFFUSION.BETA_END,
    )

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

    decoder = None
    if not cfg.MODEL.USE_VAE_LATENT:
        decoder = VAE_Decoder()
        vae_checkpoint = torch.load(cfg.CHECKPOINTS.VAE_PATH, map_location="cpu")
        decoder.load_state_dict(vae_checkpoint["decoder"])

    train_diffusion_model(
        meta=meta,
        save_weights_path=str(save_dir),
        dataloader=train_dataloader,
        diffused_model=diffused_model,
        ibe_model=ibe_model,
        decoder=decoder,
        noise_predictor=noise_predictor,
        h_=hyper_params,
        logger=logger,
    )

    with open(save_dir / "resolved_config.yaml", "w", encoding="utf-8") as handle:
        handle.write(cfg.dump())
