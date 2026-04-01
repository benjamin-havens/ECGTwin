"""CLI-facing Lightning workflow for diffusion training."""

from pathlib import Path

import torch
from diffusers import DDPMScheduler
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from ecgtwin.config import load_config, resolve_serialized_data_path
from ecgtwin.core.logging import configure_logger
from ecgtwin.data.collate import paired_ecg_collate_fn
from ecgtwin.data.datasets import PairedECGDataset
from ecgtwin.models.conditioner import conditioner_hparams, load_conditioner
from ecgtwin.models.factory import build_noise_predictor
from ecgtwin.models.vae_model import VAE_Decoder
from ecgtwin.training.diffusion_lightning import DiffusionTrainingModule
from ecgtwin.training.lightning_common import (
    LightningMetricLogger,
    StateDictExportCallback,
    build_trainer,
    next_experiment_dir,
    seed_everything,
    write_resolved_config,
)


def _hyper_params(cfg):
    """Translate the config tree into the model-factory hyperparameter format."""
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


def run(config_path, overrides):
    """Execute diffusion training from config using Lightning."""
    cfg = load_config(config_path, overrides)
    seed_everything(cfg.SYSTEM.SEED)
    hyper_params = _hyper_params(cfg)

    save_dir = next_experiment_dir(Path(cfg.PATHS.CHECKPOINTS_DIR), cfg.MODEL.EXP_NAME)
    save_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(save_dir, cfg)
    logger = configure_logger(cfg.MODEL.EXP_NAME, save_dir / "train.log")
    logger.info(cfg.dump())

    train_dataset = PairedECGDataset(str(resolve_serialized_data_path(cfg, cfg.DATA.DATASET_PATH)))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.DATA.SHUFFLE,
        num_workers=cfg.SYSTEM.NUM_WORKERS,
        pin_memory=cfg.SYSTEM.PIN_MEMORY,
        collate_fn=paired_ecg_collate_fn,
    )

    n_channels = 4 if cfg.MODEL.USE_VAE_LATENT else 12
    noise_predictor = build_noise_predictor(cfg.MODEL.NAME, n_channels, hyper_params)
    diffused_model = DDPMScheduler(
        num_train_timesteps=cfg.DIFFUSION.NUM_TRAIN_STEPS,
        beta_start=cfg.DIFFUSION.BETA_START,
        beta_end=cfg.DIFFUSION.BETA_END,
    )
    conditioner = load_conditioner(cfg, map_location="cpu")

    decoder = None
    if not cfg.MODEL.USE_VAE_LATENT:
        decoder = VAE_Decoder()
        vae_checkpoint = torch.load(cfg.CHECKPOINTS.VAE_PATH, map_location="cpu")
        decoder.load_state_dict(vae_checkpoint["decoder"])

    module = DiffusionTrainingModule(cfg, noise_predictor, diffused_model, conditioner, decoder=decoder)
    callbacks = [
        LightningMetricLogger(logger, ["train_loss"]),
        ModelCheckpoint(
            dirpath=str(save_dir),
            filename=f"{cfg.MODEL.NAME}" + "-{epoch:02d}-{train_loss:.4f}",
            monitor="train_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        StateDictExportCallback(
            module_attr="noise_predictor",
            best_path=save_dir / f"{cfg.MODEL.NAME}_best.pth",
            monitor="train_loss",
            mode="min",
            every_n_epochs=50,
            epoch_filename_pattern=f"{cfg.MODEL.NAME}" + "_{epoch}.pth",
            trigger_stage="train",
        ),
    ]
    trainer = build_trainer(cfg, save_dir, callbacks=callbacks)
    trainer.fit(module, train_dataloaders=train_dataloader)
    return {"save_dir": str(save_dir), "checkpoint_path": str(save_dir / f"{cfg.MODEL.NAME}_best.pth")}
