"""CLI-facing Lightning workflow for VAE training."""

from pathlib import Path

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from ecgtwin.config import load_config
from ecgtwin.core.logging import configure_logger
from ecgtwin.data.datasets import MIMIC_IV_ECG_Dataset
from ecgtwin.inference.rendering import save_ecg_plot
from ecgtwin.training.lightning_common import build_trainer, next_experiment_dir, seed_everything, write_resolved_config
from ecgtwin.training.vae import VAETrainingModule


def _build_dataset(cfg, usage: str):
    test_fold = None if int(cfg.DATA.TEST_FOLD) < 0 else int(cfg.DATA.TEST_FOLD)
    return MIMIC_IV_ECG_Dataset(
        cfg.PATHS.MIMIC_ROOT,
        usage=usage,
        num_folds=cfg.DATA.NUM_FOLDS,
        test_fold=test_fold,
        seed=cfg.SYSTEM.SEED,
        resample_length=cfg.DATA.RESAMPLE_LENGTH,
        demo_label=cfg.DATA.DEMO_LABEL,
        patients_csv_path=cfg.PATHS.PATIENTS_CSV,
        exclude_list_path=cfg.PATHS.EXCLUDE_LIST,
    )


def _save_validation_previews(module: VAETrainingModule, dataloader, save_dir: Path) -> None:
    if not module.cfg.MODEL.VAE.SAVE_RECONSTRUCTIONS:
        return
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        return
    signals, _ = batch
    device = module.device
    with torch.no_grad():
        reconstruction, _, _, _ = module(signals.to(device=device, dtype=torch.float32))
    lead_index = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]
    preview_dir = save_dir / "reconstructions"
    preview_dir.mkdir(parents=True, exist_ok=True)
    count = min(int(module.cfg.MODEL.VAE.RECONSTRUCTION_COUNT), reconstruction.shape[0])
    for index in range(count):
        save_ecg_plot(signals[index].numpy(), preview_dir / f"{index}_target.png", lead_index)
        save_ecg_plot(reconstruction[index].detach().cpu().numpy(), preview_dir / f"{index}_reconstruction.png", lead_index)


def run(config_path, overrides):
    """Execute raw-waveform VAE training from config."""
    cfg = load_config(config_path, overrides)
    seed_everything(cfg.SYSTEM.SEED)

    save_dir = next_experiment_dir(Path(cfg.PATHS.CHECKPOINTS_DIR), "vae")
    save_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(save_dir, cfg)
    logger = configure_logger("vae", save_dir / "train.log")
    logger.info(cfg.dump())

    train_dataset = _build_dataset(cfg, usage="train")
    val_dataset = _build_dataset(cfg, usage="test")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.MINI_BATCH_SIZE,
        shuffle=cfg.DATA.SHUFFLE,
        num_workers=cfg.SYSTEM.NUM_WORKERS,
        pin_memory=cfg.SYSTEM.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.TRAIN.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.SYSTEM.NUM_WORKERS,
        pin_memory=cfg.SYSTEM.PIN_MEMORY,
    )

    module = VAETrainingModule(cfg)
    callbacks = [
        ModelCheckpoint(
            dirpath=str(save_dir),
            filename="vae-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
    ]
    trainer = build_trainer(
        cfg,
        save_dir,
        callbacks=callbacks,
        accumulate_grad_batches=max(cfg.TRAIN.BATCH_SIZE // cfg.TRAIN.MINI_BATCH_SIZE, 1),
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_checkpoint = callbacks[0].best_model_path or callbacks[0].last_model_path
    best_module = VAETrainingModule.load_from_checkpoint(best_checkpoint, cfg=cfg)
    best_module.export_checkpoint(str(save_dir / "vae_model.pth"))
    _save_validation_previews(best_module, val_loader, save_dir)
    return {"save_dir": str(save_dir), "checkpoint_path": str(save_dir / "vae_model.pth")}
