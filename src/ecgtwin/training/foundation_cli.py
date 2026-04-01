"""CLI-facing Lightning workflow for foundation-conditioner training."""

from pathlib import Path

from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from ecgtwin.config import load_config, resolve_serialized_data_path
from ecgtwin.core.logging import configure_logger
from ecgtwin.data.collate import paired_ecg_collate_fn
from ecgtwin.data.datasets import PairedECGDataset
from ecgtwin.training.foundation import FoundationJEPAModule
from ecgtwin.training.lightning_common import (
    LightningMetricLogger,
    StateDictExportCallback,
    build_trainer,
    next_experiment_dir,
    seed_everything,
    write_resolved_config,
)


def run(config_path, overrides):
    """Execute JEPA-style foundation-conditioner training from config."""
    cfg = load_config(config_path, overrides)
    seed_everything(cfg.SYSTEM.SEED)

    save_dir = next_experiment_dir(Path(cfg.PATHS.CHECKPOINTS_DIR), "foundation")
    save_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(save_dir, cfg)
    logger = configure_logger("foundation", save_dir / "train.log")
    logger.info(cfg.dump())

    train_dataset = PairedECGDataset(str(resolve_serialized_data_path(cfg, cfg.DATA.DATASET_PATH)))
    test_dataset = PairedECGDataset(
        str(resolve_serialized_data_path(cfg, cfg.DATA.TEST_DATASET_PATH or cfg.DATA.DATASET_PATH))
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.MINI_BATCH_SIZE,
        shuffle=cfg.DATA.SHUFFLE,
        num_workers=cfg.SYSTEM.NUM_WORKERS,
        pin_memory=cfg.SYSTEM.PIN_MEMORY,
        collate_fn=paired_ecg_collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.TRAIN.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.SYSTEM.NUM_WORKERS,
        pin_memory=cfg.SYSTEM.PIN_MEMORY,
        collate_fn=paired_ecg_collate_fn,
    )

    module = FoundationJEPAModule(cfg)
    resume_checkpoint = cfg.TRAIN.LOAD_PRETRAIN if cfg.TRAIN.LOAD_PRETRAIN.endswith(".ckpt") else None
    if cfg.TRAIN.LOAD_PRETRAIN and resume_checkpoint is None:
        module.load_runtime_teacher_weights(cfg.TRAIN.LOAD_PRETRAIN)

    callbacks = [
        LightningMetricLogger(logger, ["train_loss", "train_token_loss", "train_global_loss", "val_alignment"]),
        ModelCheckpoint(
            dirpath=str(save_dir),
            filename="foundation-{epoch:02d}-{val_alignment:.4f}",
            monitor="val_alignment",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        StateDictExportCallback(
            module_attr="teacher",
            best_path=save_dir / "conditioner_best.pth",
            monitor="val_alignment",
            mode="max",
            every_n_epochs=10,
            epoch_filename_pattern="conditioner_ep{epoch}.pth",
            trigger_stage="validation",
        ),
    ]
    trainer = build_trainer(
        cfg,
        save_dir,
        callbacks=callbacks,
        accumulate_grad_batches=max(cfg.TRAIN.BATCH_SIZE // cfg.TRAIN.MINI_BATCH_SIZE, 1),
    )
    trainer.fit(module, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader, ckpt_path=resume_checkpoint)
    return {"save_dir": str(save_dir), "checkpoint_path": str(save_dir / "conditioner_best.pth")}
