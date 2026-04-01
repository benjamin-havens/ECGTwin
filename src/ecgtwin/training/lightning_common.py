"""Shared Lightning helpers for ECGTwin training workflows."""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities import rank_zero_only

from ecgtwin.core.runtime_env import configure_torch_runtime


def next_experiment_dir(root: Path, experiment_name: str) -> Path:
    """Allocate the next numbered experiment directory for a workflow."""
    root.mkdir(parents=True, exist_ok=True)
    existing_indices = []
    for item in root.iterdir():
        if item.is_dir() and item.name.startswith(f"{experiment_name}_"):
            try:
                existing_indices.append(int(item.name.split("_")[-1]))
            except ValueError:
                continue
    return root / f"{experiment_name}_{max(existing_indices, default=0) + 1}"


def write_resolved_config(save_dir: Path, cfg) -> None:
    """Persist the resolved YACS config beside trainer outputs."""
    with open(save_dir / "resolved_config.yaml", "w", encoding="utf-8") as handle:
        handle.write(cfg.dump())


def seed_everything(seed: int) -> None:
    """Seed the training process using Lightning's worker-aware helper."""
    pl.seed_everything(seed, workers=True)


def _lightning_device(device_string: str) -> tuple[str, int | list[int]]:
    normalized = device_string.lower()
    if normalized.startswith("cuda"):
        if ":" in normalized:
            return "gpu", [int(normalized.split(":", maxsplit=1)[1])]
        return "gpu", 1
    return "cpu", 1


def build_trainer(
    cfg,
    save_dir: Path,
    callbacks: list[Callback] | None = None,
    max_epochs: int | None = None,
    accumulate_grad_batches: int = 1,
) -> pl.Trainer:
    """Build a single-node Lightning trainer aligned with repo config."""
    configure_torch_runtime(cfg)
    accelerator, devices = _lightning_device(cfg.SYSTEM.DEVICE if torch.cuda.is_available() else "cpu")
    precision = "16-mixed" if cfg.SYSTEM.AMP and accelerator == "gpu" else 32
    logger = CSVLogger(save_dir=str(save_dir), name="metrics")
    return pl.Trainer(
        default_root_dir=str(save_dir),
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs or cfg.TRAIN.EPOCHS,
        accumulate_grad_batches=accumulate_grad_batches,
        precision=precision,
        callbacks=callbacks or [],
        logger=logger,
        log_every_n_steps=10,
    )


class LightningMetricLogger(Callback):
    """Mirror key Lightning metrics into the repo's standard train.log file."""

    def __init__(self, logger, metric_names: list[str]):
        super().__init__()
        self.logger = logger
        self.metric_names = metric_names

    @staticmethod
    def _metric_value(value) -> float | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def _emit_metrics(self, trainer: pl.Trainer, stage: str) -> None:
        metrics = []
        for metric_name in self.metric_names:
            metric_value = self._metric_value(trainer.callback_metrics.get(metric_name))
            if metric_value is not None:
                metrics.append(f"{metric_name}={metric_value:.6f}")
        if metrics:
            self.logger.info("epoch=%s stage=%s %s", trainer.current_epoch + 1, stage, " ".join(metrics))

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._emit_metrics(trainer, "train")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        self._emit_metrics(trainer, "val")


class StateDictExportCallback(Callback):
    """Export a selected submodule's state dict in the legacy `.pth` format."""

    def __init__(
        self,
        module_attr: str,
        best_path: Path,
        monitor: str,
        mode: str,
        every_n_epochs: int = 0,
        epoch_filename_pattern: str | None = None,
        trigger_stage: str = "validation",
    ):
        super().__init__()
        self.module_attr = module_attr
        self.best_path = best_path
        self.monitor = monitor
        self.mode = mode
        self.every_n_epochs = every_n_epochs
        self.epoch_filename_pattern = epoch_filename_pattern
        self.trigger_stage = trigger_stage
        self.best_value = None

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "max":
            return value > self.best_value
        return value < self.best_value

    @staticmethod
    def _metric_value(value) -> float | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    @rank_zero_only
    def _save_state_dict(self, trainer: pl.Trainer, pl_module: pl.LightningModule, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        module = getattr(pl_module, self.module_attr)
        torch.save(module.state_dict(), path)

    def _maybe_save_best(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metric_value = self._metric_value(trainer.callback_metrics.get(self.monitor))
        if metric_value is None or not self._is_better(metric_value):
            return
        self.best_value = metric_value
        self._save_state_dict(trainer, pl_module, self.best_path)

    def _maybe_save_periodic(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.every_n_epochs or not self.epoch_filename_pattern:
            return
        epoch_index = trainer.current_epoch + 1
        if epoch_index % self.every_n_epochs != 0:
            return
        self._save_state_dict(
            trainer,
            pl_module,
            self.best_path.parent / self.epoch_filename_pattern.format(epoch=epoch_index),
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.trigger_stage != "validation" or trainer.sanity_checking:
            return
        self._maybe_save_best(trainer, pl_module)
        self._maybe_save_periodic(trainer, pl_module)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.trigger_stage != "train":
            return
        self._maybe_save_best(trainer, pl_module)
        self._maybe_save_periodic(trainer, pl_module)
