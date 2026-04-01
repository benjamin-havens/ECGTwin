"""Shared Lightning execution for maintained inference-generation workflows."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ecgtwin.core.runtime_env import configure_runtime_environment, configure_torch_runtime
from ecgtwin.evaluation.runtime import load_generation_runtime
from ecgtwin.inference.generation import run_infer_task
from ecgtwin.inference.task_datasets import ListTaskDataset, task_collate_fn

configure_runtime_environment()

try:  # pragma: no cover - exercised indirectly in environments with Lightning
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover - exercised indirectly in environments without Lightning
    pl = None


def _require_lightning():
    if pl is None:
        raise ImportError("pytorch_lightning is required for inference-generation commands")
    return pl


def _device_index_from_string(device_string: str) -> int | None:
    normalized = str(device_string).lower()
    if not normalized.startswith("cuda"):
        return None
    if ":" not in normalized:
        return 0
    return int(normalized.split(":", maxsplit=1)[1])


def resolve_execution_gpu_ids(cfg, scope: str) -> list[int]:
    """Resolve the GPU ids to use for one inference-generation surface."""
    if not torch.cuda.is_available():
        return []
    explicit_gpu_ids = list(getattr(cfg.EXECUTION, "GPU_IDS", []))
    if explicit_gpu_ids:
        return [int(gpu_id) for gpu_id in explicit_gpu_ids]
    if scope == "generation":
        legacy_ids = list(getattr(cfg.EVAL.GENERATION, "GPU_IDS", []))
        if legacy_ids:
            return [int(gpu_id) for gpu_id in legacy_ids]
    if scope == "privacy":
        legacy_ids = list(getattr(cfg.PRIVACY, "GPU_IDS", []))
        if legacy_ids:
            return [int(gpu_id) for gpu_id in legacy_ids]
    if scope == "pecg":
        pecg_index = _device_index_from_string(cfg.APPS.PECG_MONITOR.GPU_DEVICE)
        if pecg_index is not None:
            return [pecg_index]
    system_index = _device_index_from_string(cfg.SYSTEM.DEVICE)
    return [] if system_index is None else [system_index]


def task_batch_size(cfg, scope: str) -> int:
    """Resolve the number of task dictionaries to process per predict step."""
    if scope == "generation":
        return max(int(cfg.EVAL.GENERATION.BATCH_SIZE), 1)
    return max(int(cfg.EXECUTION.TASK_BATCH_SIZE), 1)


def build_predict_trainer(cfg, output_root: Path, scope: str):
    """Build a Lightning trainer configured for inference-generation predict."""
    lightning = _require_lightning()
    configure_torch_runtime(cfg)
    gpu_ids = resolve_execution_gpu_ids(cfg, scope)
    if torch.cuda.is_available() and gpu_ids:
        accelerator = "gpu"
        devices = gpu_ids
        strategy = cfg.EXECUTION.STRATEGY if len(gpu_ids) > 1 else "auto"
        precision = "16-mixed" if cfg.SYSTEM.AMP else 32
    else:
        accelerator = "cpu"
        devices = 1
        strategy = "auto"
        precision = 32
    return lightning.Trainer(
        default_root_dir=str(output_root),
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=bool(cfg.EXECUTION.ENABLE_PROGRESS_BAR),
        precision=precision,
        inference_mode=True,
    )


if pl is not None:  # pragma: no branch
    class GenerationPredictModule(pl.LightningModule):
        """Predict-only Lightning module that writes generation artifacts on each rank."""

        def __init__(self, cfg, scope: str, task_handler):
            super().__init__()
            self.cfg = cfg
            self.scope = scope
            self.task_handler = task_handler
            self.runtime = None

        def configure_optimizers(self):
            return None

        def _runtime_flags(self) -> tuple[bool, bool]:
            include_decoder = self.scope in {"generation", "infer"}
            include_text_encoder = self.scope in {"pecg", "infer"}
            return include_decoder, include_text_encoder

        def _ensure_runtime(self) -> None:
            if self.runtime is not None:
                return
            include_decoder, include_text_encoder = self._runtime_flags()
            self.runtime = load_generation_runtime(
                self.cfg,
                device_override=self.device,
                include_decoder=include_decoder,
                include_text_encoder=include_text_encoder,
            )

        def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
            self._ensure_runtime()
            return [self.task_handler(task, self.runtime, self.cfg) for task in batch]
else:
    class GenerationPredictModule:  # pragma: no cover - runtime guard only
        """Placeholder that raises a clear error when Lightning is unavailable."""

        def __init__(self, *args, **kwargs):
            _require_lightning()


def run_generation_tasks(cfg, tasks: list[dict], scope: str, task_handler, output_root: Path):
    """Execute one generation surface through Lightning predict."""
    if not tasks:
        return {"task_count": 0, "devices": []}
    dataset = ListTaskDataset(tasks)
    dataloader = DataLoader(
        dataset,
        batch_size=task_batch_size(cfg, scope),
        shuffle=False,
        num_workers=0,
        collate_fn=task_collate_fn,
    )
    trainer = build_predict_trainer(cfg, output_root=output_root, scope=scope)
    module = GenerationPredictModule(cfg, scope=scope, task_handler=task_handler)
    trainer.predict(module, dataloaders=dataloader, return_predictions=False)
    return {
        "task_count": len(tasks),
        "devices": [f"cuda:{gpu_id}" for gpu_id in resolve_execution_gpu_ids(cfg, scope)] if torch.cuda.is_available() else ["cpu"],
    }
