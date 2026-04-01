"""Process-level runtime environment guards for CLI workflows."""

from __future__ import annotations

import os

import torch


def configure_runtime_environment() -> None:
    """Normalize BLAS/OpenMP settings for spawned worker processes.

    PyTorch multiprocessing workers can inherit an environment where
    ``MKL_THREADING_LAYER=INTEL`` conflicts with ``libgomp`` consumers loaded by
    PyTorch and related libraries. Favor the GNU threading layer so spawned
    workers start reliably on Linux CUDA nodes.
    """
    current_layer = os.environ.get("MKL_THREADING_LAYER", "").strip().upper()
    if current_layer in {"", "INTEL"}:
        os.environ["MKL_THREADING_LAYER"] = "GNU"


def configure_torch_runtime(cfg=None) -> None:
    """Apply runtime Torch backend settings from config when available."""
    if cfg is None:
        return
    precision = str(getattr(cfg.SYSTEM, "MATMUL_PRECISION", "default")).strip().lower()
    if precision in {"medium", "high", "highest"}:
        torch.set_float32_matmul_precision(precision)
