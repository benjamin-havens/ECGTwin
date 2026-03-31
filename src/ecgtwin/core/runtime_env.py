"""Process-level runtime environment guards for CLI workflows."""

from __future__ import annotations

import os


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
