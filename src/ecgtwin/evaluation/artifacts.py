"""Artifact manifest helpers for reproducibility workflows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def json_ready(value):
    """Recursively convert repo payloads into JSON-serializable Python types."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value

def write_manifest(output_path: Path, payload: dict) -> Path:
    """Persist a JSON manifest and ensure its parent directory exists."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def read_manifest(input_path: Path) -> dict:
    """Load a JSON manifest, returning an empty payload when absent."""
    if not input_path.exists():
        return {}
    return json.loads(input_path.read_text(encoding="utf-8"))
