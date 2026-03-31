"""Helpers for loading and resolving ECGTwin configuration files."""

from pathlib import Path
from typing import Iterable

from .defaults import get_cfg_defaults


def _normalize_config_paths(config_path: str | Path | Iterable[str | Path] | None) -> list[str]:
    """Normalize config-path inputs into an ordered list of filesystem paths."""
    if config_path is None:
        return []
    if isinstance(config_path, (str, Path)):
        return [str(config_path)]
    return [str(path) for path in config_path]


def load_config(config_path: str | Path | Iterable[str | Path] | None = None, overrides: list[str] | None = None):
    """Load defaults, merge one or more YAML files in order, then merge CLI overrides."""
    cfg = get_cfg_defaults()
    for path in _normalize_config_paths(config_path):
        cfg.merge_from_file(path)
    if overrides:
        cfg.merge_from_list(overrides)
    cfg.freeze()
    return cfg


def resolve_path(path_like: str | Path) -> Path:
    """Expand and resolve a filesystem path-like value."""
    return Path(path_like).expanduser().resolve()


def resolve_serialized_data_path(cfg, path_like: str | Path) -> Path:
    """Resolve serialized dataset paths relative to the configured serialized-data root."""
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return (Path(cfg.PATHS.SERIALIZED_DATA_ROOT).expanduser() / path).resolve()
