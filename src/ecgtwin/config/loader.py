"""Helpers for loading and resolving ECGTwin configuration files."""

from pathlib import Path

from .defaults import get_cfg_defaults


def load_config(config_path: str | None = None, overrides: list[str] | None = None):
    """Load defaults, merge a YAML file, then merge any CLI overrides."""
    cfg = get_cfg_defaults()
    if config_path:
        cfg.merge_from_file(config_path)
    if overrides:
        cfg.merge_from_list(overrides)
    cfg.freeze()
    return cfg


def resolve_path(path_like: str | Path) -> Path:
    """Expand and resolve a filesystem path-like value."""
    return Path(path_like).expanduser().resolve()
