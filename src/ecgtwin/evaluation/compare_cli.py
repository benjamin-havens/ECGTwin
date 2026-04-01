"""CLI entrypoint for maintained run comparison reporting."""

from ecgtwin.config import load_config
from ecgtwin.evaluation.compare import compare_runs


def run(config_path, overrides):
    """Build a paper-aware comparison bundle from baseline and candidate run roots."""
    cfg = load_config(config_path, overrides)
    return compare_runs(cfg)
