"""CLI entrypoint for maintained personalization evaluation."""

from ecgtwin.config import load_config
from ecgtwin.evaluation.personalization import evaluate_personalization


def run(config_path, overrides):
    """Compute personalization metrics and figure exports."""
    cfg = load_config(config_path, overrides)
    return evaluate_personalization(cfg)
