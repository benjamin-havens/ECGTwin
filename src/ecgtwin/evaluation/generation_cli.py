"""CLI entrypoints for maintained generation evaluation workflows."""

from pathlib import Path

from ecgtwin.config import load_config
from ecgtwin.evaluation.generation import (
    build_pair_generation_tasks,
    evaluate_generation,
    write_generation_manifest,
    write_pair_generation_artifacts,
)
from ecgtwin.inference.lightning_predict import run_generation_tasks


def run_generate_batch(config_path, overrides):
    """Generate a batch-evaluation artifact set from a paired dataset."""
    cfg = load_config(config_path, overrides)
    output_root = Path(cfg.EVAL.GENERATION.OUTPUT_DIR)
    summary = run_generation_tasks(
        cfg,
        build_pair_generation_tasks(cfg),
        scope="generation",
        output_root=output_root,
        task_handler=write_pair_generation_artifacts,
    )
    manifest_path = write_generation_manifest(cfg, output_dir=output_root)
    return {"output_dir": str(output_root), "manifest_path": str(manifest_path), **summary}


def run_evaluate(config_path, overrides):
    """Evaluate a generated batch artifact set."""
    cfg = load_config(config_path, overrides)
    return evaluate_generation(cfg)
