"""CLI entrypoint for maintained ECGTwin inference generation."""

from pathlib import Path

from ecgtwin.config import load_config
from ecgtwin.inference.generation import build_infer_tasks, run_infer_task
from ecgtwin.inference.lightning_predict import run_generation_tasks


def run(config_path, overrides):
    """Generate ECG samples for one configured inference request."""
    cfg = load_config(config_path, overrides)
    output_root = Path(cfg.INFERENCE.SAVE_SAMPLE_PATH)
    summary = run_generation_tasks(
        cfg,
        build_infer_tasks(cfg),
        scope="infer",
        output_root=output_root,
        task_handler=run_infer_task,
    )
    return {"output_dir": str(output_root), **summary}
