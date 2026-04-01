"""Top-level paper reproduction orchestrator for maintained ECGTwin workflows."""

from __future__ import annotations

from pathlib import Path

from ecgtwin.config import load_config, resolve_path, resolve_serialized_data_path
from ecgtwin.evaluation.artifacts import read_manifest, write_manifest


STAGE_DEPENDENCIES = {
    "vae": [],
    "preprocess": ["vae"],
    "text_embed": ["preprocess"],
    "pair": ["text_embed"],
    "foundation": ["pair"],
    "diffusion": ["pair", "foundation", "vae"],
    "clip": ["text_embed", "vae"],
    "generate_batch": ["pair", "foundation", "diffusion", "vae"],
    "privacy": ["text_embed", "foundation", "diffusion", "vae"],
    "pecg_generate": ["foundation", "diffusion", "vae"],
    "pecg_train": ["pecg_generate", "vae"],
    "pecg_test": ["pecg_train", "vae"],
    "compare": ["generate_batch"],
}


def _checkpoint_override_key_for_stage(cfg, stage: str) -> str | None:
    if stage == "vae":
        return "CHECKPOINTS.VAE_PATH"
    if stage == "foundation":
        return "CHECKPOINTS.CONDITIONER_PATH"
    if stage == "diffusion":
        return "CHECKPOINTS.NOISE_PREDICTOR_PATH"
    if stage == "clip":
        return "CHECKPOINTS.CLIP_PATH"
    return None


def _configured_checkpoint_path(cfg, stage: str) -> str:
    if stage == "foundation" and not getattr(cfg.CHECKPOINTS, "CONDITIONER_PATH", ""):
        return str(getattr(cfg.CHECKPOINTS, "IBE_PATH", ""))
    override_key = _checkpoint_override_key_for_stage(cfg, stage)
    if not override_key:
        return ""
    section, key = override_key.split(".", maxsplit=1)
    return str(getattr(getattr(cfg, section), key, ""))


def _existing_checkpoint_artifact(cfg, stage: str) -> dict | None:
    configured_stages = {str(item) for item in getattr(cfg.REPRO, "USE_EXISTING_MODEL_STAGES", [])}
    if stage not in configured_stages:
        return None
    checkpoint_path = _configured_checkpoint_path(cfg, stage)
    if not checkpoint_path:
        return None
    resolved_path = resolve_path(checkpoint_path)
    if not resolved_path.exists():
        return None
    payload = {
        "stage": stage,
        "status": "completed",
        "checkpoint_path": str(resolved_path),
        "source": "configured_checkpoint",
    }
    override_key = _checkpoint_override_key_for_stage(cfg, stage)
    if override_key:
        payload["checkpoint_override"] = [override_key, str(resolved_path)]
    return payload


def _resolve_stage_order(requested_stages: list[str]) -> list[str]:
    ordered = []
    temporary = set()
    permanent = set()

    def visit(stage: str) -> None:
        if stage in permanent:
            return
        if stage in temporary:
            raise ValueError(f"Cyclic stage dependency detected at {stage}")
        temporary.add(stage)
        for dependency in STAGE_DEPENDENCIES.get(stage, []):
            visit(dependency)
        temporary.remove(stage)
        permanent.add(stage)
        ordered.append(stage)

    for stage in requested_stages:
        visit(stage)
    return ordered


def _stage_dir(run_root: Path, stage: str) -> Path:
    return run_root / "stages" / stage


def _stage_manifest_path(run_root: Path, stage: str) -> Path:
    return _stage_dir(run_root, stage) / "manifest.json"


def _existing_stage_manifest(run_root: Path, stage: str):
    manifest_path = _stage_manifest_path(run_root, stage)
    if manifest_path.exists():
        return read_manifest(manifest_path)
    return None


def _write_stage_manifest(run_root: Path, stage: str, payload: dict) -> dict:
    stage_dir = _stage_dir(run_root, stage)
    stage_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(_stage_manifest_path(run_root, stage), payload)
    return payload


def _merge_overrides(base_overrides: list[str], *extra_groups: list[str]) -> list[str]:
    merged = list(base_overrides)
    for group in extra_groups:
        merged.extend(group)
    return merged


def _artifact_value(artifacts: dict[str, dict], stage: str, key: str, default: str = "") -> str:
    return str(artifacts.get(stage, {}).get(key, default))


def _normalize_config_paths(config_path) -> list[str]:
    if isinstance(config_path, (list, tuple)):
        return [str(path) for path in config_path]
    return [str(config_path)]


def _stage_config_paths(config_path, stage: str, conditioner_type: str) -> list[str]:
    config_paths = _normalize_config_paths(config_path)
    stage_defaults: list[str] = []
    if stage == "foundation":
        if conditioner_type == "ibe":
            stage_defaults.append("configs/experiments/ibe/base.yaml")
        else:
            stage_defaults.append("configs/experiments/foundation/base.yaml")
    elif stage == "diffusion":
        stage_defaults.append("configs/experiments/diffusion/dit_ecgtwin.yaml")
    elif stage == "clip":
        stage_defaults.append("configs/experiments/clip/base.yaml")
    return stage_defaults + config_paths


def _serialized_exists(cfg, path_like: str) -> bool:
    if not path_like:
        return False
    return resolve_serialized_data_path(cfg, path_like).exists()


def _pecg_inputs_available(cfg) -> bool:
    paths = [
        cfg.APPS.PECG_MONITOR.TEST_DATASET_PATH,
        cfg.APPS.PECG_MONITOR.TRAIN_DATASET_PATH,
        cfg.APPS.PECG_MONITOR.VAL_DATASET_PATH,
        cfg.APPS.PECG_MONITOR.TEST_CLASSIFIER_DATASET_PATH,
    ]
    return all(_serialized_exists(cfg, path) for path in paths)


def _common_checkpoint_overrides(artifacts: dict[str, dict]) -> list[str]:
    overrides = []
    if _artifact_value(artifacts, "vae", "checkpoint_path"):
        overrides.extend(["CHECKPOINTS.VAE_PATH", _artifact_value(artifacts, "vae", "checkpoint_path")])
    if _artifact_value(artifacts, "foundation", "checkpoint_path"):
        overrides.extend(["CHECKPOINTS.CONDITIONER_PATH", _artifact_value(artifacts, "foundation", "checkpoint_path")])
    if _artifact_value(artifacts, "diffusion", "checkpoint_path"):
        overrides.extend(["CHECKPOINTS.NOISE_PREDICTOR_PATH", _artifact_value(artifacts, "diffusion", "checkpoint_path")])
    if _artifact_value(artifacts, "clip", "checkpoint_path"):
        overrides.extend(["CHECKPOINTS.CLIP_PATH", _artifact_value(artifacts, "clip", "checkpoint_path")])
    return overrides


def _run_or_skip(run_root: Path, stage: str, skip_existing: bool, runner) -> dict:
    if skip_existing:
        existing = _existing_stage_manifest(run_root, stage)
        if existing is not None:
            return existing
    result = runner()
    result.setdefault("stage", stage)
    result.setdefault("status", "completed")
    return _write_stage_manifest(run_root, stage, result)


def run(config_path, overrides):
    """Execute a config-driven multi-stage reproduction run."""
    config_paths = _normalize_config_paths(config_path)
    cfg = load_config(config_paths, overrides)
    run_root = resolve_path(Path(cfg.REPRO.ROOT_DIR) / cfg.REPRO.RUN_NAME)
    run_root.mkdir(parents=True, exist_ok=True)

    requested_stages = list(cfg.REPRO.STAGES)
    ordered_stages = _resolve_stage_order(requested_stages)
    top_manifest = {
        "run_root": str(run_root),
        "requested_stages": requested_stages,
        "resolved_stage_order": ordered_stages,
        "dry_run": bool(cfg.REPRO.DRY_RUN),
        "stages": {},
    }
    if cfg.REPRO.DRY_RUN:
        write_manifest(run_root / "manifest.json", top_manifest)
        return {"output_dir": str(run_root), "manifest_path": str(run_root / "manifest.json")}

    artifacts = {}
    for stage in ordered_stages:
        stage_dir = _stage_dir(run_root, stage)
        stage_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_overrides = _common_checkpoint_overrides(artifacts)

        if stage == "vae":
            from ecgtwin.training.vae_cli import run as run_vae_train

            existing = _existing_checkpoint_artifact(cfg, stage)
            if existing is not None:
                artifacts[stage] = _write_stage_manifest(run_root, stage, existing)
            else:
                artifacts[stage] = _run_or_skip(
                    run_root,
                    stage,
                    bool(cfg.REPRO.SKIP_EXISTING),
                    lambda: run_vae_train(
                        config_path,
                        _merge_overrides(overrides, ["PATHS.CHECKPOINTS_DIR", str(run_root / "checkpoints" / "vae")]),
                    ),
                )
        elif stage == "preprocess":
            from ecgtwin.data.preprocess.vae_encoding import run as run_vae_encoding

            artifacts[stage] = _run_or_skip(
                run_root,
                stage,
                bool(cfg.REPRO.SKIP_EXISTING),
                lambda: run_vae_encoding(
                    config_path,
                    _merge_overrides(
                        overrides,
                        checkpoint_overrides,
                        ["PATHS.OUTPUT_DIR", str(run_root / "artifacts" / "preprocess")],
                    ),
                ),
            )
        elif stage == "text_embed":
            from ecgtwin.data.preprocess.store_text_embeddings import run as run_store_text_embeddings

            dataset_path = _artifact_value(artifacts, "preprocess", "dataset_path", cfg.DATA.DATASET_PATH)
            artifacts[stage] = _run_or_skip(
                run_root,
                stage,
                bool(cfg.REPRO.SKIP_EXISTING),
                lambda: run_store_text_embeddings(
                    config_path,
                    _merge_overrides(
                        overrides,
                        [
                            "DATA.DATASET_PATH",
                            dataset_path,
                            "DATA.TRAIN_DATASET_PATH",
                            str(run_root / "artifacts" / "text_embed" / "embedded.pt"),
                        ],
                    ),
                ),
            )
        elif stage == "pair":
            from ecgtwin.data.preprocess.pair_dataset import run as run_pair_dataset

            dataset_path = _artifact_value(artifacts, "text_embed", "dataset_path", cfg.DATA.DATASET_PATH)
            artifacts[stage] = _run_or_skip(
                run_root,
                stage,
                bool(cfg.REPRO.SKIP_EXISTING),
                lambda: run_pair_dataset(
                    config_path,
                    _merge_overrides(
                        overrides,
                        [
                            "DATA.DATASET_PATH",
                            dataset_path,
                            "DATA.TRAIN_DATASET_PATH",
                            str(run_root / "artifacts" / "pair" / "paired.pt"),
                        ],
                    ),
                ),
            )
        elif stage == "foundation":
            existing = _existing_checkpoint_artifact(cfg, stage)
            if existing is not None:
                artifacts[stage] = _write_stage_manifest(run_root, stage, existing)
            else:
                conditioner_type = cfg.MODEL.CONDITIONER.TYPE.lower()
                if conditioner_type == "ibe":
                    from ecgtwin.training.ibe_cli import run as run_conditioner_train

                    checkpoint_dir = run_root / "checkpoints" / "ibe"
                else:
                    from ecgtwin.training.foundation_cli import run as run_conditioner_train

                    checkpoint_dir = run_root / "checkpoints" / "foundation"

                artifacts[stage] = _run_or_skip(
                    run_root,
                    stage,
                    bool(cfg.REPRO.SKIP_EXISTING),
                    lambda: run_conditioner_train(
                        _stage_config_paths(config_paths, stage, conditioner_type),
                        _merge_overrides(
                            overrides,
                            checkpoint_overrides,
                            [
                                "PATHS.CHECKPOINTS_DIR",
                                str(checkpoint_dir),
                                "DATA.DATASET_PATH",
                                _artifact_value(artifacts, "pair", "dataset_path"),
                                "DATA.TEST_DATASET_PATH",
                                _artifact_value(artifacts, "pair", "dataset_path"),
                            ],
                        ),
                    ),
                )
        elif stage == "diffusion":
            from ecgtwin.training.diffusion_cli import run as run_diffusion

            existing = _existing_checkpoint_artifact(cfg, stage)
            if existing is not None:
                artifacts[stage] = _write_stage_manifest(run_root, stage, existing)
            else:
                artifacts[stage] = _run_or_skip(
                    run_root,
                    stage,
                    bool(cfg.REPRO.SKIP_EXISTING),
                    lambda: run_diffusion(
                        _stage_config_paths(config_paths, stage, cfg.MODEL.CONDITIONER.TYPE.lower()),
                        _merge_overrides(
                            overrides,
                            checkpoint_overrides,
                            [
                                "PATHS.CHECKPOINTS_DIR",
                                str(run_root / "checkpoints" / "diffusion"),
                                "DATA.DATASET_PATH",
                                _artifact_value(artifacts, "pair", "dataset_path"),
                            ],
                        ),
                    ),
                )
        elif stage == "clip":
            from ecgtwin.training.clip_cli import run as run_clip

            existing = _existing_checkpoint_artifact(cfg, stage)
            if existing is not None:
                artifacts[stage] = _write_stage_manifest(run_root, stage, existing)
            else:
                artifacts[stage] = _run_or_skip(
                    run_root,
                    stage,
                    bool(cfg.REPRO.SKIP_EXISTING),
                    lambda: run_clip(
                        _stage_config_paths(config_paths, stage, cfg.MODEL.CONDITIONER.TYPE.lower()),
                        _merge_overrides(
                            overrides,
                            checkpoint_overrides,
                            [
                                "PATHS.CHECKPOINTS_DIR",
                                str(run_root / "checkpoints" / "clip"),
                                "DATA.DATASET_PATH",
                                _artifact_value(artifacts, "text_embed", "dataset_path"),
                                "DATA.TEST_DATASET_PATH",
                                _artifact_value(artifacts, "text_embed", "dataset_path"),
                            ],
                        ),
                    ),
                )
        elif stage == "generate_batch":
            from ecgtwin.evaluation.generation_cli import run_generate_batch

            artifacts[stage] = _run_or_skip(
                run_root,
                stage,
                bool(cfg.REPRO.SKIP_EXISTING),
                lambda: run_generate_batch(
                    config_path,
                    _merge_overrides(
                        overrides,
                        checkpoint_overrides,
                        [
                            "EVAL.GENERATION.PAIR_DATASET_PATH",
                            _artifact_value(artifacts, "pair", "dataset_path"),
                            "EVAL.GENERATION.OUTPUT_DIR",
                            str(run_root / "evaluation" / "generation"),
                        ],
                    ),
                ),
            )
        elif stage == "privacy":
            if not cfg.PRIVACY.NONMEMBER_DATASET_PATH:
                artifacts[stage] = _write_stage_manifest(
                    run_root,
                    stage,
                    {"stage": stage, "status": "skipped", "reason": "PRIVACY.NONMEMBER_DATASET_PATH is not configured"},
                )
            else:
                from ecgtwin.privacy.cli import run_audit as run_privacy_audit

                artifacts[stage] = _run_or_skip(
                    run_root,
                    stage,
                    bool(cfg.REPRO.SKIP_EXISTING),
                    lambda: run_privacy_audit(
                        config_path,
                        _merge_overrides(
                            overrides,
                            checkpoint_overrides,
                            [
                                "PRIVACY.OUTPUT_DIR",
                                str(run_root / "evaluation" / "privacy"),
                                "PRIVACY.MEMBER_DATASET_PATH",
                                _artifact_value(artifacts, "text_embed", "dataset_path"),
                            ],
                        ),
                    ),
                )
        elif stage == "pecg_generate":
            if not _serialized_exists(cfg, cfg.APPS.PECG_MONITOR.TEST_DATASET_PATH):
                artifacts[stage] = _write_stage_manifest(
                    run_root,
                    stage,
                    {"stage": stage, "status": "skipped", "reason": "pECGMonitor test dataset is unavailable"},
                )
            else:
                from ecgtwin.apps.pecg_monitor.generation import run as run_pecg_generate

                artifacts[stage] = _run_or_skip(
                    run_root,
                    stage,
                    bool(cfg.REPRO.SKIP_EXISTING),
                    lambda: run_pecg_generate(
                        config_path,
                        _merge_overrides(
                            overrides,
                            checkpoint_overrides,
                            ["APPS.PECG_MONITOR.OUTPUT_DIR", str(run_root / "evaluation" / "pecg_monitor")],
                        ),
                    ),
                )
        elif stage == "pecg_train":
            if not _pecg_inputs_available(cfg):
                artifacts[stage] = _write_stage_manifest(
                    run_root,
                    stage,
                    {"stage": stage, "status": "skipped", "reason": "pECGMonitor classifier datasets are unavailable"},
                )
            else:
                from ecgtwin.apps.pecg_monitor.classifier_train import run as run_pecg_train

                artifacts[stage] = _run_or_skip(
                    run_root,
                    stage,
                    bool(cfg.REPRO.SKIP_EXISTING),
                    lambda: run_pecg_train(
                        config_path,
                        _merge_overrides(
                            overrides,
                            checkpoint_overrides,
                            ["APPS.PECG_MONITOR.OUTPUT_DIR", str(run_root / "evaluation" / "pecg_monitor")],
                        ),
                    ),
                )
        elif stage == "pecg_test":
            checkpoint_path = _artifact_value(artifacts, "pecg_train", "checkpoint_path")
            if not checkpoint_path:
                artifacts[stage] = _write_stage_manifest(
                    run_root,
                    stage,
                    {"stage": stage, "status": "skipped", "reason": "pECGMonitor classifier checkpoint is unavailable"},
                )
            else:
                from ecgtwin.apps.pecg_monitor.classifier_test import run as run_pecg_test

                artifacts[stage] = _run_or_skip(
                    run_root,
                    stage,
                    bool(cfg.REPRO.SKIP_EXISTING),
                    lambda: run_pecg_test(
                        config_path,
                        _merge_overrides(
                            overrides,
                            checkpoint_overrides,
                            [
                                "APPS.PECG_MONITOR.OUTPUT_DIR",
                                str(run_root / "evaluation" / "pecg_monitor"),
                                "TRAIN.LOAD_PRETRAIN",
                                checkpoint_path,
                            ],
                        ),
                    ),
                )
        elif stage == "compare":
            from ecgtwin.evaluation.compare_cli import run as run_compare
            from ecgtwin.evaluation.generation_cli import run_evaluate as run_evaluate_generation
            from ecgtwin.evaluation.personalization_cli import run as run_evaluate_personalization

            generation_eval = run_evaluate_generation(
                config_path,
                _merge_overrides(
                    overrides,
                    checkpoint_overrides,
                    [
                        "EVAL.GENERATION.OUTPUT_DIR",
                        _artifact_value(artifacts, "generate_batch", "output_dir"),
                    ],
                ),
            )
            personalization_eval = run_evaluate_personalization(
                config_path,
                _merge_overrides(
                    overrides,
                    checkpoint_overrides,
                    [
                        "EVAL.PERSONALIZATION.OUTPUT_DIR",
                        str(run_root / "evaluation" / "personalization"),
                        "EVAL.PERSONALIZATION.GENERATED_ROOT",
                        _artifact_value(artifacts, "generate_batch", "output_dir"),
                        "EVAL.PERSONALIZATION.DATASET_PATH",
                        _artifact_value(artifacts, "text_embed", "dataset_path"),
                    ],
                ),
            )
            compare_result = run_compare(
                config_path,
                _merge_overrides(
                    overrides,
                    [
                        "REPORT.CANDIDATE_ROOT",
                        str(run_root),
                        "REPORT.OUTPUT_DIR",
                        str(run_root / "report"),
                    ],
                ),
            )
            compare_result["generation_evaluation"] = generation_eval
            compare_result["personalization_evaluation"] = personalization_eval
            artifacts[stage] = _write_stage_manifest(run_root, stage, {"stage": stage, "status": "completed", **compare_result})
        else:
            artifacts[stage] = _write_stage_manifest(run_root, stage, {"stage": stage, "status": "skipped", "reason": "unknown stage"})

        top_manifest["stages"][stage] = artifacts[stage]

    write_manifest(run_root / "manifest.json", top_manifest)
    return {"output_dir": str(run_root), "manifest_path": str(run_root / "manifest.json")}
