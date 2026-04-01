from pathlib import Path

import yaml

from ecgtwin.cli.main import build_parser
from ecgtwin.config import load_config, resolve_serialized_data_path


def test_all_yaml_configs_parse():
    config_paths = sorted(Path("configs").rglob("*.yaml"))
    assert config_paths
    for path in config_paths:
        with open(path, "r", encoding="utf-8") as handle:
            assert yaml.safe_load(handle.read()) is not None


def test_load_config_merges_multiple_partial_files_in_order(tmp_path):
    model_config = tmp_path / "model.yaml"
    model_config.write_text(
        "MODEL:\n"
        "  NAME: \"dit_attn\"\n"
        "  BASE_VECTOR:\n"
        "    MODE: \"noise\"\n",
        encoding="utf-8",
    )
    data_config = tmp_path / "data.yaml"
    data_config.write_text(
        "DATA:\n"
        "  DATASET_PATH: \"tmp/data.pt\"\n"
        "SYSTEM:\n"
        "  DEVICE: \"cpu\"\n",
        encoding="utf-8",
    )

    cfg = load_config([str(model_config), str(data_config)])
    assert cfg.MODEL.NAME == "dit_attn"
    assert cfg.MODEL.BASE_VECTOR.MODE == "noise"
    assert cfg.DATA.DATASET_PATH == "tmp/data.pt"
    assert cfg.SYSTEM.DEVICE == "cpu"


def test_cli_accepts_repeated_config_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "reproduce-paper",
            "--config",
            "configs/experiments/foundation/base.yaml",
            "--config",
            "configs/experiments/privacy/remove_base_vector.yaml",
            "SYSTEM.DEVICE",
            "cpu",
        ]
    )
    assert args.config == [
        "configs/experiments/foundation/base.yaml",
        "configs/experiments/privacy/remove_base_vector.yaml",
    ]
    assert args.overrides == ["SYSTEM.DEVICE", "cpu"]


def test_cli_parses_new_evaluation_commands():
    parser = build_parser()
    for command in ["train-vae", "generate-batch", "evaluate-generation", "evaluate-personalization", "compare-runs"]:
        args = parser.parse_args([command, "--config", "configs/experiments/foundation/base.yaml"])
        assert args.command == command
        assert args.config == ["configs/experiments/foundation/base.yaml"]


def test_foundation_config_enables_conditioner_path():
    cfg = load_config(["configs/experiments/foundation/base.yaml"])
    assert cfg.MODEL.CONDITIONER.TYPE == "foundation_jepa"
    assert cfg.CHECKPOINTS.CONDITIONER_PATH == "checkpoints/conditioner_best.pth"


def test_serialized_data_paths_resolve_from_shared_root():
    cfg = load_config()
    resolved = resolve_serialized_data_path(cfg, "paired_Mimic_vae_multi_nomic.pt")
    assert str(resolved) == "/data/users/havens3/ecgtwin/paired_Mimic_vae_multi_nomic.pt"


def test_repro_report_and_eval_defaults_exist():
    cfg = load_config()
    assert "compare" in cfg.REPRO.STAGES
    assert "figure10" in cfg.REPORT.TARGETS
    assert cfg.EXECUTION.STRATEGY == "ddp"
    assert cfg.SYSTEM.MATMUL_PRECISION == "default"
    assert cfg.MODEL.CLIP.TEXT_EMBED_DIM == 768
    assert cfg.EVAL.GENERATION.K_NEIGHBORS == 3
    assert cfg.EVAL.PERSONALIZATION.SCALING_PATIENT_COUNTS == [10, 20]
