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
            "train-diffusion",
            "--config",
            "configs/experiments/diffusion/dit_ecgtwin.yaml",
            "--config",
            "configs/experiments/privacy/remove_base_vector.yaml",
            "SYSTEM.DEVICE",
            "cpu",
        ]
    )
    assert args.config == [
        "configs/experiments/diffusion/dit_ecgtwin.yaml",
        "configs/experiments/privacy/remove_base_vector.yaml",
    ]
    assert args.overrides == ["SYSTEM.DEVICE", "cpu"]


def test_serialized_data_paths_resolve_from_shared_root():
    cfg = load_config()
    resolved = resolve_serialized_data_path(cfg, "paired_Mimic_vae_multi_nomic.pt")
    assert str(resolved) == "/data/users/havens3/ecgtwin/paired_Mimic_vae_multi_nomic.pt"
