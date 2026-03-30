from pathlib import Path

import yaml


def test_all_yaml_configs_parse():
    config_paths = sorted(Path("configs").rglob("*.yaml"))
    assert config_paths
    for path in config_paths:
        with open(path, "r", encoding="utf-8") as handle:
            assert yaml.safe_load(handle.read()) is not None
