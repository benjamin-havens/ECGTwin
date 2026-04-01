"""Build paired ECG datasets from serialized single-record datasets."""

from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from ecgtwin.config import load_config, resolve_serialized_data_path


def build_pair_dataset(src_path: str, dst_path: str):
    """Group records by subject and emit chronologically ordered training pairs."""
    dataset = torch.load(src_path)
    grouped_dataset = defaultdict(list)

    for entry in tqdm(dataset):
        grouped_dataset[entry["label"]["subject_id"]].append(entry)

    for subject_id in [subject_id for subject_id, entries in grouped_dataset.items() if len(entries) < 2]:
        del grouped_dataset[subject_id]

    paired_dataset = []
    for entries in tqdm(grouped_dataset.values()):
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                if entries[i]["label"]["hr"] == entries[j]["label"]["hr"] and entries[i]["label"]["text"] == entries[j]["label"]["text"]:
                    continue
                ref_ecg_time = datetime.strptime(entries[i]["label"]["ecg_time"], "%Y-%m-%d %H:%M:%S")
                tar_ecg_time = datetime.strptime(entries[j]["label"]["ecg_time"], "%Y-%m-%d %H:%M:%S")
                paired_dataset.append((entries[i], entries[j]) if ref_ecg_time <= tar_ecg_time else (entries[j], entries[i]))

    torch.save(paired_dataset, dst_path)


def run(config_path, overrides):
    """Execute dataset pairing from config."""
    cfg = load_config(config_path, overrides)
    src_path = str(resolve_serialized_data_path(cfg, cfg.DATA.DATASET_PATH))
    dst_path = (
        str(resolve_serialized_data_path(cfg, cfg.DATA.TRAIN_DATASET_PATH))
        if cfg.DATA.TRAIN_DATASET_PATH
        else str(Path(cfg.PATHS.OUTPUT_DIR) / f"paired_{Path(src_path).name}")
    )
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    build_pair_dataset(src_path, dst_path)
    return {"dataset_path": dst_path}
