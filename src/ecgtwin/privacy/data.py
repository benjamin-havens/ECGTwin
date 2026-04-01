"""Dataset loading and grouping helpers for privacy audits."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from ecgtwin.evaluation.artifacts import json_ready


def subject_id_from_record(record: dict) -> str:
    """Extract a stable string subject identifier from a serialized record."""
    subject_id = record["label"].get("subject_id", "unknown")
    if isinstance(subject_id, torch.Tensor):
        subject_id = subject_id.item()
    if isinstance(subject_id, (list, tuple)):
        subject_id = subject_id[0]
    return str(subject_id)


def record_id_from_record(record: dict, index: int | None = None) -> str:
    """Build a mostly-stable record identifier from label metadata."""
    label = record["label"]
    parts = [subject_id_from_record(record)]
    for key in ("ecg_time", "study_id", "text"):
        value = label.get(key)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            value = value.item()
        if isinstance(value, (list, tuple)):
            value = value[0]
        parts.append(str(value))
    if index is not None:
        parts.append(str(index))
    record_id = "__".join(parts)
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in record_id)


def record_cache_key_from_record(record: dict, index: int | None = None) -> str:
    """Build a short stable identifier for filesystem paths and cache artifacts."""
    record_id = record_id_from_record(record, index=index)
    digest = hashlib.sha1(record_id.encode("utf-8")).hexdigest()
    subject_id = subject_id_from_record(record)
    return f"{subject_id}__{digest[:16]}"


def load_dataset_file(dataset_path: str, mmap: bool = False):
    """Load a serialized ECG dataset with repo-compatible torch.load defaults."""
    load_kwargs = {"weights_only": False}
    if mmap:
        load_kwargs["mmap"] = True
    try:
        return torch.load(dataset_path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("mmap", None)
        return torch.load(dataset_path, **load_kwargs)


def load_records(dataset_path: str, mmap: bool = False) -> list[dict]:
    """Load a serialized ECG dataset, flattening paired datasets when needed."""
    if not dataset_path:
        return []

    raw_dataset = load_dataset_file(dataset_path, mmap=mmap)
    if not raw_dataset:
        return []

    if isinstance(raw_dataset[0], tuple):
        flattened = []
        seen = set()
        for pair in raw_dataset:
            for record in pair:
                record_id = record_id_from_record(record)
                if record_id in seen:
                    continue
                seen.add(record_id)
                flattened.append(record)
        return flattened

    return raw_dataset if isinstance(raw_dataset, list) else list(raw_dataset)


def group_records_by_subject(
    records: list[dict],
    max_patients: int = 0,
    max_records_per_patient: int = 0,
) -> dict[str, list[dict]]:
    """Group records by subject and optionally truncate the audit population."""
    grouped_records: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped_records[subject_id_from_record(record)].append(record)

    grouped_records = dict(sorted(grouped_records.items(), key=lambda item: item[0]))
    if max_patients > 0:
        grouped_records = dict(list(grouped_records.items())[:max_patients])

    if max_records_per_patient > 0:
        grouped_records = {
            subject_id: entries[:max_records_per_patient]
            for subject_id, entries in grouped_records.items()
        }

    return grouped_records


def group_record_indices_by_subject(
    records: list[dict],
    max_patients: int = 0,
    max_records_per_patient: int = 0,
) -> dict[str, list[int]]:
    """Group dataset indices by subject without duplicating record payloads."""
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped_indices[subject_id_from_record(record)].append(index)

    grouped_indices = dict(sorted(grouped_indices.items(), key=lambda item: item[0]))
    if max_patients > 0:
        grouped_indices = dict(list(grouped_indices.items())[:max_patients])

    if max_records_per_patient > 0:
        grouped_indices = {
            subject_id: indices[:max_records_per_patient]
            for subject_id, indices in grouped_indices.items()
        }

    return grouped_indices


def flatten_grouped_records(grouped_records: dict[str, list[dict]]) -> list[dict]:
    """Flatten grouped audit records back into a record list."""
    flat_records = []
    for subject_id in sorted(grouped_records):
        flat_records.extend(grouped_records[subject_id])
    return flat_records


def build_manifest(grouped_member_records: dict[str, list[dict]], grouped_nonmember_records: dict[str, list[dict]]) -> dict:
    """Summarize the audit population that will be scored."""
    return {
        "member_subjects": len(grouped_member_records),
        "nonmember_subjects": len(grouped_nonmember_records),
        "member_records": sum(len(records) for records in grouped_member_records.values()),
        "nonmember_records": sum(len(records) for records in grouped_nonmember_records.values()),
        "member_subject_ids": sorted(grouped_member_records),
        "nonmember_subject_ids": sorted(grouped_nonmember_records),
    }


def save_manifest(manifest: dict, output_path: Path) -> None:
    """Persist an audit manifest to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_ready(manifest), indent=2), encoding="utf-8")
