"""CLI workflows for ECGTwin privacy auditing."""

from __future__ import annotations

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ecgtwin.core.runtime_env import configure_runtime_environment

configure_runtime_environment()

import torch
from tqdm import tqdm

from ecgtwin.config import load_config, resolve_serialized_data_path
from ecgtwin.core.logging import configure_logger
from ecgtwin.inference.rendering import save_ecg_plot
from ecgtwin.models.vae_model import VAE_Decoder

from .black_box import score_synthetic_pool
from .data import (
    build_manifest,
    group_record_indices_by_subject,
    load_dataset_file,
    load_records,
    record_cache_key_from_record,
    record_id_from_record,
    save_manifest,
)
from .generation import generate_synthetic_latents, load_privacy_runtime, save_synthetic_pool
from .metrics import append_csv_rows, iter_csv_rows, summarize_binary_scores, write_csv
from .reconstruction import reconstruction_score
from .visualization import write_privacy_visualizations
from .white_box import diffusion_white_box_score, ibe_white_box_score


def _next_experiment_dir(root: Path, experiment_name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing_indices = []
    for item in root.iterdir():
        if item.is_dir() and item.name.startswith(f"{experiment_name}_"):
            try:
                existing_indices.append(int(item.name.split("_")[-1]))
            except ValueError:
                continue
    return root / f"{experiment_name}_{max(existing_indices, default=0) + 1}"


def _persist_config(output_dir: Path, cfg) -> None:
    with open(output_dir / "resolved_config.yaml", "w", encoding="utf-8") as handle:
        handle.write(cfg.dump())


def _dataset_paths(cfg) -> tuple[str, str]:
    """Resolve member and nonmember dataset paths for privacy evaluation."""
    member_path = str(resolve_serialized_data_path(cfg, cfg.PRIVACY.MEMBER_DATASET_PATH or cfg.DATA.DATASET_PATH))
    nonmember_path = str(resolve_serialized_data_path(cfg, cfg.PRIVACY.NONMEMBER_DATASET_PATH))
    return member_path, nonmember_path


def _filter_overlapping_nonmembers(
    grouped_member_indices: dict[str, list[int]],
    grouped_nonmember_indices: dict[str, list[int]],
) -> tuple[dict[str, list[int]], list[str]]:
    """Drop nonmember subjects that overlap with member subjects."""
    overlap_subjects = sorted(set(grouped_member_indices) & set(grouped_nonmember_indices))
    if not overlap_subjects:
        return grouped_nonmember_indices, []
    overlap_set = set(overlap_subjects)
    filtered = {
        subject_id: indices
        for subject_id, indices in grouped_nonmember_indices.items()
        if subject_id not in overlap_set
    }
    return filtered, overlap_subjects


def _prepare_audit_inputs(cfg) -> dict:
    """Load dataset metadata, filter overlap, and build the audit manifest."""
    member_path, nonmember_path = _dataset_paths(cfg)
    member_records = load_records(member_path)
    nonmember_records = load_records(nonmember_path)

    grouped_member_indices = group_record_indices_by_subject(
        member_records,
        max_patients=cfg.PRIVACY.MAX_PATIENTS,
        max_records_per_patient=cfg.PRIVACY.MAX_RECORDS_PER_PATIENT,
    )
    grouped_nonmember_indices_raw = group_record_indices_by_subject(
        nonmember_records,
        max_patients=cfg.PRIVACY.MAX_PATIENTS,
        max_records_per_patient=cfg.PRIVACY.MAX_RECORDS_PER_PATIENT,
    )
    filtered_nonmember_indices, overlap_subjects = _filter_overlapping_nonmembers(
        grouped_member_indices,
        grouped_nonmember_indices_raw,
    )

    manifest = build_manifest(grouped_member_indices, filtered_nonmember_indices)
    manifest.update(
        {
            "member_dataset_path": member_path,
            "nonmember_dataset_path": nonmember_path,
            "member_subjects_before_filter": len(grouped_member_indices),
            "nonmember_subjects_before_filter": len(grouped_nonmember_indices_raw),
            "nonmember_records_before_filter": sum(len(indices) for indices in grouped_nonmember_indices_raw.values()),
            "overlap_subject_count": len(overlap_subjects),
            "overlap_subject_ids": overlap_subjects,
            "nonmember_subjects_after_filter": len(filtered_nonmember_indices),
            "nonmember_records_after_filter": sum(len(indices) for indices in filtered_nonmember_indices.values()),
        }
    )
    return {
        "member_path": member_path,
        "nonmember_path": nonmember_path,
        "member_records": member_records,
        "nonmember_records": nonmember_records,
        "grouped_member_indices": grouped_member_indices,
        "grouped_nonmember_indices": filtered_nonmember_indices,
        "manifest": manifest,
    }


def _build_subset_tasks(
    dataset_path: str,
    grouped_indices: dict[str, list[int]],
    subset: str,
    label: int,
) -> list[dict]:
    """Build subject-aware metadata for one dataset split."""
    tasks = []
    for subject_id, indices in grouped_indices.items():
        tasks.append(
            {
                "subset": subset,
                "dataset_path": dataset_path,
                "subject_id": subject_id,
                "label": label,
                "record_indices": list(indices),
                "record_count": len(indices),
            }
        )
    return tasks


def _build_record_tasks(
    member_path: str,
    grouped_member_indices: dict[str, list[int]],
    nonmember_path: str,
    grouped_nonmember_indices: dict[str, list[int]],
) -> list[dict]:
    """Build subject-level work definitions across the member and nonmember sets."""
    member_tasks = _build_subset_tasks(
        member_path,
        grouped_member_indices,
        subset="member",
        label=1,
    )
    nonmember_tasks = _build_subset_tasks(
        nonmember_path,
        grouped_nonmember_indices,
        subset="nonmember",
        label=0,
    )
    return member_tasks + nonmember_tasks


def _resolve_worker_devices(cfg) -> list[str]:
    """Resolve the device strings that should execute privacy workers."""
    if not torch.cuda.is_available():
        return ["cpu"]
    if cfg.PRIVACY.GPU_IDS:
        return [f"cuda:{gpu_id}" for gpu_id in cfg.PRIVACY.GPU_IDS]
    return [cfg.SYSTEM.DEVICE]


def _chunk_subject_tasks(subject_tasks: list[dict], chunk_size: int) -> list[dict]:
    """Group subject tasks into bounded worker chunks measured in record count."""
    chunks = []
    current_chunk = []
    current_count = 0
    target_size = max(int(chunk_size), 1)
    for task in subject_tasks:
        task_count = int(task["record_count"])
        if current_chunk and current_count + task_count > target_size:
            chunks.append({"subjects": current_chunk, "record_count": current_count})
            current_chunk = []
            current_count = 0
        current_chunk.append(task)
        current_count += task_count
    if current_chunk:
        chunks.append({"subjects": current_chunk, "record_count": current_count})
    return chunks


def _partition_tasks(tasks: list[dict], num_partitions: int) -> list[list[dict]]:
    """Split subject-level work into balanced per-device partitions."""
    partition_count = max(num_partitions, 1)
    partitions = [[] for _ in range(partition_count)]
    partition_loads = [0 for _ in range(partition_count)]
    ordered_tasks = sorted(tasks, key=lambda task: int(task["record_count"]), reverse=True)
    for task in ordered_tasks:
        target_index = min(range(partition_count), key=lambda idx: partition_loads[idx])
        partitions[target_index].append(task)
        partition_loads[target_index] += int(task["record_count"])
    return [partition for partition in partitions if partition]


def _build_worker_batches(tasks: list[dict], num_partitions: int, chunk_size: int) -> list[dict]:
    """Build balanced worker payloads and chunk them for progress reporting."""
    partitions = _partition_tasks(tasks, num_partitions)
    return [
        {
            "chunks": _chunk_subject_tasks(partition, chunk_size),
            "record_count": sum(int(task["record_count"]) for task in partition),
            "subject_count": len(partition),
        }
        for partition in partitions
    ]


def _worker_record_cache(cfg, dataset_cache: dict[str, list[dict]], dataset_path: str) -> list[dict]:
    """Load and cache one dataset inside a worker process."""
    if dataset_path not in dataset_cache:
        dataset_cache[dataset_path] = load_records(dataset_path, mmap=not cfg.PRIVACY.PRELOAD_DATASETS)
    return dataset_cache[dataset_path]


def _load_or_generate_synthetic_payload(
    task: dict,
    cfg,
    runtime: dict,
    synthetic_root: Path,
    dataset_cache: dict[str, list[dict]],
) -> dict:
    """Load an existing synthetic pool or generate one for the current task."""
    payload_path = synthetic_root / task["subset"] / f"{task['cache_key']}.pt"
    if payload_path.exists():
        return load_dataset_file(str(payload_path))

    records = _worker_record_cache(cfg, dataset_cache, task["dataset_path"])
    record = records[task["record_index"]]
    latents = generate_synthetic_latents(record, runtime, cfg)
    save_synthetic_pool(payload_path, record, task["record_id"], task["cache_key"], task["subset"], latents)
    return {"record_id": task["record_id"], "cache_key": task["cache_key"], "subset": task["subset"], "latent_gen": latents}


def _generate_only_worker(
    config_paths,
    overrides,
    rank: int,
    device_override: str,
    worker_batch: dict,
    synthetic_root_str: str,
    output_root_str: str,
) -> dict:
    """Worker entrypoint for synthetic-pool generation on a dedicated device."""
    worker_overrides = list(overrides) + ["SYSTEM.DEVICE", device_override]
    cfg = load_config(config_paths, worker_overrides)
    runtime = load_privacy_runtime(cfg)
    synthetic_root = Path(synthetic_root_str)
    output_root = Path(output_root_str)
    dataset_cache: dict[str, list[dict]] = {}

    generated = 0
    progress = tqdm(
        total=int(worker_batch["record_count"]),
        desc=f"privacy-generate[{device_override}]",
        position=rank,
        leave=True,
        disable=not cfg.PRIVACY.PROGRESS_BAR,
    )
    try:
        for chunk in worker_batch["chunks"]:
            for subject_task in chunk["subjects"]:
                records = _worker_record_cache(cfg, dataset_cache, subject_task["dataset_path"])
                indices = subject_task["record_indices"]
                for record_index in indices:
                    record = records[record_index]
                    record_id = record_id_from_record(record)
                    cache_key = record_cache_key_from_record(record)
                    payload_path = synthetic_root / subject_task["subset"] / f"{cache_key}.pt"
                    if payload_path.exists():
                        progress.update(1)
                        continue
                    latents = generate_synthetic_latents(record, runtime, cfg)
                    save_synthetic_pool(payload_path, record, record_id, cache_key, subject_task["subset"], latents)
                    generated += 1
                    progress.update(1)
    finally:
        progress.close()

    shard_path = output_root / "generate_shards" / f"generate_manifest_rank{rank}.json"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_summary = {
        "rank": rank,
        "device": device_override,
        "record_count": int(worker_batch["record_count"]),
        "chunk_count": len(worker_batch["chunks"]),
        "generated_count": generated,
    }
    shard_path.write_text(json.dumps(shard_summary, indent=2), encoding="utf-8")
    return shard_summary


def _audit_worker(
    config_paths,
    overrides,
    rank: int,
    device_override: str,
    worker_batch: dict,
    synthetic_root_str: str,
    output_root_str: str,
) -> dict:
    """Worker entrypoint for per-record privacy audit scoring on one device."""
    worker_overrides = list(overrides) + ["SYSTEM.DEVICE", device_override]
    cfg = load_config(config_paths, worker_overrides)
    runtime = load_privacy_runtime(cfg)
    ibe_model = runtime["ibe_model"]
    device = runtime["device"]
    synthetic_root = Path(synthetic_root_str)
    output_root = Path(output_root_str)
    dataset_cache: dict[str, list[dict]] = {}

    shard_path = output_root / "score_shards" / f"record_scores_rank{rank}.csv"
    rows_buffer = []
    row_count = 0
    progress = tqdm(
        total=int(worker_batch["record_count"]),
        desc=f"privacy-audit[{device_override}]",
        position=rank,
        leave=True,
        disable=not cfg.PRIVACY.PROGRESS_BAR,
    )
    task_index = 0
    try:
        for chunk in worker_batch["chunks"]:
            for subject_task in chunk["subjects"]:
                records = _worker_record_cache(cfg, dataset_cache, subject_task["dataset_path"])
                indices = subject_task["record_indices"]
                for local_index, record_index in enumerate(indices):
                    partner_index = indices[(local_index + 1) % len(indices)] if len(indices) > 1 else record_index
                    record = records[record_index]
                    partner = records[partner_index]
                    record_id = record_id_from_record(record)
                    cache_key = record_cache_key_from_record(record)
                    task = {
                        "subset": subject_task["subset"],
                        "dataset_path": subject_task["dataset_path"],
                        "record_index": record_index,
                        "partner_index": partner_index,
                        "subject_id": subject_task["subject_id"],
                        "label": subject_task["label"],
                        "record_id": record_id,
                        "cache_key": cache_key,
                        "task_index": task_index,
                    }
                    synthetic_payload = _load_or_generate_synthetic_payload(task, cfg, runtime, synthetic_root, dataset_cache)

                    bb_scores = score_synthetic_pool(
                        record,
                        synthetic_payload["latent_gen"],
                        feature_space=cfg.PRIVACY.FEATURE_SPACE,
                        ibe_model=ibe_model,
                        device=device,
                        k=cfg.PRIVACY.BLACK_BOX.KNN_K,
                        distance=cfg.PRIVACY.BLACK_BOX.DISTANCE,
                        reference_split=cfg.PRIVACY.DOMIAS.REFERENCE_SPLIT,
                        use_amp=cfg.PRIVACY.USE_AMP,
                    )
                    rows_buffer.extend(
                        [
                            {
                                "attack": "black_box",
                                "level": "record",
                                "subset": subject_task["subset"],
                                "subject_id": subject_task["subject_id"],
                                "record_id": record_id,
                                "label": subject_task["label"],
                                **bb_scores["black_box"],
                            },
                            {
                                "attack": "domias",
                                "level": "record",
                                "subset": subject_task["subset"],
                                "subject_id": subject_task["subject_id"],
                                "record_id": record_id,
                                "label": subject_task["label"],
                                **bb_scores["domias"],
                            },
                        ]
                    )

                    ibe_scores = ibe_white_box_score(record, partner, ibe_model=ibe_model, device=device)
                    rows_buffer.append(
                        {
                            "attack": "white_box_ibe",
                            "level": "record",
                            "subset": subject_task["subset"],
                            "subject_id": subject_task["subject_id"],
                            "record_id": record_id,
                            "label": subject_task["label"],
                            **ibe_scores,
                        }
                    )

                    diffusion_scores = diffusion_white_box_score(
                        record,
                        ibe_model=ibe_model,
                        noise_predictor=runtime["noise_predictor"],
                        diffused_model=runtime["diffused_model"],
                        device=device,
                        timesteps=list(cfg.PRIVACY.WHITE_BOX.TIMESTEPS),
                        base_vector_mode=cfg.MODEL.BASE_VECTOR.MODE,
                        base_vector_noise_std=cfg.MODEL.BASE_VECTOR.NOISE_STD,
                        seed=cfg.PRIVACY.RANDOM_SEED + task_index,
                    )
                    rows_buffer.append(
                        {
                            "attack": "white_box_diffusion",
                            "level": "record",
                            "subset": subject_task["subset"],
                            "subject_id": subject_task["subject_id"],
                            "record_id": record_id,
                            "label": subject_task["label"],
                            **diffusion_scores,
                        }
                    )
                    if cfg.PRIVACY.RECONSTRUCTION.ENABLED:
                        rows_buffer.append(
                            {
                                "attack": "reconstruction",
                                "level": "record",
                                "subset": subject_task["subset"],
                                "subject_id": subject_task["subject_id"],
                                "record_id": record_id,
                                "label": subject_task["label"],
                                "cache_key": cache_key,
                                "dataset_path": subject_task["dataset_path"],
                                "record_index": record_index,
                                **reconstruction_score(
                                    record["data"],
                                    synthetic_payload["latent_gen"],
                                    distance=cfg.PRIVACY.RECONSTRUCTION.DISTANCE,
                                ),
                            }
                        )
                    row_count += 4 + int(cfg.PRIVACY.RECONSTRUCTION.ENABLED)
                    task_index += 1
                    progress.update(1)

                    if len(rows_buffer) >= 1024:
                        append_csv_rows(rows_buffer, shard_path)
                        rows_buffer = []
    finally:
        progress.close()

    if rows_buffer:
        append_csv_rows(rows_buffer, shard_path)

    manifest_path = output_root / "score_shards" / f"worker_manifest_rank{rank}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    worker_summary = {
        "rank": rank,
        "device": device_override,
        "record_count": int(worker_batch["record_count"]),
        "chunk_count": len(worker_batch["chunks"]),
        "row_count": row_count,
        "score_shard_path": str(shard_path),
    }
    manifest_path.write_text(json.dumps(worker_summary, indent=2), encoding="utf-8")
    return worker_summary


def _execute_worker_batches(
    worker_fn,
    config_paths,
    overrides,
    devices: list[str],
    worker_batches: list[dict],
    synthetic_root: Path,
    output_root: Path,
) -> list[dict]:
    """Run privacy work on one or more devices using one process per device."""
    if not worker_batches:
        return []
    active_devices = devices[: len(worker_batches)]
    if len(active_devices) == 1:
        return [
            worker_fn(
                config_paths,
                overrides,
                0,
                active_devices[0],
                worker_batches[0],
                str(synthetic_root),
                str(output_root),
            )
        ]

    with ProcessPoolExecutor(max_workers=len(active_devices), mp_context=mp.get_context("spawn")) as executor:
        futures = [
            executor.submit(
                worker_fn,
                config_paths,
                overrides,
                rank,
                device,
                task_batch,
                str(synthetic_root),
                str(output_root),
            )
            for rank, (device, task_batch) in enumerate(zip(active_devices, worker_batches, strict=True))
        ]
        return [future.result() for future in futures]


def _log_preflight(logger, manifest: dict, devices: list[str], worker_batches: list[dict]) -> None:
    """Emit a compact preflight summary before launching workers."""
    total_records = sum(int(batch["record_count"]) for batch in worker_batches)
    logger.info(
        "Preflight member subjects=%s records=%s nonmember subjects(before/after filter)=%s/%s records(before/after filter)=%s/%s overlap_subjects=%s",
        manifest["member_subjects"],
        manifest["member_records"],
        manifest["nonmember_subjects_before_filter"],
        manifest["nonmember_subjects_after_filter"],
        manifest["nonmember_records_before_filter"],
        manifest["nonmember_records_after_filter"],
        manifest["overlap_subject_count"],
    )
    logger.info(
        "Selected devices: %s; record_count=%s; worker_count=%s; estimated synthetic generations=%s",
        ", ".join(devices),
        total_records,
        len(worker_batches),
        total_records * manifest.get("synthetic_num_samples", 0),
    )
    for device, worker_batch in zip(devices[: len(worker_batches)], worker_batches, strict=True):
        logger.info(
            "Device %s assigned %s records across %s chunks and %s subjects",
            device,
            worker_batch["record_count"],
            len(worker_batch["chunks"]),
            worker_batch["subject_count"],
        )


def _attack_aggregations(cfg) -> dict[str, str]:
    """Map each attack to its patient-level aggregation policy."""
    aggregations = {
        "black_box": cfg.PRIVACY.BLACK_BOX.AGGREGATION,
        "domias": cfg.PRIVACY.DOMIAS.AGGREGATION,
        "white_box_ibe": cfg.PRIVACY.WHITE_BOX.AGGREGATION,
        "white_box_diffusion": cfg.PRIVACY.WHITE_BOX.AGGREGATION,
    }
    if cfg.PRIVACY.RECONSTRUCTION.ENABLED:
        aggregations["reconstruction"] = cfg.PRIVACY.RECONSTRUCTION.AGGREGATION
    return aggregations


def _aggregate_subject_scores(scores: list[float], aggregation: str) -> float:
    """Collapse record-level scores into a patient-level score."""
    if aggregation == "mean":
        return sum(scores) / len(scores)
    return max(scores)


def _stream_score_rows(worker_summaries: list[dict], output_path: Path, aggregations: dict[str, str]):
    """Merge worker shards into the final scores CSV while aggregating metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    metric_inputs: dict[tuple[str, str], dict[str, list[float]]] = {}
    subject_groups: dict[tuple[str, str, str], dict[str, object]] = {}
    write_buffer = []

    for summary in worker_summaries:
        shard_path = Path(summary["score_shard_path"])
        for row in iter_csv_rows(shard_path):
            write_buffer.append(row)
            attack = row["attack"]
            metric_key = (attack, "record")
            metric_inputs.setdefault(metric_key, {"labels": [], "scores": []})
            metric_inputs[metric_key]["labels"].append(int(row["label"]))
            metric_inputs[metric_key]["scores"].append(float(row["score"]))

            subject_key = (attack, row["subset"], row["subject_id"])
            state = subject_groups.setdefault(subject_key, {"label": int(row["label"]), "scores": []})
            state["scores"].append(float(row["score"]))

            if len(write_buffer) >= 4096:
                append_csv_rows(write_buffer, output_path)
                write_buffer = []

    if write_buffer:
        append_csv_rows(write_buffer, output_path)

    patient_buffer = []
    for (attack, subset, subject_id), state in subject_groups.items():
        scores = state["scores"]
        patient_score = _aggregate_subject_scores(scores, aggregations.get(attack, "max"))
        row = {
            "attack": attack,
            "level": "patient",
            "subset": subset,
            "subject_id": subject_id,
            "record_id": "",
            "label": int(state["label"]),
            "score": patient_score,
        }
        patient_buffer.append(row)

        metric_key = (attack, "patient")
        metric_inputs.setdefault(metric_key, {"labels": [], "scores": []})
        metric_inputs[metric_key]["labels"].append(int(row["label"]))
        metric_inputs[metric_key]["scores"].append(float(row["score"]))

        if len(patient_buffer) >= 4096:
            append_csv_rows(patient_buffer, output_path)
            patient_buffer = []

    if patient_buffer:
        append_csv_rows(patient_buffer, output_path)

    return metric_inputs


def _select_reconstruction_examples(scores_path: Path, limit_per_label: int) -> list[dict]:
    """Select the highest-scoring record-level reconstruction rows per subset."""
    if limit_per_label <= 0:
        return []

    grouped: dict[str, list[dict]] = {}
    for row in iter_csv_rows(scores_path):
        if row.get("attack") != "reconstruction" or row.get("level") != "record":
            continue
        grouped.setdefault(row.get("subset", "unknown"), []).append(row)

    selected = []
    for subset in sorted(grouped):
        rows = sorted(grouped[subset], key=lambda row: float(row["score"]), reverse=True)
        selected.extend(rows[:limit_per_label])
    return selected


def _load_decoder_for_examples(cfg) -> tuple[VAE_Decoder, torch.device]:
    """Load the VAE decoder for qualitative reconstruction exports."""
    device = torch.device(cfg.SYSTEM.DEVICE if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(cfg.CHECKPOINTS.VAE_PATH, map_location="cpu")
    decoder = VAE_Decoder()
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.to(device)
    decoder.eval()
    return decoder, device


def _save_reconstruction_waveforms(
    example_dir: Path,
    decoder: VAE_Decoder,
    device: torch.device,
    reference_latent: torch.Tensor,
    reconstructed_latent: torch.Tensor,
) -> None:
    """Decode and save waveform-space qualitative reconstruction artifacts."""
    lead_index = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]
    with torch.no_grad():
        batch = torch.cat([reference_latent, reconstructed_latent], dim=0).to(device=device, dtype=torch.float32)
        decoded = decoder(batch).detach().cpu()

    reference_ecg = decoded[0:1]
    reconstructed_ecg = decoded[1:2]
    torch.save(reference_ecg, example_dir / "reference_ecg.pt")
    torch.save(reconstructed_ecg, example_dir / "reconstruction_ecg.pt")
    save_ecg_plot(reference_ecg[0].numpy().transpose(1, 0), example_dir / "reference_ecg.png", lead_index)
    save_ecg_plot(reconstructed_ecg[0].numpy().transpose(1, 0), example_dir / "reconstruction_ecg.png", lead_index)


def _export_reconstruction_examples(cfg, scores_path: Path, synthetic_root: Path, output_dir: Path) -> list[Path]:
    """Persist bounded reconstruction examples after the audit scores are merged."""
    if not cfg.PRIVACY.RECONSTRUCTION.ENABLED:
        return []

    selected_rows = _select_reconstruction_examples(
        scores_path=scores_path,
        limit_per_label=cfg.PRIVACY.RECONSTRUCTION.SAVE_EXAMPLE_COUNT_PER_LABEL,
    )
    if not selected_rows:
        return []

    example_root = output_dir / "reconstruction_examples"
    dataset_cache: dict[str, list[dict]] = {}
    decoder = None
    decoder_device = None
    if cfg.PRIVACY.RECONSTRUCTION.DECODE_EXAMPLES:
        decoder, decoder_device = _load_decoder_for_examples(cfg)

    exported_dirs: list[Path] = []
    for example_index, row in enumerate(selected_rows):
        subset = row["subset"]
        cache_key = row["cache_key"]
        dataset_path = row["dataset_path"]
        if dataset_path not in dataset_cache:
            dataset_cache[dataset_path] = load_records(dataset_path, mmap=False)
        record = dataset_cache[dataset_path][int(row["record_index"])]
        payload = load_dataset_file(str(synthetic_root / subset / f"{cache_key}.pt"))
        best_candidate_index = int(float(row["best_candidate_index"]))

        reference_latent = record["data"].unsqueeze(0).to(dtype=torch.float32)
        reconstructed_latent = payload["latent_gen"][best_candidate_index : best_candidate_index + 1].to(dtype=torch.float32)

        example_dir = example_root / subset / f"{example_index:03d}_{cache_key}"
        example_dir.mkdir(parents=True, exist_ok=True)
        torch.save(reference_latent.cpu(), example_dir / "reference_latent.pt")
        torch.save(reconstructed_latent.cpu(), example_dir / "reconstruction_latent.pt")
        (example_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "attack": row["attack"],
                    "subset": subset,
                    "subject_id": row["subject_id"],
                    "record_id": row["record_id"],
                    "cache_key": cache_key,
                    "dataset_path": dataset_path,
                    "record_index": int(row["record_index"]),
                    "score": float(row["score"]),
                    "distance_min": float(row["distance_min"]),
                    "cosine_max": float(row["cosine_max"]),
                    "best_candidate_index": best_candidate_index,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if decoder is not None and decoder_device is not None:
            _save_reconstruction_waveforms(
                example_dir=example_dir,
                decoder=decoder,
                device=decoder_device,
                reference_latent=reference_latent,
                reconstructed_latent=reconstructed_latent,
            )
        exported_dirs.append(example_dir)

    return exported_dirs


def run_splits(config_path, overrides):
    """Emit a manifest describing the privacy-audit member and nonmember pools."""
    cfg = load_config(config_path, overrides)
    output_dir = _next_experiment_dir(Path(cfg.PRIVACY.OUTPUT_DIR), "privacy_splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger("privacy_splits", output_dir / "splits.log")

    prepared = _prepare_audit_inputs(cfg)
    manifest = prepared["manifest"]
    manifest["synthetic_num_samples"] = cfg.PRIVACY.SYNTHETIC_NUM_SAMPLES
    save_manifest(manifest, output_dir / "audit_manifest.json")
    _persist_config(output_dir, cfg)
    logger.info(
        "Prepared privacy split manifest with %s member subjects, %s nonmember subjects after filtering, and %s overlapping nonmember subjects removed",
        manifest["member_subjects"],
        manifest["nonmember_subjects_after_filter"],
        manifest["overlap_subject_count"],
    )


def run_generate(config_path, overrides):
    """Generate synthetic pools for privacy attacks and persist them to disk."""
    cfg = load_config(config_path, overrides)
    output_dir = _next_experiment_dir(Path(cfg.PRIVACY.OUTPUT_DIR), "privacy_generate")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger("privacy_generate", output_dir / "generate.log")

    prepared = _prepare_audit_inputs(cfg)
    manifest = prepared["manifest"]
    manifest["synthetic_num_samples"] = cfg.PRIVACY.SYNTHETIC_NUM_SAMPLES
    save_manifest(manifest, output_dir / "audit_manifest.json")

    tasks = _build_record_tasks(
        prepared["member_path"],
        prepared["grouped_member_indices"],
        prepared["nonmember_path"],
        prepared["grouped_nonmember_indices"],
    )
    del prepared["member_records"]
    del prepared["nonmember_records"]

    devices = _resolve_worker_devices(cfg)
    synthetic_root = Path(cfg.PRIVACY.SYNTHETIC_DIR) if cfg.PRIVACY.SYNTHETIC_DIR else output_dir / "synthetic"
    worker_batches = _build_worker_batches(tasks, len(devices), cfg.PRIVACY.WORKER_CHUNK_SIZE)
    _log_preflight(logger, manifest, devices, worker_batches)
    worker_summaries = _execute_worker_batches(
        _generate_only_worker,
        config_path,
        overrides,
        devices,
        worker_batches,
        synthetic_root,
        output_dir,
    )
    logger.info("Generated %s new synthetic pools across %s workers", sum(summary["generated_count"] for summary in worker_summaries), len(worker_summaries))
    _persist_config(output_dir, cfg)


def run_audit(config_path, overrides):
    """Run membership and reconstruction privacy scoring for ECGTwin."""
    cfg = load_config(config_path, overrides)
    output_dir = _next_experiment_dir(Path(cfg.PRIVACY.OUTPUT_DIR), cfg.PRIVACY.EXP_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(cfg.PRIVACY.EXP_NAME, output_dir / "audit.log")
    logger.info(cfg.dump())

    prepared = _prepare_audit_inputs(cfg)
    manifest = prepared["manifest"]
    manifest["synthetic_num_samples"] = cfg.PRIVACY.SYNTHETIC_NUM_SAMPLES
    save_manifest(manifest, output_dir / "audit_manifest.json")

    tasks = _build_record_tasks(
        prepared["member_path"],
        prepared["grouped_member_indices"],
        prepared["nonmember_path"],
        prepared["grouped_nonmember_indices"],
    )
    del prepared["member_records"]
    del prepared["nonmember_records"]

    devices = _resolve_worker_devices(cfg)
    synthetic_root = Path(cfg.PRIVACY.SYNTHETIC_DIR) if cfg.PRIVACY.SYNTHETIC_DIR else output_dir / "synthetic"
    worker_batches = _build_worker_batches(tasks, len(devices), cfg.PRIVACY.WORKER_CHUNK_SIZE)
    _log_preflight(logger, manifest, devices, worker_batches)
    worker_summaries = _execute_worker_batches(
        _audit_worker,
        config_path,
        overrides,
        devices,
        worker_batches,
        synthetic_root,
        output_dir,
    )
    metric_inputs = _stream_score_rows(
        worker_summaries,
        output_dir / "scores.csv",
        aggregations=_attack_aggregations(cfg),
    )

    metrics = {}
    roc_rows = []
    for attack_name in sorted({attack for attack, _ in metric_inputs}):
        for level in cfg.PRIVACY.LEVELS:
            metric_key = (attack_name, level)
            if metric_key not in metric_inputs:
                continue
            labels = metric_inputs[metric_key]["labels"]
            scores = metric_inputs[metric_key]["scores"]
            summary = summarize_binary_scores(labels, scores)
            metrics[f"{attack_name}:{level}"] = {key: value for key, value in summary.items() if key != "roc_curve"}
            for point in summary["roc_curve"]:
                roc_rows.append({"attack": attack_name, "level": level, **point})

    write_csv(roc_rows, output_dir / "roc.csv")
    scores_path = output_dir / "scores.csv"
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _persist_config(output_dir, cfg)

    visualization_paths = []
    if cfg.PRIVACY.PLOTS:
        visualization_paths = write_privacy_visualizations(
            output_dir=output_dir,
            roc_rows=roc_rows,
            metrics=metrics,
            scores_path=scores_path,
        )
    reconstruction_example_paths = _export_reconstruction_examples(
        cfg=cfg,
        scores_path=scores_path,
        synthetic_root=synthetic_root,
        output_dir=output_dir,
    )

    summary_lines = ["# Privacy Audit Summary", ""]
    for key, values in sorted(metrics.items()):
        summary_lines.append(f"## {key}")
        for metric_name, metric_value in sorted(values.items()):
            summary_lines.append(f"- {metric_name}: {metric_value:.6f}")
        summary_lines.append("")
    if reconstruction_example_paths:
        summary_lines.append("## Reconstruction Examples")
        summary_lines.append(f"- exported: {len(reconstruction_example_paths)}")
        summary_lines.append("- directory: reconstruction_examples")
        summary_lines.append("")
    if visualization_paths:
        summary_lines.append("## Visualizations")
        for path in visualization_paths:
            summary_lines.append(f"- {path.name}")
        summary_lines.append("")
    (output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info(
        "Completed privacy audit with %s worker score rows across %s workers; overlap-filtered nonmember subjects=%s; generated_plots=%s",
        sum(summary["row_count"] for summary in worker_summaries),
        len(worker_summaries),
        manifest["overlap_subject_count"],
        len(visualization_paths),
    )
