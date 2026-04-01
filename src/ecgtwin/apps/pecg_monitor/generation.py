"""Personalized ECG generation workflow for the pECGMonitor app."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from ecgtwin.config import load_config, resolve_serialized_data_path
from ecgtwin.data.patient import normalize_patient_value, sex_to_binary
from ecgtwin.data.text_embeddings import get_text_embedding
from ecgtwin.inference.generation import ddpm_generation
from ecgtwin.inference.lightning_predict import run_generation_tasks
from ecgtwin.models.base_vector import apply_base_vector_ablation
from ecgtwin.privacy.features import sample_patient_tensor, sample_text_tensors


def load_generation_source_entries(cfg) -> list[dict]:
    """Load the label/source definitions used to synthesize per-subject trainsets."""
    with open(cfg.APPS.PECG_MONITOR.GENERATION_SOURCE_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_pecg_generation_tasks(cfg) -> list[dict]:
    """Build one pECGMonitor generation task per subject."""
    test_dataset = torch.load(resolve_serialized_data_path(cfg, cfg.APPS.PECG_MONITOR.TEST_DATASET_PATH), map_location="cpu")
    output_dir = Path(cfg.APPS.PECG_MONITOR.OUTPUT_DIR) / cfg.MODEL.NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        {
            "task_id": str(subject_id),
            "subject_id": str(subject_id),
            "reference": ecg_list[0],
            "output_path": str(output_dir / f"{subject_id}.pt"),
        }
        for subject_id, ecg_list in test_dataset.items()
    ]


def write_pecg_generation_task(task: dict, runtime: dict, cfg, source_entries: list[dict]) -> dict:
    """Generate and persist the personalized latent trainset for one subject."""
    device = runtime["device"]
    ecg_ref = task["reference"]
    ref_latent = ecg_ref["data"].unsqueeze(0).transpose(2, 1).to(device=device, dtype=torch.float32)
    ref_text_embed, ref_text_mask = sample_text_tensors(ecg_ref, device)
    ref_pat_info = sample_patient_tensor(ecg_ref, device)
    base_vector = runtime["conditioner"].extract_features(ref_latent, ref_text_embed, ref_text_mask, ref_pat_info, reduce=True)
    if cfg.MODEL.BASE_VECTOR.APPLY_AT_INFERENCE:
        base_vector = apply_base_vector_ablation(
            base_vector,
            mode=cfg.MODEL.BASE_VECTOR.MODE,
            noise_std=cfg.MODEL.BASE_VECTOR.NOISE_STD,
        )

    personal_trainset = []
    for entry in source_entries:
        gen_batch = 128 if entry["label"] == 0 else 64
        ib_vector_dp = base_vector.repeat(gen_batch, 1)
        pat_info_vector_tar = torch.tensor(
            [
                normalize_patient_value("hr", entry["hr"] + torch.randint(-10, 10, (1,))),
                normalize_patient_value("age", ecg_ref["label"]["age"] + torch.randint(0, 20, (1,))),
                normalize_patient_value("sex", sex_to_binary(ecg_ref["label"]["sex"])),
            ],
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0).repeat(gen_batch, 1)

        text_embed_tar = get_text_embedding(
            text=entry["text"],
            tokenizer=runtime["tokenizer"],
            embedding_model=runtime["embedding_model"],
            mix=cfg.MODEL.MIX_TEXT,
        ).unsqueeze(0).repeat(gen_batch, 1, 1)

        latent_gen = ddpm_generation(
            diffused_model=runtime["scheduler"],
            noise_predictor=runtime["noise_predictor"],
            batch_size=gen_batch,
            device=device,
            text_embed=text_embed_tar,
            text_embed_mask=None,
            pat_info=pat_info_vector_tar,
            base_vector=ib_vector_dp,
            progress_bar=False,
        ).detach().cpu()
        personal_trainset.extend([{"data": sample, "label": entry["label"]} for sample in latent_gen])

    output_path = Path(task["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(personal_trainset, output_path)
    (output_path.with_suffix(".json")).write_text(
        json.dumps({"subject_id": task["subject_id"], "num_samples": len(personal_trainset)}, indent=2),
        encoding="utf-8",
    )
    return {"task_id": task["task_id"], "output_path": str(output_path)}


def run(config_path, overrides):
    """Generate subject-specific training samples for pECGMonitor."""
    cfg = load_config(config_path, overrides)
    if not getattr(cfg.EXECUTION, "GPU_IDS", []):
        legacy_gpu = cfg.APPS.PECG_MONITOR.GPU_DEVICE
        if legacy_gpu.lower().startswith("cuda:"):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.EXECUTION.GPU_IDS = [int(legacy_gpu.split(":", maxsplit=1)[1])]
            cfg.freeze()

    source_entries = load_generation_source_entries(cfg)
    output_dir = Path(cfg.APPS.PECG_MONITOR.OUTPUT_DIR) / cfg.MODEL.NAME
    summary = run_generation_tasks(
        cfg,
        build_pecg_generation_tasks(cfg),
        scope="pecg",
        output_root=output_dir,
        task_handler=lambda task, runtime, run_cfg: write_pecg_generation_task(task, runtime, run_cfg, source_entries),
    )
    return {"output_dir": str(output_dir), **summary}
