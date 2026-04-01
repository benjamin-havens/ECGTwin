"""Synthetic-pool generation helpers for black-box privacy attacks."""

from __future__ import annotations

from pathlib import Path
from contextlib import nullcontext

import torch

from ecgtwin.evaluation.runtime import load_generation_runtime
from ecgtwin.inference.generation import ddpm_generation
from ecgtwin.inference.scheduler import build_inference_scheduler
from ecgtwin.models.base_vector import apply_base_vector_ablation
from ecgtwin.models.conditioner import conditioner_hparams, load_conditioner
from ecgtwin.models.factory import build_noise_predictor
from ecgtwin.privacy.data import load_records, record_cache_key_from_record, record_id_from_record

from .features import sample_latent_batch, sample_patient_tensor, sample_text_tensors


def _hyper_params(cfg):
    """Translate the config tree into the model-factory hyperparameter format."""
    conditioner = conditioner_hparams(cfg)
    return {
        "ddpm": {
            "num_train_steps": cfg.DIFFUSION.NUM_TRAIN_STEPS,
            "beta_start": cfg.DIFFUSION.BETA_START,
            "beta_end": cfg.DIFFUSION.BETA_END,
        },
        "dit": {
            "hidden_size": cfg.MODEL.DIT.HIDDEN_SIZE,
            "depth": cfg.MODEL.DIT.DEPTH,
            "num_heads": cfg.MODEL.DIT.NUM_HEADS,
            "patient_info_size": cfg.MODEL.DIT.PATIENT_INFO_SIZE,
        },
        "unet": {
            "kernel_size": cfg.MODEL.UNET.KERNEL_SIZE,
            "num_level": cfg.MODEL.UNET.NUM_LEVEL,
            "n_heads": cfg.MODEL.UNET.N_HEADS,
            "patient_info_size": cfg.MODEL.UNET.PATIENT_INFO_SIZE,
        },
        "conditioner": {
            "embed_dim": conditioner["embed_dim"],
            "text_embed_dim": conditioner["text_embed_dim"],
            "patient_info_size": conditioner["patient_info_size"],
        },
    }


def load_privacy_runtime(cfg):
    """Load the trained ECGTwin components needed for privacy scoring."""
    if not cfg.MODEL.USE_VAE_LATENT:
        raise NotImplementedError("Privacy-audit generation currently supports VAE-latent ECGTwin checkpoints only")
    runtime = load_generation_runtime(cfg, include_decoder=False, include_text_encoder=False)
    return {
        "device": runtime["device"],
        "noise_predictor": runtime["noise_predictor"],
        "diffused_model": runtime["scheduler"],
        "conditioner": runtime["conditioner"],
    }


@torch.no_grad()
def generate_synthetic_latents(sample: dict, runtime: dict, cfg, num_samples: int | None = None) -> torch.Tensor:
    """Generate a synthetic latent pool conditioned on one candidate record."""
    device = runtime["device"]
    num_samples = num_samples or cfg.PRIVACY.SYNTHETIC_NUM_SAMPLES
    batch_size = cfg.PRIVACY.SYNTHETIC_BATCH_SIZE
    generated_batches = []
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if cfg.PRIVACY.USE_AMP and device.type == "cuda"
        else nullcontext()
    )

    for offset in range(0, num_samples, batch_size):
        current_batch = min(batch_size, num_samples - offset)
        with amp_context:
            ref_latent = sample_latent_batch(sample, device, repeat=current_batch).transpose(2, 1)
            ref_text, ref_mask = sample_text_tensors(sample, device, repeat=current_batch)
            ref_pat = sample_patient_tensor(sample, device, repeat=current_batch)
            base_vector = runtime["conditioner"].extract_features(ref_latent, ref_text, ref_mask, ref_pat, reduce=True)
            base_vector = apply_base_vector_ablation(
                base_vector,
                mode=cfg.MODEL.BASE_VECTOR.MODE,
                noise_std=cfg.MODEL.BASE_VECTOR.NOISE_STD,
            )

            tar_text, tar_mask = sample_text_tensors(sample, device, repeat=current_batch)
            tar_pat = sample_patient_tensor(sample, device, repeat=current_batch)
            latent_batch = ddpm_generation(
                diffused_model=runtime["diffused_model"],
                noise_predictor=runtime["noise_predictor"],
                batch_size=current_batch,
                device=device,
                text_embed=tar_text,
                text_embed_mask=tar_mask,
                pat_info=tar_pat,
                base_vector=base_vector,
                progress_bar=False,
            )
        generated_batches.append(latent_batch.cpu())

    return torch.cat(generated_batches, dim=0)


def save_synthetic_pool(
    output_path: Path,
    sample: dict,
    record_id: str,
    cache_key: str,
    subset: str,
    latents: torch.Tensor,
) -> None:
    """Persist a synthetic latent pool and its metadata to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "record_id": record_id,
            "cache_key": cache_key,
            "subject_id": sample["label"].get("subject_id"),
            "subset": subset,
            "latent_gen": latents,
        },
        output_path,
    )


def build_privacy_generation_tasks(subject_tasks: list[dict], synthetic_root: Path) -> list[dict]:
    """Build one generation task per record for the shared Lightning executor."""
    dataset_cache: dict[str, list[dict]] = {}
    tasks = []
    for subject_task in subject_tasks:
        dataset_path = subject_task["dataset_path"]
        if dataset_path not in dataset_cache:
            dataset_cache[dataset_path] = load_records(dataset_path, mmap=False)
        records = dataset_cache[dataset_path]
        for record_index in subject_task["record_indices"]:
            record = records[record_index]
            record_id = record_id_from_record(record)
            cache_key = record_cache_key_from_record(record)
            tasks.append(
                {
                    "task_id": cache_key,
                    "sample": record,
                    "subset": subject_task["subset"],
                    "record_id": record_id,
                    "cache_key": cache_key,
                    "output_path": str(synthetic_root / subject_task["subset"] / f"{cache_key}.pt"),
                }
            )
    return tasks


def write_privacy_generation_task(task: dict, runtime: dict, cfg) -> dict:
    """Generate and persist one synthetic latent pool."""
    latents = generate_synthetic_latents(task["sample"], runtime, cfg)
    save_synthetic_pool(
        Path(task["output_path"]),
        task["sample"],
        task["record_id"],
        task["cache_key"],
        task["subset"],
        latents,
    )
    return {"task_id": task["task_id"], "output_path": task["output_path"]}
