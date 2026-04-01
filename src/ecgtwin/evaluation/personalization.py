"""Base-vector personalization metrics and figure export utilities."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ecgtwin.config import resolve_serialized_data_path
from ecgtwin.data.patient import build_patient_info_tensor, sex_to_binary
from ecgtwin.evaluation.artifacts import write_manifest
from ecgtwin.evaluation.generation import iter_pair_directories
from ecgtwin.models.conditioner import load_conditioner
from ecgtwin.privacy.data import group_records_by_subject, load_records, subject_id_from_record
from ecgtwin.privacy.features import sample_patient_tensor, sample_text_tensors

try:
    from sklearn.manifold import TSNE as SklearnTSNE
    from sklearn.metrics import silhouette_score as sklearn_silhouette_score
except ImportError:  # pragma: no cover - exercised indirectly when sklearn is absent
    SklearnTSNE = None
    sklearn_silhouette_score = None


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    return (a * b).sum(dim=-1)


def _patient_info_from_metadata(label: dict) -> torch.Tensor:
    sex = label.get("sex", "F")
    sex_binary = sex_to_binary(sex) if isinstance(sex, str) else sex
    return build_patient_info_tensor(
        normalize=True,
        add_token=False,
        hr=torch.tensor([float(label.get("hr", 0.0))]),
        age=torch.tensor([float(label.get("age", 0.0))]),
        sex=torch.tensor([float(sex_binary)]),
    )


def _load_saved_conditioning(pair_dir: Path, suffix: str) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    text_embed = torch.load(pair_dir / f"text_embed_{suffix}.pt", map_location="cpu")
    if not isinstance(text_embed, torch.Tensor):
        text_embed = torch.tensor(text_embed, dtype=torch.float32)
    if text_embed.ndim == 2:
        text_embed = text_embed.unsqueeze(0)
    text_mask_path = pair_dir / f"text_mask_{suffix}.pt"
    text_mask = torch.load(text_mask_path, map_location="cpu") if text_mask_path.exists() else None
    if isinstance(text_mask, torch.Tensor) and text_mask.ndim == 1:
        text_mask = text_mask.unsqueeze(0)
    pat_info = torch.load(pair_dir / f"pat_info_{suffix}.pt", map_location="cpu")
    return text_embed.float(), None if text_mask is None else text_mask.float(), pat_info.float()


def _extract_features(conditioner, device: torch.device, latent: torch.Tensor, text_embed, text_mask, pat_info) -> torch.Tensor:
    latent = latent.to(device=device, dtype=torch.float32)
    if latent.ndim == 2:
        latent = latent.unsqueeze(0)
    if text_embed is not None:
        text_embed = text_embed.to(device=device, dtype=torch.float32)
    if text_mask is not None:
        text_mask = text_mask.to(device=device, dtype=torch.float32)
    pat_info = pat_info.to(device=device, dtype=torch.float32)
    if pat_info.ndim == 1:
        pat_info = pat_info.unsqueeze(0)
    with torch.no_grad():
        return conditioner.extract_features(latent.transpose(2, 1), text_embed, text_mask, pat_info, reduce=True).detach().cpu()


def _real_subject_embeddings(cfg, conditioner, device: torch.device) -> dict[str, list[torch.Tensor]]:
    dataset_path = cfg.EVAL.PERSONALIZATION.DATASET_PATH
    if dataset_path:
        records = load_records(str(resolve_serialized_data_path(cfg, dataset_path)))
        grouped = group_records_by_subject(
            records,
            max_patients=int(cfg.EVAL.PERSONALIZATION.MAX_PATIENTS),
            max_records_per_patient=int(cfg.EVAL.PERSONALIZATION.MAX_RECORDS_PER_PATIENT),
        )
        embeddings = {}
        for subject_id, subject_records in grouped.items():
            subject_features = []
            for record in subject_records:
                latent = record["data"]
                text_embed, text_mask = sample_text_tensors(record, device)
                pat_info = sample_patient_tensor(record, device)
                subject_features.append(_extract_features(conditioner, device, latent, text_embed, text_mask, pat_info).squeeze(0))
            embeddings[str(subject_id)] = subject_features
        return embeddings

    generated_root = Path(cfg.EVAL.PERSONALIZATION.GENERATED_ROOT or cfg.EVAL.GENERATION.OUTPUT_DIR).expanduser().resolve()
    embeddings = defaultdict(list)
    for pair_dir in iter_pair_directories(generated_root):
        metadata = json.loads((pair_dir / "metadata.json").read_text(encoding="utf-8"))
        for suffix, key in (("ref", "reference_subject_id"), ("tar", "target_subject_id")):
            latent = torch.load(pair_dir / f"latent_{suffix}.pt", map_location="cpu")
            text_embed, text_mask, pat_info = _load_saved_conditioning(pair_dir, suffix)
            embeddings[metadata[key]].append(_extract_features(conditioner, device, latent, text_embed, text_mask, pat_info).squeeze(0))
    return dict(embeddings)


def _generated_subject_embeddings(cfg, conditioner, device: torch.device) -> tuple[dict[str, list[torch.Tensor]], list[dict[str, float]]]:
    generated_root = Path(cfg.EVAL.PERSONALIZATION.GENERATED_ROOT or cfg.EVAL.GENERATION.OUTPUT_DIR).expanduser().resolve()
    grouped_generated = defaultdict(list)
    similarity_rows = []
    for pair_dir in iter_pair_directories(generated_root):
        metadata = json.loads((pair_dir / "metadata.json").read_text(encoding="utf-8"))
        subject_id = metadata["reference_subject_id"]

        ref_latent = torch.load(pair_dir / "latent_ref.pt", map_location="cpu")
        tar_latent = torch.load(pair_dir / "latent_tar.pt", map_location="cpu")
        gen_latents = torch.load(pair_dir / "latent_gen.pt", map_location="cpu")
        ref_text_embed, ref_text_mask, ref_pat_info = _load_saved_conditioning(pair_dir, "ref")
        tar_text_embed, tar_text_mask, tar_pat_info = _load_saved_conditioning(pair_dir, "tar")

        ref_feature = _extract_features(conditioner, device, ref_latent, ref_text_embed, ref_text_mask, ref_pat_info)
        target_feature = _extract_features(conditioner, device, tar_latent, tar_text_embed, tar_text_mask, tar_pat_info)
        gen_features = _extract_features(
            conditioner,
            device,
            gen_latents,
            tar_text_embed.repeat(gen_latents.shape[0], 1, 1),
            None if tar_text_mask is None else tar_text_mask.repeat(gen_latents.shape[0], 1),
            tar_pat_info.repeat(gen_latents.shape[0], 1),
        )
        grouped_generated[subject_id].extend(list(gen_features))
        similarity_rows.append(
            {
                "subject_id": subject_id,
                "pair_dir": str(pair_dir),
                "reference_to_target_cosine": float(_cosine_similarity(ref_feature, target_feature).mean().item()),
                "reference_to_generated_cosine": float(_cosine_similarity(ref_feature.repeat(gen_features.shape[0], 1), gen_features).mean().item()),
                "target_to_generated_cosine": float(_cosine_similarity(target_feature.repeat(gen_features.shape[0], 1), gen_features).mean().item()),
            }
        )
    return dict(grouped_generated), similarity_rows


def _flatten_embeddings(grouped_embeddings: dict[str, list[torch.Tensor]]) -> tuple[torch.Tensor, list[str]]:
    features = []
    labels = []
    for subject_id in sorted(grouped_embeddings):
        for feature in grouped_embeddings[subject_id]:
            features.append(feature)
            labels.append(subject_id)
    if not features:
        return torch.empty(0, 0), []
    return torch.stack(features), labels


def _safe_silhouette(features: torch.Tensor, labels: list[str]) -> float:
    if features.shape[0] < 3 or len(set(labels)) < 2 or sklearn_silhouette_score is None:
        return float("nan")
    return float(sklearn_silhouette_score(features.numpy(), labels))


def _plot_tsne(features: torch.Tensor, labels: list[str], title: str, output_path: Path, save_pdf: bool, style_labels: list[str] | None = None) -> list[Path]:
    if features.shape[0] < 2:
        return []
    array = features.numpy()
    if SklearnTSNE is None or features.shape[0] < 3:
        centered = array - array.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        embedding = centered @ vt[:2].T
        if embedding.shape[1] < 2:
            embedding = np.pad(embedding, ((0, 0), (0, 2 - embedding.shape[1])))
    else:
        perplexity = max(2, min(30, features.shape[0] - 1))
        embedding = SklearnTSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(array)
    fig, ax = plt.subplots(figsize=(7, 6))
    unique_subjects = sorted(set(labels))
    markers = {"real": "o", "generated": "x"}
    for subject_id in unique_subjects:
        indices = [index for index, label in enumerate(labels) if label == subject_id]
        if style_labels is None:
            ax.scatter(embedding[indices, 0], embedding[indices, 1], label=subject_id, alpha=0.75)
            continue
        for style in sorted(set(style_labels[index] for index in indices)):
            style_indices = [index for index in indices if style_labels[index] == style]
            ax.scatter(
                embedding[style_indices, 0],
                embedding[style_indices, 1],
                label=f"{subject_id}:{style}",
                alpha=0.75,
                marker=markers.get(style, "o"),
            )
    ax.set_title(title)
    if len(unique_subjects) <= 12:
        ax.legend(loc="best", fontsize=8)
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight")
    generated_paths = [png_path]
    if save_pdf:
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        generated_paths.append(pdf_path)
    plt.close(fig)
    return generated_paths


def _plot_scaling_curve(grouped_embeddings: dict[str, list[torch.Tensor]], scaling_counts: list[int], output_dir: Path, save_pdf: bool) -> tuple[list[Path], list[dict[str, float]]]:
    subject_ids = sorted(grouped_embeddings)
    rows = []
    for count in scaling_counts:
        selected = {subject_id: grouped_embeddings[subject_id] for subject_id in subject_ids[: min(count, len(subject_ids))]}
        features, labels = _flatten_embeddings(selected)
        rows.append({"patient_count": int(count), "silhouette": _safe_silhouette(features, labels)})

    if not rows:
        return [], []

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([row["patient_count"] for row in rows], [row["silhouette"] for row in rows], marker="o")
    ax.set_xlabel("Patients")
    ax.set_ylabel("Silhouette")
    ax.set_title("Figure 9: Personalization Scaling")
    png_path = output_dir / "figure9_scaling.png"
    fig.savefig(png_path, bbox_inches="tight")
    generated_paths = [png_path]
    if save_pdf:
        pdf_path = output_dir / "figure9_scaling.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        generated_paths.append(pdf_path)
    plt.close(fig)
    return generated_paths, rows


def evaluate_personalization(cfg) -> dict:
    """Compute paper-style base-vector analyses for real and generated ECG twins."""
    output_dir = Path(cfg.EVAL.PERSONALIZATION.OUTPUT_DIR).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.SYSTEM.DEVICE if torch.cuda.is_available() else "cpu")
    conditioner = load_conditioner(cfg, map_location="cpu")
    conditioner.to(device)
    conditioner.eval()

    real_grouped = _real_subject_embeddings(cfg, conditioner, device)
    generated_grouped, similarity_rows = _generated_subject_embeddings(cfg, conditioner, device)
    real_features, real_labels = _flatten_embeddings(real_grouped)
    generated_features, generated_labels = _flatten_embeddings(generated_grouped)

    figure_paths = []
    figure_paths.extend(
        _plot_tsne(
            real_features,
            real_labels,
            "Figure 3: Real Base Vectors",
            output_dir / "figure3_real_tsne",
            bool(cfg.EVAL.PERSONALIZATION.SAVE_PDF),
        )
    )

    combined_features = torch.cat([real_features, generated_features], dim=0) if real_features.numel() and generated_features.numel() else torch.empty(0, 0)
    combined_labels = real_labels + generated_labels
    combined_sources = ["real"] * len(real_labels) + ["generated"] * len(generated_labels)
    if combined_features.numel():
        figure_paths.extend(
            _plot_tsne(
                combined_features,
                combined_labels,
                "Figure 3: Real vs Generated Base Vectors",
                output_dir / "figure3_real_vs_generated_tsne",
                bool(cfg.EVAL.PERSONALIZATION.SAVE_PDF),
                style_labels=combined_sources,
            )
        )

    scaling_paths, scaling_rows = _plot_scaling_curve(
        real_grouped,
        list(cfg.EVAL.PERSONALIZATION.SCALING_PATIENT_COUNTS),
        output_dir,
        bool(cfg.EVAL.PERSONALIZATION.SAVE_PDF),
    )
    figure_paths.extend(scaling_paths)

    generated_similarity = [row["reference_to_generated_cosine"] for row in similarity_rows]
    target_similarity = [row["reference_to_target_cosine"] for row in similarity_rows]
    metrics = {
        "generated_similarity_mean": float(np.mean(generated_similarity)) if generated_similarity else float("nan"),
        "target_similarity_mean": float(np.mean(target_similarity)) if target_similarity else float("nan"),
        "similarity_delta": float(np.mean(generated_similarity) - np.mean(target_similarity))
        if generated_similarity and target_similarity
        else float("nan"),
        "real_silhouette": _safe_silhouette(real_features, real_labels),
        "generated_silhouette": _safe_silhouette(generated_features, generated_labels),
        "combined_silhouette": _safe_silhouette(combined_features, combined_labels) if combined_features.numel() else float("nan"),
        "num_real_subjects": len(real_grouped),
        "num_generated_subjects": len(generated_grouped),
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    csv_lines = ["metric,value"] + [f"{key},{value}" for key, value in metrics.items()]
    (output_dir / "metrics.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    (output_dir / "similarity_rows.json").write_text(json.dumps(similarity_rows, indent=2), encoding="utf-8")
    (output_dir / "scaling_rows.json").write_text(json.dumps(scaling_rows, indent=2), encoding="utf-8")
    write_manifest(
        output_dir / "manifest.json",
        {
            "metrics_path": str(output_dir / "metrics.json"),
            "figure_paths": [str(path) for path in figure_paths],
            "num_real_subjects": len(real_grouped),
            "num_generated_subjects": len(generated_grouped),
        },
    )
    return {
        "output_dir": str(output_dir),
        "metrics_path": str(output_dir / "metrics.json"),
        "metrics_csv_path": str(output_dir / "metrics.csv"),
        "figure_paths": [str(path) for path in figure_paths],
    }
