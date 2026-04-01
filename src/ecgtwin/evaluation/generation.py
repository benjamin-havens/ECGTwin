"""Batch generation, metrics, and figure export for paper-style evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import sqrtm
from scipy.signal import find_peaks

from ecgtwin.config import resolve_serialized_data_path
from ecgtwin.core.runtime_env import configure_runtime_environment
from ecgtwin.data.patient import build_patient_info_tensor, sex_to_binary
from ecgtwin.evaluation.artifacts import json_ready, read_manifest, write_manifest
from ecgtwin.evaluation.runtime import load_clip_runtime, load_generation_runtime
from ecgtwin.inference.generation import ddpm_generation
from ecgtwin.inference.rendering import save_ecg_plot
from ecgtwin.privacy.data import load_dataset_file, record_id_from_record, subject_id_from_record
from ecgtwin.privacy.features import sample_patient_tensor, sample_text_tensors


LEAD_INDEX = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]

configure_runtime_environment()

try:
    from sklearn.decomposition import PCA as SklearnPCA
    from sklearn.manifold import TSNE as SklearnTSNE
except ImportError:  # pragma: no cover - exercised indirectly when sklearn is absent
    SklearnPCA = None
    SklearnTSNE = None


def _pca_2d(array: np.ndarray) -> np.ndarray:
    centered = array - array.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    projected = centered @ basis
    if projected.shape[1] < 2:
        projected = np.pad(projected, ((0, 0), (0, 2 - projected.shape[1])))
    return projected[:, :2]


def _project_2d(array: np.ndarray) -> np.ndarray:
    if array.shape[0] < 3 or SklearnTSNE is None:
        return _pca_2d(array)
    perplexity = min(30, array.shape[0] - 1)
    if perplexity < 2:
        return _pca_2d(array)
    return SklearnTSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(array)


def _label_metadata(record: dict) -> dict:
    return json_ready(record.get("label", {}))


def _save_optional_tensor(path: Path, payload) -> None:
    if payload is None:
        return
    if isinstance(payload, torch.Tensor):
        payload = payload.detach().cpu()
    torch.save(payload, path)


def _pair_output_root(cfg) -> Path:
    return Path(cfg.EVAL.GENERATION.OUTPUT_DIR).expanduser().resolve()


def _pair_dataset_path(cfg) -> Path:
    dataset_path = cfg.EVAL.GENERATION.PAIR_DATASET_PATH or cfg.DATA.TEST_DATASET_PATH or cfg.DATA.DATASET_PATH
    return resolve_serialized_data_path(cfg, dataset_path)


def _decode_latents(decoder, latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim == 2:
        latents = latents.unsqueeze(0)
    try:
        decoder_device = next(decoder.parameters()).device
    except (AttributeError, StopIteration):
        decoder_device = latents.device
    return decoder(latents.to(device=decoder_device, dtype=torch.float32))


def _estimate_heart_rate_from_signal(ecg_signal: np.ndarray, sample_rate_hz: float = 100.0) -> float:
    lead = ecg_signal[:, 1] if ecg_signal.shape[1] > 1 else ecg_signal[:, 0]
    centered = lead - np.mean(lead)
    scale = np.std(centered)
    if not np.isfinite(scale) or scale == 0:
        return float("nan")
    peaks, _ = find_peaks(centered, distance=max(int(sample_rate_hz * 0.35), 1), prominence=scale * 0.3)
    if len(peaks) < 2:
        return float("nan")
    rr = np.diff(peaks) / sample_rate_hz
    rr = rr[rr > 0]
    if rr.size == 0:
        return float("nan")
    return float(60.0 / np.mean(rr))


def estimate_heart_rate_batch(ecgs: torch.Tensor, sample_rate_hz: float = 100.0) -> list[float]:
    """Estimate heart rate for each ECG in a batch."""
    return [_estimate_heart_rate_from_signal(ecg.detach().cpu().numpy(), sample_rate_hz) for ecg in ecgs]


def clip_score_saved_samples(sample_root: Path, clip_model, decoder, device: torch.device) -> dict[str, float]:
    """Compute CLIP score over a directory of generated pair outputs."""
    total_score = 0.0
    total_count = 0
    for pair_dir in iter_pair_directories(sample_root):
        latent_gen = torch.load(pair_dir / "latent_gen.pt", map_location="cpu").to(device)
        text_embedding = torch.load(pair_dir / "text_embed_tar.pt", map_location="cpu")
        if text_embedding.ndim == 2:
            text_embedding = text_embedding.unsqueeze(0)
        text_embedding = torch.mean(text_embedding, dim=1).to(device)

        ecgs = decoder(latent_gen) if decoder is not None else latent_gen.transpose(2, 1)
        signal_embedding = clip_model.encode_signal(ecgs)
        signal_features = clip_model.ecg_projector(signal_embedding)
        text_features = clip_model.text_projector(text_embedding).repeat(signal_features.shape[0], 1)
        signal_features = signal_features / signal_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        total_score += float(torch.trace(signal_features @ text_features.t()).item())
        total_count += int(signal_features.shape[0])
    return {"clip_score": total_score / total_count if total_count else float("nan"), "num_samples": total_count}


def _feature_tensor_from_pair_dir(pair_dir: Path, clip_model, decoder, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    latent_gen = torch.load(pair_dir / "latent_gen.pt", map_location="cpu").to(device)
    latent_tar = torch.load(pair_dir / "latent_tar.pt", map_location="cpu").to(device)
    if latent_tar.ndim == 2:
        latent_tar = latent_tar.unsqueeze(0)

    gen_ecgs = decoder(latent_gen)
    tar_ecgs = decoder(latent_tar)
    gen_features = clip_model.ecg_projector(clip_model.encode_signal(gen_ecgs)).detach().cpu()
    tar_features = clip_model.ecg_projector(clip_model.encode_signal(tar_ecgs)).detach().cpu()
    return gen_features, tar_features


def generate_feature_matrices(sample_root: Path, clip_model, decoder, device: torch.device) -> dict[str, torch.Tensor]:
    """Project generated and real targets into CLIP feature space."""
    generated = []
    real = []
    for pair_dir in iter_pair_directories(sample_root):
        gen_features, tar_features = _feature_tensor_from_pair_dir(pair_dir, clip_model, decoder, device)
        generated.append(gen_features)
        real.append(tar_features)
    if not generated:
        return {"gen": torch.empty(0, 0), "real": torch.empty(0, 0)}
    return {"gen": torch.cat(generated, dim=0), "real": torch.cat(real, dim=0)}


def fid_score(features_a: torch.Tensor, features_b: torch.Tensor) -> float:
    """Compute Fréchet distance between two feature sets."""
    if features_a.numel() == 0 or features_b.numel() == 0:
        return float("nan")
    array_a = features_a.detach().cpu().numpy()
    array_b = features_b.detach().cpu().numpy()
    mu_a, sigma_a = array_a.mean(axis=0), np.cov(array_a, rowvar=False)
    mu_b, sigma_b = array_b.mean(axis=0), np.cov(array_b, rowvar=False)
    if np.ndim(sigma_a) == 0:
        sigma_a = np.array([[sigma_a]])
    if np.ndim(sigma_b) == 0:
        sigma_b = np.array([[sigma_b]])
    sigma_a = sigma_a + np.eye(sigma_a.shape[0]) * 1.0e-6
    sigma_b = sigma_b + np.eye(sigma_b.shape[0]) * 1.0e-6
    covmean = sqrtm(sigma_a.dot(sigma_b))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    mean_diff = np.sum((mu_a - mu_b) ** 2.0)
    return float(mean_diff + np.trace(sigma_a + sigma_b - 2.0 * covmean))


class ManifoldDetector:
    """k-NN manifold approximation used for precision/recall generation metrics."""

    def __init__(self, data: torch.Tensor, k: int = 3):
        self.k = max(int(k), 1)
        self.data = data
        if data.shape[0] == 0:
            self.radii = torch.empty(0)
            return
        distances = torch.cdist(data, data)
        neighbors = min(self.k + 1, data.shape[0])
        _, indices = torch.topk(distances, k=neighbors, dim=1, largest=False)
        indices = indices[:, 1:] if neighbors > 1 else indices
        if indices.numel() == 0:
            self.radii = torch.zeros(data.shape[0], 1)
        else:
            self.radii = torch.gather(distances, 1, indices[:, -1].view(-1, 1))


def _points_in_manifold(test_points: torch.Tensor, manifold: ManifoldDetector) -> int:
    if test_points.shape[0] == 0 or manifold.data.shape[0] == 0:
        return 0
    distances = torch.cdist(test_points, manifold.data)
    inside = distances <= manifold.radii.squeeze(1).unsqueeze(0)
    return int(inside.any(dim=1).sum().item())


def precision_recall(generated: torch.Tensor, real: torch.Tensor, k: int = 3) -> dict[str, float]:
    """Compute precision, recall, and F1 using the ECGTwin research metric."""
    if generated.shape[0] == 0 or real.shape[0] == 0:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    generated = generated.float()
    real = real.float()
    manifold_generated = ManifoldDetector(generated, k=k)
    manifold_real = ManifoldDetector(real, k=k)
    precision = _points_in_manifold(generated, manifold_real) / max(generated.shape[0], 1)
    recall = _points_in_manifold(real, manifold_generated) / max(real.shape[0], 1)
    if precision == 0 or recall == 0:
        f1_score_value = 0.0
    else:
        f1_score_value = float(2.0 / ((1.0 / precision) + (1.0 / recall)))
    return {"precision": float(precision), "recall": float(recall), "f1": f1_score_value}


def iter_pair_directories(sample_root: Path) -> list[Path]:
    """Return generated pair directories ordered by pair index."""
    if not sample_root.exists():
        return []
    return sorted(
        [path for path in sample_root.iterdir() if path.is_dir() and (path / "metadata.json").exists()],
        key=lambda path: path.name,
    )


def build_pair_generation_tasks(cfg) -> list[dict]:
    """Build stable per-pair generation tasks for the paired evaluation dataset."""
    output_dir = _pair_output_root(cfg)
    paired_dataset = load_dataset_file(str(_pair_dataset_path(cfg)))
    max_pairs = min(len(paired_dataset), int(cfg.EVAL.GENERATION.MAX_PAIRS))
    tasks = []
    for index in range(max_pairs):
        reference, target = paired_dataset[index]
        tasks.append(
            {
                "task_id": f"pair_{index:05d}",
                "pair_index": index,
                "reference": reference,
                "target": target,
                "output_dir": str(output_dir / f"pair_{index:05d}"),
                "generation_batch": int(cfg.EVAL.GENERATION.GENERATIONS_PER_PAIR),
            }
        )
    return tasks


def write_pair_generation_artifacts(task: dict, runtime: dict, cfg) -> dict:
    """Generate and persist artifacts for one paired-reference task."""
    pair_dir = Path(task["output_dir"])
    pair_dir.mkdir(parents=True, exist_ok=True)
    reference = task["reference"]
    target = task["target"]
    device = runtime["device"]

    reference_latent = reference["data"].unsqueeze(0).to(device=device, dtype=torch.float32)
    reference_tokens = reference_latent.transpose(2, 1)
    ref_text_embed, ref_text_mask = sample_text_tensors(reference, device)
    ref_patient = sample_patient_tensor(reference, device)
    base_vector = runtime["conditioner"].extract_features(
        reference_tokens,
        ref_text_embed,
        ref_text_mask,
        ref_patient,
        reduce=True,
    )
    batch_size = int(task["generation_batch"])
    base_vector = base_vector.repeat(batch_size, 1)
    target_text_embed, target_text_mask = sample_text_tensors(target, device, repeat=batch_size)
    target_patient = sample_patient_tensor(target, device, repeat=batch_size)
    latent_gen = ddpm_generation(
        diffused_model=runtime["scheduler"],
        noise_predictor=runtime["noise_predictor"],
        batch_size=batch_size,
        device=device,
        text_embed=target_text_embed,
        text_embed_mask=target_text_mask,
        pat_info=target_patient,
        base_vector=base_vector,
        progress_bar=False,
    ).detach().cpu()

    torch.save(reference["data"].detach().cpu(), pair_dir / "latent_ref.pt")
    torch.save(target["data"].detach().cpu(), pair_dir / "latent_tar.pt")
    torch.save(latent_gen, pair_dir / "latent_gen.pt")
    _save_optional_tensor(pair_dir / "text_embed_ref.pt", reference["label"].get("text_embed"))
    _save_optional_tensor(pair_dir / "text_mask_ref.pt", reference["label"].get("text_embed_mask"))
    _save_optional_tensor(pair_dir / "text_embed_tar.pt", target["label"].get("text_embed"))
    _save_optional_tensor(pair_dir / "text_mask_tar.pt", target["label"].get("text_embed_mask"))
    torch.save(sample_patient_tensor(reference, torch.device("cpu")), pair_dir / "pat_info_ref.pt")
    torch.save(sample_patient_tensor(target, torch.device("cpu")), pair_dir / "pat_info_tar.pt")
    torch.save(base_vector[0:1].detach().cpu(), pair_dir / "base_vector.pt")

    reference_ecg = _decode_latents(runtime["decoder"], reference["data"].unsqueeze(0)).squeeze(0).detach().cpu().numpy()
    target_ecg = _decode_latents(runtime["decoder"], target["data"].unsqueeze(0)).squeeze(0).detach().cpu().numpy()
    generated_ecg = _decode_latents(runtime["decoder"], latent_gen).detach().cpu()
    save_ecg_plot(reference_ecg, pair_dir / "reference_ecg.png", LEAD_INDEX)
    save_ecg_plot(target_ecg, pair_dir / "target_ecg.png", LEAD_INDEX)
    for sample_index, sample in enumerate(generated_ecg):
        save_ecg_plot(sample.numpy(), pair_dir / f"generated_{sample_index:02d}.png", LEAD_INDEX)

    metadata = {
        "pair_index": int(task["pair_index"]),
        "reference_record_id": record_id_from_record(reference, index=int(task["pair_index"])),
        "target_record_id": record_id_from_record(target, index=int(task["pair_index"])),
        "reference_subject_id": subject_id_from_record(reference),
        "target_subject_id": subject_id_from_record(target),
        "reference": _label_metadata(reference),
        "target": _label_metadata(target),
        "generation_batch": batch_size,
    }
    (pair_dir / "metadata.json").write_text(json.dumps(json_ready(metadata), indent=2), encoding="utf-8")
    return {"task_id": task["task_id"], "output_dir": str(pair_dir), "pair_index": int(task["pair_index"])}


def write_generation_manifest(cfg, output_dir: Path | None = None) -> Path:
    """Rebuild the top-level generation manifest from completed pair directories."""
    output_dir = output_dir or _pair_output_root(cfg)
    pair_rows = []
    for pair_dir in iter_pair_directories(output_dir):
        metadata = json.loads((pair_dir / "metadata.json").read_text(encoding="utf-8"))
        pair_rows.append({"pair_dir": str(pair_dir), **metadata})
    return write_manifest(
        output_dir / "manifest.json",
        {
            "pair_dataset_path": str(_pair_dataset_path(cfg)),
            "num_pairs": len(pair_rows),
            "generations_per_pair": int(cfg.EVAL.GENERATION.GENERATIONS_PER_PAIR),
            "pairs": pair_rows,
        },
    )


def generation_index_batches(num_pairs: int, num_workers: int) -> list[dict[str, int]]:
    """Partition global pair indices into contiguous worker slices."""
    if num_pairs <= 0 or num_workers <= 0:
        return []
    active_workers = min(num_pairs, num_workers)
    base = num_pairs // active_workers
    remainder = num_pairs % active_workers
    batches = []
    start = 0
    for rank in range(active_workers):
        length = base + (1 if rank < remainder else 0)
        end = start + length
        batches.append({"start_index": start, "end_index": end, "pair_count": length})
        start = end
    return batches


def generate_batch(cfg, start_index: int = 0, end_index: int | None = None, write_manifest_file: bool = True) -> dict:
    """Generate evaluation samples for an entire paired dataset or one index shard."""
    output_dir = _pair_output_root(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = load_generation_runtime(cfg)
    tasks = build_pair_generation_tasks(cfg)
    max_pairs = len(tasks)
    end_index = max_pairs if end_index is None else min(int(end_index), max_pairs)
    start_index = min(max(int(start_index), 0), end_index)

    with torch.no_grad():
        for task in tasks[start_index:end_index]:
            write_pair_generation_artifacts(task, runtime, cfg)

    manifest_path = write_generation_manifest(cfg, output_dir=output_dir) if write_manifest_file else output_dir / "manifest.json"
    return {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "start_index": start_index,
        "end_index": end_index,
        "generated_pairs": end_index - start_index,
    }


def _plot_latent_scatter(output_dir: Path, pair_dirs: list[Path], sample_limit: int, save_pdf: bool) -> list[Path]:
    if not pair_dirs:
        return []
    target_vectors = []
    generated_vectors = []
    for pair_dir in pair_dirs[:sample_limit]:
        target_vectors.append(torch.load(pair_dir / "latent_tar.pt", map_location="cpu").reshape(-1))
        generated = torch.load(pair_dir / "latent_gen.pt", map_location="cpu")[0].reshape(-1)
        generated_vectors.append(generated)
    combined = torch.stack(target_vectors + generated_vectors).numpy()
    embedding = _project_2d(combined)
    split = len(target_vectors)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(embedding[:split, 0], embedding[:split, 1], label="real target", alpha=0.8)
    ax.scatter(embedding[split:, 0], embedding[split:, 1], label="generated", alpha=0.8)
    ax.set_title("Figure 7: Latent Scatter")
    ax.legend()
    png_path = output_dir / "figure7_latent_scatter.png"
    fig.savefig(png_path, bbox_inches="tight")
    generated_paths = [png_path]
    if save_pdf:
        pdf_path = output_dir / "figure7_latent_scatter.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        generated_paths.append(pdf_path)
    plt.close(fig)
    return generated_paths


def _plot_heart_rate_scatter(output_dir: Path, pair_dirs: list[Path], decoder, save_pdf: bool) -> tuple[list[Path], dict[str, float]]:
    target_hr = []
    generated_hr = []
    for pair_dir in pair_dirs:
        metadata = json.loads((pair_dir / "metadata.json").read_text(encoding="utf-8"))
        target_value = float(metadata["target"].get("hr", float("nan")))
        latents = torch.load(pair_dir / "latent_gen.pt", map_location="cpu")
        generated_ecgs = _decode_latents(decoder, latents).detach().cpu()
        estimated = [value for value in estimate_heart_rate_batch(generated_ecgs) if np.isfinite(value)]
        if not estimated or not np.isfinite(target_value):
            continue
        target_hr.extend([target_value] * len(estimated))
        generated_hr.extend(estimated)

    if not target_hr:
        return [], {"hr_mae": float("nan"), "num_hr_samples": 0}

    target_array = np.asarray(target_hr)
    generated_array = np.asarray(generated_hr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(target_array, generated_array, alpha=0.7)
    low = float(min(target_array.min(), generated_array.min()))
    high = float(max(target_array.max(), generated_array.max()))
    ax.plot([low, high], [low, high], linestyle="--", color="black")
    ax.set_xlabel("Target HR")
    ax.set_ylabel("Generated HR")
    ax.set_title("Figure 8: Heart Rate Scatter")
    png_path = output_dir / "figure8_hr_scatter.png"
    fig.savefig(png_path, bbox_inches="tight")
    generated_paths = [png_path]
    if save_pdf:
        pdf_path = output_dir / "figure8_hr_scatter.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        generated_paths.append(pdf_path)
    plt.close(fig)
    return generated_paths, {"hr_mae": float(np.mean(np.abs(target_array - generated_array))), "num_hr_samples": int(len(target_array))}


def _plot_case_study(output_dir: Path, pair_dirs: list[Path], decoder, save_pdf: bool) -> list[Path]:
    if not pair_dirs:
        return []
    pair_dir = pair_dirs[0]
    ref = _decode_latents(decoder, torch.load(pair_dir / "latent_ref.pt", map_location="cpu").unsqueeze(0)).squeeze(0).detach().cpu().numpy()
    tar = _decode_latents(decoder, torch.load(pair_dir / "latent_tar.pt", map_location="cpu").unsqueeze(0)).squeeze(0).detach().cpu().numpy()
    gen = _decode_latents(decoder, torch.load(pair_dir / "latent_gen.pt", map_location="cpu")[0:1]).squeeze(0).detach().cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, signal, title in zip(axes, [ref, tar, gen], ["Reference", "Target", "Generated"], strict=True):
        ax.plot(signal[:, 1] if signal.shape[1] > 1 else signal[:, 0], linewidth=1.0)
        ax.set_title(title)
    axes[-1].set_xlabel("Time")
    fig.suptitle("Figure 4: Qualitative Case Study")
    png_path = output_dir / "figure4_case_study.png"
    fig.savefig(png_path, bbox_inches="tight")
    generated_paths = [png_path]
    if save_pdf:
        pdf_path = output_dir / "figure4_case_study.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        generated_paths.append(pdf_path)
    plt.close(fig)
    return generated_paths


def evaluate_generation(cfg) -> dict:
    """Score generated outputs and export the paper-style generation figures."""
    output_dir = _pair_output_root(cfg)
    if not (output_dir / "manifest.json").exists():
        raise FileNotFoundError(f"Generation manifest not found at {output_dir / 'manifest.json'}")

    runtime = load_generation_runtime(cfg)
    clip_runtime = load_clip_runtime(cfg)
    pair_dirs = iter_pair_directories(output_dir)

    clip_result = clip_score_saved_samples(output_dir, clip_runtime["clip_model"], runtime["decoder"], clip_runtime["device"])
    matrices = generate_feature_matrices(output_dir, clip_runtime["clip_model"], runtime["decoder"], clip_runtime["device"])
    fid_value = fid_score(matrices["real"], matrices["gen"])
    precision_recall_result = precision_recall(
        matrices["gen"],
        matrices["real"],
        k=int(cfg.EVAL.GENERATION.K_NEIGHBORS),
    )

    figures = []
    figures.extend(_plot_case_study(output_dir, pair_dirs, runtime["decoder"], bool(cfg.EVAL.GENERATION.SAVE_PDF)))
    figures.extend(
        _plot_latent_scatter(
            output_dir,
            pair_dirs,
            sample_limit=int(cfg.EVAL.GENERATION.SCATTER_SAMPLES),
            save_pdf=bool(cfg.EVAL.GENERATION.SAVE_PDF),
        )
    )
    hr_figure_paths, hr_metrics = _plot_heart_rate_scatter(output_dir, pair_dirs, runtime["decoder"], bool(cfg.EVAL.GENERATION.SAVE_PDF))
    figures.extend(hr_figure_paths)

    metrics = {
        "clip_score": clip_result["clip_score"],
        "num_generated_samples": clip_result["num_samples"],
        "fid": fid_value,
        "precision": precision_recall_result["precision"],
        "recall": precision_recall_result["recall"],
        "f1": precision_recall_result["f1"],
        "hr_mae": hr_metrics["hr_mae"],
        "num_hr_samples": hr_metrics["num_hr_samples"],
    }
    (output_dir / "metrics.json").write_text(json.dumps(json_ready(metrics), indent=2), encoding="utf-8")
    csv_lines = ["metric,value"] + [f"{key},{value}" for key, value in metrics.items()]
    (output_dir / "metrics.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    write_manifest(
        output_dir / "evaluation_manifest.json",
        {
            "source_manifest": read_manifest(output_dir / "manifest.json"),
            "metrics_path": str(output_dir / "metrics.json"),
            "figure_paths": [str(path) for path in figures],
        },
    )
    return {
        "output_dir": str(output_dir),
        "metrics_path": str(output_dir / "metrics.json"),
        "metrics_csv_path": str(output_dir / "metrics.csv"),
        "figure_paths": [str(path) for path in figures],
    }
