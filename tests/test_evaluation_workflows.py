import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ecgtwin.apps.pecg_monitor.generation import build_pecg_generation_tasks
from ecgtwin.config.defaults import get_cfg_defaults
from ecgtwin.evaluation.compare import compare_runs
from ecgtwin.evaluation.generation import build_pair_generation_tasks, generate_batch, generation_index_batches, precision_recall
from ecgtwin.evaluation.personalization import evaluate_personalization
from ecgtwin.evaluation.reproduce_cli import run as run_reproduce
from ecgtwin.inference.generation import build_infer_tasks
from ecgtwin.inference.lightning_predict import resolve_execution_gpu_ids
from ecgtwin.privacy.generation import build_privacy_generation_tasks

try:
    import pytorch_lightning  # noqa: F401
    from ecgtwin.training.vae import VAETrainingModule
except ImportError:
    VAETrainingModule = None


class _DummyConditioner:
    def extract_features(self, latent, text_embed, text_mask, pat_info, reduce=True):
        pooled = latent.mean(dim=1)
        if text_embed is not None:
            pooled = pooled + text_embed.mean(dim=1)[:, : pooled.shape[1]]
        padded_pat_info = torch.nn.functional.pad(pat_info, (0, max(0, pooled.shape[1] - pat_info.shape[1])))
        pooled = pooled + padded_pat_info[:, : pooled.shape[1]]
        if reduce:
            return pooled
        return pooled.unsqueeze(1).repeat(1, latent.shape[1], 1)

    def to(self, device):
        return self

    def eval(self):
        return self


class _DummyDecoder:
    def __call__(self, latents):
        if latents.ndim == 2:
            latents = latents.unsqueeze(0)
        signal = latents.transpose(2, 1).repeat(1, 8, 3)
        return signal[:, :, :12]


class _DummyClipModel:
    def encode_signal(self, ecgs):
        return ecgs.mean(dim=1)

    def ecg_projector(self, embedding):
        return embedding

    def text_projector(self, text_embedding):
        return text_embedding[:, :12]


def _make_pair(subject_id: int, offset: float):
    text_embed = torch.full((2, 12), fill_value=offset + 1.0)
    reference = {
        "data": torch.full((4, 8), fill_value=offset),
        "label": {
            "subject_id": subject_id,
            "ecg_time": f"2025-01-0{subject_id} 00:00:00",
            "text": "sinus rhythm|normal ecg.",
            "hr": 60.0 + offset,
            "age": 50.0,
            "sex": "F",
            "text_embed": text_embed,
        },
    }
    target = {
        "data": torch.full((4, 8), fill_value=offset + 1.0),
        "label": {
            "subject_id": subject_id,
            "ecg_time": f"2025-01-1{subject_id} 00:00:00",
            "text": "sinus tachycardia|abnormal ecg.",
            "hr": 70.0 + offset,
            "age": 51.0,
            "sex": "F",
            "text_embed": text_embed + 1.0,
        },
    }
    return reference, target


@pytest.mark.skipif(VAETrainingModule is None, reason="Lightning is not installed in this environment")
def test_vae_module_exports_legacy_checkpoint(tmp_path):
    cfg = get_cfg_defaults()
    cfg.SYSTEM.DEVICE = "cpu"
    module = VAETrainingModule(cfg)
    batch = torch.randn(2, 1024, 12)
    loss = module.training_step((batch, {}), 0)
    assert torch.isfinite(loss)

    export_path = tmp_path / "vae_model.pth"
    module.export_checkpoint(str(export_path))
    exported = torch.load(export_path, map_location="cpu")
    assert "encoder" in exported
    assert "decoder" in exported


def test_generate_batch_emits_manifest_and_pair_directories(tmp_path, monkeypatch):
    cfg = get_cfg_defaults()
    cfg.SYSTEM.DEVICE = "cpu"
    cfg.EVAL.GENERATION.OUTPUT_DIR = str(tmp_path / "generated")
    cfg.EVAL.GENERATION.GENERATIONS_PER_PAIR = 2
    cfg.EVAL.GENERATION.MAX_PAIRS = 2
    cfg.PATHS.SERIALIZED_DATA_ROOT = str(tmp_path)
    cfg.EVAL.GENERATION.PAIR_DATASET_PATH = "paired.pt"
    torch.save([_make_pair(1, 0.0), _make_pair(2, 2.0)], tmp_path / "paired.pt")

    monkeypatch.setattr(
        "ecgtwin.evaluation.generation.load_generation_runtime",
        lambda cfg: {
            "device": torch.device("cpu"),
            "noise_predictor": object(),
            "conditioner": _DummyConditioner(),
            "decoder": _DummyDecoder(),
            "scheduler": object(),
        },
    )
    monkeypatch.setattr(
        "ecgtwin.evaluation.generation.ddpm_generation",
        lambda diffused_model, noise_predictor, batch_size, device, text_embed, text_embed_mask, pat_info, base_vector, progress_bar: torch.ones(batch_size, 4, 8),
    )
    monkeypatch.setattr(
        "ecgtwin.evaluation.generation.save_ecg_plot",
        lambda signal, target_path, lead_index: Path(target_path).write_text("plot", encoding="utf-8"),
    )

    result = generate_batch(cfg)
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["num_pairs"] == 2
    assert (Path(result["output_dir"]) / "pair_00000" / "metadata.json").exists()
    assert (Path(result["output_dir"]) / "pair_00001" / "generated_01.png").exists()


def test_generate_batch_serializes_numpy_scalar_metadata(tmp_path, monkeypatch):
    cfg = get_cfg_defaults()
    cfg.SYSTEM.DEVICE = "cpu"
    cfg.EVAL.GENERATION.OUTPUT_DIR = str(tmp_path / "generated")
    cfg.EVAL.GENERATION.GENERATIONS_PER_PAIR = 1
    cfg.EVAL.GENERATION.MAX_PAIRS = 1
    cfg.PATHS.SERIALIZED_DATA_ROOT = str(tmp_path)
    cfg.EVAL.GENERATION.PAIR_DATASET_PATH = "paired.pt"
    reference, target = _make_pair(1, 0.0)
    reference["label"]["study_id"] = np.int64(101)
    target["label"]["subject_id"] = np.int64(7)
    torch.save([(reference, target)], tmp_path / "paired.pt")

    monkeypatch.setattr(
        "ecgtwin.evaluation.generation.load_generation_runtime",
        lambda cfg: {
            "device": torch.device("cpu"),
            "noise_predictor": object(),
            "conditioner": _DummyConditioner(),
            "decoder": _DummyDecoder(),
            "scheduler": object(),
        },
    )
    monkeypatch.setattr(
        "ecgtwin.evaluation.generation.ddpm_generation",
        lambda diffused_model, noise_predictor, batch_size, device, text_embed, text_embed_mask, pat_info, base_vector, progress_bar: torch.ones(batch_size, 4, 8),
    )
    monkeypatch.setattr(
        "ecgtwin.evaluation.generation.save_ecg_plot",
        lambda signal, target_path, lead_index: Path(target_path).write_text("plot", encoding="utf-8"),
    )

    result = generate_batch(cfg)
    metadata = json.loads((Path(result["output_dir"]) / "pair_00000" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["reference"]["study_id"] == 101
    assert metadata["target_subject_id"] == "7"


def test_build_pair_generation_tasks_preserves_global_indices(tmp_path):
    cfg = get_cfg_defaults()
    cfg.PATHS.SERIALIZED_DATA_ROOT = str(tmp_path)
    cfg.EVAL.GENERATION.PAIR_DATASET_PATH = "paired.pt"
    torch.save([_make_pair(1, 0.0), _make_pair(2, 2.0)], tmp_path / "paired.pt")
    tasks = build_pair_generation_tasks(cfg)
    assert [task["task_id"] for task in tasks] == ["pair_00000", "pair_00001"]
    assert tasks[1]["pair_index"] == 1


def test_precision_recall_wrapper_returns_finite_values():
    generated = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.1, 0.1]])
    real = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.2, 0.2]])
    metrics = precision_recall(generated, real, k=1)
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0


def test_generation_index_batches_cover_full_range_without_overlap():
    batches = generation_index_batches(num_pairs=10, num_workers=4)
    assert [batch["pair_count"] for batch in batches] == [3, 3, 2, 2]
    assert batches[0]["start_index"] == 0
    assert batches[-1]["end_index"] == 10
    covered = []
    for batch in batches:
        covered.extend(range(batch["start_index"], batch["end_index"]))
    assert covered == list(range(10))


def test_resolve_execution_gpu_ids_uses_shared_then_legacy_fields(monkeypatch):
    cfg = get_cfg_defaults()
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    cfg.EXECUTION.GPU_IDS = [0, 1]
    assert resolve_execution_gpu_ids(cfg, "generation") == [0, 1]
    cfg.EXECUTION.GPU_IDS = []
    cfg.EVAL.GENERATION.GPU_IDS = [2, 3]
    assert resolve_execution_gpu_ids(cfg, "generation") == [2, 3]
    assert resolve_execution_gpu_ids(cfg, "pecg") == [1]


def test_build_privacy_generation_tasks_uses_cache_keys_in_output_paths(tmp_path):
    record = _make_pair(1, 0.0)[0]
    subject_tasks = [{"dataset_path": str(tmp_path / "member.pt"), "subset": "member", "record_indices": [0]}]
    torch.save([record], tmp_path / "member.pt")
    tasks = build_privacy_generation_tasks(subject_tasks, tmp_path / "synthetic")
    assert tasks[0]["task_id"].startswith("1__")
    assert tasks[0]["output_path"].endswith(".pt")


def test_build_pecg_and_infer_tasks_have_stable_output_paths(tmp_path):
    cfg = get_cfg_defaults()
    cfg.PATHS.SERIALIZED_DATA_ROOT = str(tmp_path)
    cfg.APPS.PECG_MONITOR.OUTPUT_DIR = str(tmp_path / "pecg")
    cfg.MODEL.NAME = "DiT_ECGTwin"
    pecg_dataset = {"7": [_make_pair(7, 0.0)[0]]}
    torch.save(pecg_dataset, tmp_path / "clf_test_dataset.pt")
    cfg.APPS.PECG_MONITOR.TEST_DATASET_PATH = "clf_test_dataset.pt"
    pecg_tasks = build_pecg_generation_tasks(cfg)
    assert pecg_tasks[0]["output_path"].endswith("pecg/DiT_ECGTwin/7.pt")

    reference_path = tmp_path / "reference.pt"
    torch.save(_make_pair(3, 0.0)[0], reference_path)
    cfg.PATHS.REFERENCE_SAMPLE = str(reference_path)
    cfg.INFERENCE.SAVE_SAMPLE_PATH = str(tmp_path / "infer")
    infer_tasks = build_infer_tasks(cfg)
    assert infer_tasks[0]["prerequisites"]["tar"]["save_sample_path"] == str(tmp_path / "infer")


def test_evaluate_personalization_writes_metrics_and_figures(tmp_path, monkeypatch):
    generated_root = tmp_path / "generated"
    pair_dir = generated_root / "pair_00000"
    pair_dir.mkdir(parents=True)
    torch.save(torch.ones(4, 8), pair_dir / "latent_ref.pt")
    torch.save(torch.ones(4, 8) * 2, pair_dir / "latent_tar.pt")
    torch.save(torch.ones(2, 4, 8) * 3, pair_dir / "latent_gen.pt")
    torch.save(torch.ones(2, 12), pair_dir / "text_embed_ref.pt")
    torch.save(torch.ones(2, 12), pair_dir / "text_embed_tar.pt")
    torch.save(torch.ones(1, 3), pair_dir / "pat_info_ref.pt")
    torch.save(torch.ones(1, 3), pair_dir / "pat_info_tar.pt")
    (pair_dir / "metadata.json").write_text(
        json.dumps(
            {
                "reference_subject_id": "1",
                "target_subject_id": "1",
                "reference": {"hr": 60.0, "age": 50.0, "sex": "F"},
                "target": {"hr": 70.0, "age": 51.0, "sex": "F"},
            }
        ),
        encoding="utf-8",
    )

    cfg = get_cfg_defaults()
    cfg.SYSTEM.DEVICE = "cpu"
    cfg.EVAL.PERSONALIZATION.GENERATED_ROOT = str(generated_root)
    cfg.EVAL.PERSONALIZATION.OUTPUT_DIR = str(tmp_path / "personalization")
    cfg.EVAL.PERSONALIZATION.SCALING_PATIENT_COUNTS = [1]
    monkeypatch.setattr("ecgtwin.evaluation.personalization.load_conditioner", lambda cfg, map_location="cpu": _DummyConditioner())

    result = evaluate_personalization(cfg)
    metrics = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
    assert "generated_similarity_mean" in metrics
    assert Path(result["output_dir"]).joinpath("figure9_scaling.png").exists()


def test_compare_runs_merges_metrics_and_marks_missing_targets(tmp_path):
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    (baseline_root / "evaluation" / "generation").mkdir(parents=True)
    (candidate_root / "evaluation" / "generation").mkdir(parents=True)
    (candidate_root / "evaluation" / "personalization").mkdir(parents=True)
    (baseline_root / "evaluation" / "generation" / "metrics.json").write_text(json.dumps({"fid": 2.0}), encoding="utf-8")
    (candidate_root / "evaluation" / "generation" / "metrics.json").write_text(json.dumps({"fid": 1.0}), encoding="utf-8")
    (candidate_root / "evaluation" / "generation" / "figure7_latent_scatter.png").write_text("png", encoding="utf-8")

    cfg = get_cfg_defaults()
    cfg.REPORT.BASELINE_ROOT = str(baseline_root)
    cfg.REPORT.CANDIDATE_ROOT = str(candidate_root)
    cfg.REPORT.OUTPUT_DIR = str(tmp_path / "report")
    result = compare_runs(cfg)
    summary = Path(result["summary_path"]).read_text(encoding="utf-8")
    assert "generation.fid" in summary
    assert "figure8" in summary


def test_reproduce_paper_dry_run_resolves_dependencies(tmp_path):
    config_path = tmp_path / "repro.yaml"
    config_path.write_text(
        "REPRO:\n"
        "  RUN_NAME: repro_test\n"
        "  ROOT_DIR: " + str(tmp_path / "runs") + "\n"
        "  DRY_RUN: true\n"
        "  STAGES: [preprocess, compare]\n",
        encoding="utf-8",
    )
    result = run_reproduce([str(config_path)], [])
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["resolved_stage_order"] == ["vae", "preprocess", "text_embed", "pair", "foundation", "diffusion", "generate_batch", "compare"]
