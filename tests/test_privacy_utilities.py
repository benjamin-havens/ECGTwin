from pathlib import Path

import pytest
import torch

from ecgtwin.models.base_vector import BaseVectorBottleneck, apply_base_vector_ablation, apply_random_feature_mask
from ecgtwin.privacy.black_box import black_box_score, domias_score

from ecgtwin.privacy.cli import (
    _build_record_tasks,
    _build_worker_batches,
    _export_reconstruction_examples,
    _filter_overlapping_nonmembers,
    _partition_tasks,
    _resolve_worker_devices,
    _stream_score_rows,
)
from ecgtwin.privacy.data import (
    group_record_indices_by_subject,
    group_records_by_subject,
    record_cache_key_from_record,
    record_id_from_record,
)
from ecgtwin.privacy.metrics import append_csv_rows, auc
from ecgtwin.privacy.metrics import summarize_binary_scores
from ecgtwin.privacy.reconstruction import reconstruction_score
from ecgtwin.privacy.visualization import write_privacy_visualizations


def _make_record(subject_id: int, ecg_time: str, text: str = "sinus rhythm|normal ecg."):
    return {
        "data": torch.ones(4, 8) * subject_id,
        "label": {
            "subject_id": subject_id,
            "ecg_time": ecg_time,
            "text": text,
            "hr": 70.0,
            "age": 60.0,
            "sex": "F",
            "text_embed": torch.ones(2, 768),
        },
    }


def test_base_vector_ablation_modes_and_mask():
    base_vector = torch.ones(2, 4)
    assert torch.equal(apply_base_vector_ablation(base_vector, mode="standard"), base_vector)
    assert torch.equal(apply_base_vector_ablation(base_vector, mode="remove"), torch.zeros_like(base_vector))

    noised = apply_base_vector_ablation(base_vector, mode="noise", noise_std=0.5)
    assert noised.shape == base_vector.shape
    assert not torch.equal(noised, base_vector)

    masked = apply_random_feature_mask(base_vector, mask_prob=1.0)
    assert torch.equal(masked, torch.zeros_like(base_vector))


def test_base_vector_bottleneck_preserves_shape():
    adapter = BaseVectorBottleneck(embed_dim=8, bottleneck_dim=4)
    output = adapter(torch.randn(3, 8))
    assert output.shape == (3, 8)


def test_black_box_and_domias_scores_are_finite():
    query = torch.tensor([0.0, 0.0, 0.0])
    candidates = torch.tensor(
        [
            [0.1, 0.0, 0.0],
            [0.2, 0.1, 0.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ]
    )
    black_box = black_box_score(query, candidates, k=2, distance="l2")
    domias = domias_score(query, candidates, k=2, distance="l2", reference_split=0.5)
    assert isinstance(black_box["score"], float)
    assert isinstance(domias["score"], float)


def test_reconstruction_score_tracks_best_candidate_and_score():
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    candidates = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [0.5, 0.5]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ]
    )
    scored = reconstruction_score(query, candidates, distance="l2")
    assert scored["best_candidate_index"] == 0
    assert scored["distance_min"] == 0.0
    assert scored["score"] == -scored["distance_min"]
    assert scored["cosine_max"] == pytest.approx(1.0)


def test_privacy_metrics_summary_has_expected_fields():
    summary = summarize_binary_scores(labels=[1, 1, 0, 0], scores=[0.9, 0.8, 0.2, 0.1])
    assert summary["roc_auc"] >= 0.99
    assert summary["pr_auc"] >= 0.99
    assert "roc_curve" in summary


def test_auc_works_with_numpy_compatibility_fallback():
    assert auc(torch.tensor([0.0, 1.0]).numpy(), torch.tensor([0.0, 1.0]).numpy()) == 0.5


def test_record_ids_and_grouping_are_subject_aware():
    records = [
        _make_record(1, "2025-01-01 00:00:00"),
        _make_record(1, "2025-01-02 00:00:00"),
        _make_record(2, "2025-01-03 00:00:00"),
    ]
    grouped = group_records_by_subject(records, max_patients=1, max_records_per_patient=1)
    assert list(grouped.keys()) == ["1"]
    assert len(grouped["1"]) == 1
    assert "2025-01-01_00_00_00" in record_id_from_record(records[0])


def test_record_cache_key_is_short_and_stable():
    record = _make_record(123, "2025-01-01 00:00:00", text="very long report " * 20)
    key_a = record_cache_key_from_record(record)
    key_b = record_cache_key_from_record(record)
    assert key_a == key_b
    assert key_a.startswith("123__")
    assert len(key_a) < 32


def test_partition_tasks_balances_work():
    tasks = [{"id": index, "record_count": index + 1} for index in range(7)]
    partitions = _partition_tasks(tasks, 3)
    loads = [sum(task["record_count"] for task in partition) for partition in partitions]
    assert max(loads) - min(loads) <= 3


def test_overlap_filter_removes_nonmember_subjects_only():
    grouped_member = {"1": [0, 1], "2": [2]}
    grouped_nonmember = {"2": [0], "3": [1, 2]}
    filtered_nonmember, overlap = _filter_overlapping_nonmembers(grouped_member, grouped_nonmember)
    assert overlap == ["2"]
    assert filtered_nonmember == {"3": [1, 2]}


def test_build_record_tasks_uses_indices_not_tensor_payloads():
    member_records = [
        _make_record(1, "2025-01-01 00:00:00"),
        _make_record(1, "2025-01-02 00:00:00"),
    ]
    nonmember_records = [_make_record(2, "2025-01-03 00:00:00")]
    member_grouped = group_record_indices_by_subject(member_records)
    nonmember_grouped = group_record_indices_by_subject(nonmember_records)
    tasks = _build_record_tasks(
        "/tmp/member.pt",
        member_grouped,
        "/tmp/nonmember.pt",
        nonmember_grouped,
    )
    assert tasks[0]["dataset_path"] == "/tmp/member.pt"
    assert tasks[0]["record_indices"] == [0, 1]
    assert tasks[0]["record_count"] == 2
    assert tasks[-1]["dataset_path"] == "/tmp/nonmember.pt"
    assert "record" not in tasks[0]
    assert "partner" not in tasks[0]


def test_build_worker_batches_balances_record_counts_and_chunks():
    tasks = [
        {"dataset_path": "/tmp/a.pt", "subject_id": "1", "label": 1, "subset": "member", "record_indices": [0, 1, 2], "record_count": 3},
        {"dataset_path": "/tmp/a.pt", "subject_id": "2", "label": 1, "subset": "member", "record_indices": [3, 4], "record_count": 2},
        {"dataset_path": "/tmp/b.pt", "subject_id": "3", "label": 0, "subset": "nonmember", "record_indices": [0], "record_count": 1},
    ]
    worker_batches = _build_worker_batches(tasks, num_partitions=2, chunk_size=2)
    assert len(worker_batches) == 2
    assert sum(batch["record_count"] for batch in worker_batches) == 6
    assert sorted(batch["record_count"] for batch in worker_batches) == [3, 3]
    assert sum(len(batch["chunks"]) for batch in worker_batches) >= 3


def test_stream_score_rows_merges_shards_and_adds_patient_scores(tmp_path):
    shard_path = tmp_path / "score_shards" / "record_scores_rank0.csv"
    append_csv_rows(
        [
            {"attack": "black_box", "level": "record", "subset": "member", "subject_id": "1", "record_id": "a", "label": 1, "score": 0.9},
            {"attack": "black_box", "level": "record", "subset": "member", "subject_id": "1", "record_id": "b", "label": 1, "score": 0.7},
            {"attack": "black_box", "level": "record", "subset": "nonmember", "subject_id": "2", "record_id": "c", "label": 0, "score": 0.2},
            {"attack": "reconstruction", "level": "record", "subset": "member", "subject_id": "1", "record_id": "a", "label": 1, "score": 0.5},
            {"attack": "reconstruction", "level": "record", "subset": "member", "subject_id": "1", "record_id": "b", "label": 1, "score": 0.1},
            {"attack": "reconstruction", "level": "record", "subset": "nonmember", "subject_id": "2", "record_id": "c", "label": 0, "score": 0.3},
        ],
        shard_path,
    )
    metric_inputs = _stream_score_rows(
        [{"score_shard_path": str(shard_path)}],
        tmp_path / "scores.csv",
        aggregations={"black_box": "max", "reconstruction": "mean"},
    )
    assert ("black_box", "record") in metric_inputs
    assert ("black_box", "patient") in metric_inputs
    assert ("reconstruction", "patient") in metric_inputs
    assert metric_inputs[("reconstruction", "patient")]["scores"] == [0.3, 0.3]
    scores_path = tmp_path / "scores.csv"
    assert scores_path.exists()
    assert len(scores_path.read_text(encoding="utf-8").strip().splitlines()) == 11


def test_write_privacy_visualizations_emits_plot_files(tmp_path):
    scores_path = tmp_path / "scores.csv"
    append_csv_rows(
        [
            {"attack": "black_box", "level": "record", "subset": "member", "subject_id": "1", "record_id": "a", "label": 1, "score": 0.9},
            {"attack": "black_box", "level": "record", "subset": "nonmember", "subject_id": "2", "record_id": "b", "label": 0, "score": 0.2},
            {"attack": "black_box", "level": "patient", "subset": "member", "subject_id": "1", "record_id": "", "label": 1, "score": 0.9},
            {"attack": "black_box", "level": "patient", "subset": "nonmember", "subject_id": "2", "record_id": "", "label": 0, "score": 0.2},
            {"attack": "reconstruction", "level": "record", "subset": "member", "subject_id": "1", "record_id": "a", "label": 1, "score": 0.8},
            {"attack": "reconstruction", "level": "record", "subset": "nonmember", "subject_id": "2", "record_id": "b", "label": 0, "score": 0.1},
        ],
        scores_path,
    )
    roc_rows = [
        {"attack": "black_box", "level": "record", "fpr": 0.0, "tpr": 0.0, "threshold": 1.0},
        {"attack": "black_box", "level": "record", "fpr": 0.0, "tpr": 1.0, "threshold": 0.9},
        {"attack": "black_box", "level": "record", "fpr": 1.0, "tpr": 1.0, "threshold": 0.2},
        {"attack": "reconstruction", "level": "record", "fpr": 0.0, "tpr": 0.0, "threshold": 1.0},
        {"attack": "reconstruction", "level": "record", "fpr": 0.0, "tpr": 1.0, "threshold": 0.8},
        {"attack": "reconstruction", "level": "record", "fpr": 1.0, "tpr": 1.0, "threshold": 0.1},
    ]
    metrics = {
        "black_box:record": {
            "roc_auc": 1.0,
            "pr_auc": 1.0,
            "attack_advantage": 1.0,
            "tpr_at_1pct_fpr": 1.0,
        },
        "reconstruction:record": {
            "roc_auc": 1.0,
            "pr_auc": 1.0,
            "attack_advantage": 1.0,
            "tpr_at_1pct_fpr": 1.0,
        },
    }
    generated = write_privacy_visualizations(tmp_path, roc_rows, metrics, scores_path)
    assert generated
    assert all(path.exists() for path in generated)
    assert (tmp_path / "distributions" / "scores_reconstruction_record.png").exists()


def test_append_csv_rows_preserves_existing_header_order(tmp_path):
    output_path = tmp_path / "mixed.csv"
    append_csv_rows(
        [
            {"attack": "black_box", "level": "record", "label": 1, "score": 0.9, "distance_min": 0.1},
        ],
        output_path,
    )
    append_csv_rows(
        [
            {"attack": "white_box_ibe", "level": "record", "label": 0, "score": 0.2, "gradient_norm": 1.5},
        ],
        output_path,
    )
    rows = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0].split(",") == ["attack", "distance_min", "label", "level", "score"]
    assert "white_box_ibe" in rows[2]
    assert rows[2].split(",")[2] == "0"


def test_resolve_worker_devices_falls_back_to_cpu_without_cuda():
    class PrivacyCfg:
        GPU_IDS = [0, 2, 3]

    class Cfg:
        SYSTEM = type("SystemCfg", (), {"DEVICE": "cuda:0"})
        PRIVACY = PrivacyCfg()

    assert _resolve_worker_devices(Cfg()) == ["cpu"]


def test_export_reconstruction_examples_caps_examples_and_skips_decoding(tmp_path):
    member_dataset_path = tmp_path / "member.pt"
    nonmember_dataset_path = tmp_path / "nonmember.pt"
    member_records = [
        _make_record(1, "2025-01-01 00:00:00"),
        _make_record(1, "2025-01-02 00:00:00"),
    ]
    nonmember_records = [
        _make_record(2, "2025-01-03 00:00:00"),
        _make_record(2, "2025-01-04 00:00:00"),
    ]
    torch.save(member_records, member_dataset_path)
    torch.save(nonmember_records, nonmember_dataset_path)

    synthetic_root = tmp_path / "synthetic"
    member_key_a = record_cache_key_from_record(member_records[0])
    member_key_b = record_cache_key_from_record(member_records[1])
    nonmember_key_a = record_cache_key_from_record(nonmember_records[0])
    nonmember_key_b = record_cache_key_from_record(nonmember_records[1])
    for subset, cache_key in [
        ("member", member_key_a),
        ("member", member_key_b),
        ("nonmember", nonmember_key_a),
        ("nonmember", nonmember_key_b),
    ]:
        payload_path = synthetic_root / subset / f"{cache_key}.pt"
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"latent_gen": torch.stack([torch.zeros(4, 8), torch.ones(4, 8)])}, payload_path)

    scores_path = tmp_path / "scores.csv"
    append_csv_rows(
        [
            {"attack": "reconstruction", "level": "record", "subset": "member", "subject_id": "1", "record_id": "a", "label": 1, "score": 0.9, "distance_min": 0.1, "cosine_max": 0.9, "best_candidate_index": 1, "cache_key": member_key_a, "dataset_path": str(member_dataset_path), "record_index": 0},
            {"attack": "reconstruction", "level": "record", "subset": "member", "subject_id": "1", "record_id": "b", "label": 1, "score": 0.3, "distance_min": 0.3, "cosine_max": 0.7, "best_candidate_index": 0, "cache_key": member_key_b, "dataset_path": str(member_dataset_path), "record_index": 1},
            {"attack": "reconstruction", "level": "record", "subset": "nonmember", "subject_id": "2", "record_id": "c", "label": 0, "score": 0.8, "distance_min": 0.2, "cosine_max": 0.8, "best_candidate_index": 1, "cache_key": nonmember_key_a, "dataset_path": str(nonmember_dataset_path), "record_index": 0},
            {"attack": "reconstruction", "level": "record", "subset": "nonmember", "subject_id": "2", "record_id": "d", "label": 0, "score": 0.1, "distance_min": 0.9, "cosine_max": 0.2, "best_candidate_index": 0, "cache_key": nonmember_key_b, "dataset_path": str(nonmember_dataset_path), "record_index": 1},
        ],
        scores_path,
    )

    class ReconstructionCfg:
        ENABLED = True
        SAVE_EXAMPLE_COUNT_PER_LABEL = 1
        DECODE_EXAMPLES = False

    class PrivacyCfg:
        RECONSTRUCTION = ReconstructionCfg()

    class CheckpointsCfg:
        VAE_PATH = str(tmp_path / "unused.pth")

    class SystemCfg:
        DEVICE = "cpu"

    class Cfg:
        PRIVACY = PrivacyCfg()
        CHECKPOINTS = CheckpointsCfg()
        SYSTEM = SystemCfg()

    exported = _export_reconstruction_examples(Cfg(), scores_path, synthetic_root, tmp_path)
    metrics_files = sorted((tmp_path / "reconstruction_examples").glob("*/*/metrics.json"))
    assert len(exported) == 2
    assert len(metrics_files) == 2
    for metrics_file in metrics_files:
        example_dir = metrics_file.parent
        assert (example_dir / "reference_latent.pt").exists()
        assert (example_dir / "reconstruction_latent.pt").exists()
        assert not (example_dir / "reference_ecg.pt").exists()
