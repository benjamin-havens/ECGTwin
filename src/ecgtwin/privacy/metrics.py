"""Binary-classification metrics used for privacy attack reports."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def roc_curve(labels: list[int], scores: list[float]):
    """Compute an ROC curve without depending on scikit-learn."""
    labels_np = np.asarray(labels, dtype=np.int64)
    scores_np = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores_np, kind="mergesort")
    labels_np = labels_np[order]
    scores_np = scores_np[order]

    positives = max(int(labels_np.sum()), 1)
    negatives = max(int((1 - labels_np).sum()), 1)

    tps = np.cumsum(labels_np)
    fps = np.cumsum(1 - labels_np)
    tpr = np.concatenate(([0.0], tps / positives))
    fpr = np.concatenate(([0.0], fps / negatives))
    thresholds = np.concatenate(([np.inf], scores_np))
    return fpr, tpr, thresholds


def precision_recall_curve(labels: list[int], scores: list[float]):
    """Compute a precision-recall curve without third-party metrics packages."""
    labels_np = np.asarray(labels, dtype=np.int64)
    scores_np = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores_np, kind="mergesort")
    labels_np = labels_np[order]

    tps = np.cumsum(labels_np)
    fps = np.cumsum(1 - labels_np)
    positives = max(int(labels_np.sum()), 1)
    precision = np.concatenate(([1.0], tps / np.maximum(tps + fps, 1)))
    recall = np.concatenate(([0.0], tps / positives))
    return precision, recall


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute trapezoidal area under a curve."""
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(y, x))
    return float(np.trapz(y, x))


def summarize_binary_scores(labels: list[int], scores: list[float]) -> dict[str, float]:
    """Summarize attack scores into ROC/PR metrics."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    precision, recall = precision_recall_curve(labels, scores)
    attack_advantage = float(np.max(tpr - fpr))

    mask = fpr <= 0.01
    tpr_at_1pct_fpr = float(np.max(tpr[mask])) if mask.any() else 0.0
    return {
        "num_examples": float(len(labels)),
        "roc_auc": auc(fpr, tpr),
        "pr_auc": auc(recall, precision),
        "attack_advantage": attack_advantage,
        "tpr_at_1pct_fpr": tpr_at_1pct_fpr,
        "roc_curve": [
            {"fpr": float(fp), "tpr": float(tp), "threshold": float(th)}
            for fp, tp, th in zip(fpr, tpr, thresholds, strict=True)
        ],
    }


def write_csv(rows: list[dict], output_path: Path) -> None:
    """Write a list of dictionaries to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv_rows(rows: list[dict], output_path: Path) -> None:
    """Append rows to a CSV file, writing a header on first use."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    file_exists = output_path.exists() and output_path.stat().st_size > 0
    if file_exists:
        with output_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            fieldnames = next(reader)
    else:
        fieldnames = sorted({key for row in rows for key in row})

    normalized_rows = [{field: row.get(field, "") for field in fieldnames} for row in rows]
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(normalized_rows)


def read_csv_rows(input_path: Path) -> list[dict]:
    """Read a CSV file back into a list of dictionaries."""
    if not input_path.exists() or input_path.stat().st_size == 0:
        return []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_csv_rows(input_path: Path):
    """Iterate over CSV rows without loading the whole file into memory."""
    if not input_path.exists() or input_path.stat().st_size == 0:
        return
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row
