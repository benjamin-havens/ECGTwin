"""Visualization helpers for privacy-audit outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .metrics import iter_csv_rows


def _save_figure(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_roc_curves(roc_rows: list[dict], output_path: Path) -> None:
    """Render ROC curves grouped by attack and audit level."""
    if not roc_rows:
        return
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in roc_rows:
        grouped.setdefault((row["attack"], row["level"]), []).append(row)

    fig, ax = plt.subplots(figsize=(8, 6))
    for (attack, level), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row["fpr"])
        ax.plot(
            [row["fpr"] for row in rows],
            [row["tpr"] for row in rows],
            label=f"{attack}:{level}",
            linewidth=2,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Privacy Audit ROC Curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    _save_figure(output_path)


def plot_score_distributions(scores_path: Path, output_dir: Path) -> None:
    """Render per-attack score histograms for member and nonmember samples."""
    grouped: dict[tuple[str, str], dict[int, list[float]]] = {}
    for row in iter_csv_rows(scores_path):
        try:
            label_value = int(row["label"])
            score_value = float(row["score"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (row.get("attack", "unknown"), row.get("level", "unknown"))
        grouped.setdefault(key, {0: [], 1: []})
        if label_value in {0, 1}:
            grouped[key][label_value].append(score_value)

    for (attack, level), by_label in grouped.items():
        if not by_label[0] and not by_label[1]:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = 60
        if by_label[1]:
            ax.hist(by_label[1], bins=bins, alpha=0.55, density=True, label="member", color="#b33a3a")
        if by_label[0]:
            ax.hist(by_label[0], bins=bins, alpha=0.55, density=True, label="nonmember", color="#2f6db3")
        ax.set_title(f"Score Distribution: {attack} ({level})")
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.2)
        _save_figure(output_dir / f"scores_{attack}_{level}.png")


def plot_metric_heatmap(metrics: dict[str, dict[str, float]], output_path: Path) -> None:
    """Render a compact heatmap over key privacy metrics."""
    if not metrics:
        return
    metric_names = ["roc_auc", "pr_auc", "attack_advantage", "tpr_at_1pct_fpr"]
    row_labels = sorted(metrics)
    values = np.array(
        [
            [float(metrics[row_label].get(metric_name, 0.0)) for metric_name in metric_names]
            for row_label in row_labels
        ],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(9, max(3, 0.55 * len(row_labels))))
    image = ax.imshow(values, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=max(1.0, float(values.max(initial=1.0))))
    ax.set_xticks(range(len(metric_names)), labels=metric_names, rotation=20, ha="right")
    ax.set_yticks(range(len(row_labels)), labels=row_labels)
    ax.set_title("Privacy Audit Metric Heatmap")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            ax.text(column_index, row_index, f"{values[row_index, column_index]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _save_figure(output_path)


def write_privacy_visualizations(
    output_dir: Path,
    roc_rows: list[dict],
    metrics: dict[str, dict[str, float]],
    scores_path: Path,
) -> list[Path]:
    """Create the standard visualization bundle for a privacy-audit run."""
    generated_paths: list[Path] = []
    roc_path = output_dir / "roc_curves.png"
    plot_roc_curves(roc_rows, roc_path)
    if roc_path.exists():
        generated_paths.append(roc_path)

    heatmap_path = output_dir / "metric_heatmap.png"
    plot_metric_heatmap(metrics, heatmap_path)
    if heatmap_path.exists():
        generated_paths.append(heatmap_path)

    distribution_dir = output_dir / "distributions"
    plot_score_distributions(scores_path, distribution_dir)
    generated_paths.extend(sorted(distribution_dir.glob("*.png")))
    return generated_paths
