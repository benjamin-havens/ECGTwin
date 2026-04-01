"""Comparison and reporting utilities for baseline-vs-candidate ECGTwin runs."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

from ecgtwin.config import resolve_path
from ecgtwin.evaluation.paper import PAPER_ID, PAPER_TARGETS


KNOWN_METRIC_FILES = {
    "generation": "metrics.json",
    "personalization": "metrics.json",
    "privacy": "metrics.json",
    "pecg": "metrics.json",
}


def _find_metrics_files(root: Path, ignore_dir: Path | None = None) -> dict[str, Path]:
    metric_files = {}
    for path in root.rglob("metrics.json"):
        if ignore_dir is not None and ignore_dir in path.parents:
            continue
        relative = path.relative_to(root)
        parts = {part.lower() for part in relative.parts}
        if "privacy" in parts:
            metric_files.setdefault("privacy", path)
        elif "personalization" in parts:
            metric_files.setdefault("personalization", path)
        elif "generation" in parts:
            metric_files.setdefault("generation", path)
        elif "pecg" in parts or "clf_" in relative.parent.name.lower():
            metric_files.setdefault("pecg", path)
        elif relative.parent.name.lower() in {"generation", "personalization", "privacy"}:
            metric_files.setdefault(relative.parent.name.lower(), path)
    return metric_files


def _read_metric_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metric_bundle(root: Path, ignore_dir: Path | None = None) -> dict[str, dict]:
    """Load stage metrics from a run root."""
    root = resolve_path(root)
    metric_files = _find_metrics_files(root, ignore_dir=ignore_dir)
    return {name: _read_metric_file(path) for name, path in metric_files.items()}


def _find_figure(root: Path, target: str) -> list[Path]:
    return sorted(root.rglob(f"{target}*.png")) + sorted(root.rglob(f"{target}*.pdf"))


def _flatten_metrics(bundle_name: str, metrics: dict[str, dict]) -> list[dict[str, object]]:
    rows = []
    for section, section_metrics in sorted(metrics.items()):
        for metric_name, metric_value in sorted(section_metrics.items()):
            rows.append(
                {
                    "bundle": bundle_name,
                    "section": section,
                    "metric": metric_name,
                    "value": metric_value,
                }
            )
    return rows


def _numeric_delta(baseline_value, candidate_value):
    if not isinstance(baseline_value, (int, float)) or not isinstance(candidate_value, (int, float)):
        return None
    if math.isnan(baseline_value) or math.isnan(candidate_value):
        return None
    return float(candidate_value - baseline_value)


def compare_runs(cfg) -> dict:
    """Build a paper-keyed report bundle comparing baseline and candidate runs."""
    baseline_root = resolve_path(cfg.REPORT.BASELINE_ROOT or cfg.CHECKPOINTS.BASELINE_ROOT)
    candidate_root = resolve_path(cfg.REPORT.CANDIDATE_ROOT or cfg.CHECKPOINTS.CANDIDATE_ROOT)
    output_dir = resolve_path(cfg.REPORT.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = load_metric_bundle(baseline_root, ignore_dir=output_dir) if baseline_root.exists() else {}
    candidate_metrics = load_metric_bundle(candidate_root, ignore_dir=output_dir) if candidate_root.exists() else {}

    comparison_rows = []
    for section in sorted(set(baseline_metrics) | set(candidate_metrics)):
        baseline_section = baseline_metrics.get(section, {})
        candidate_section = candidate_metrics.get(section, {})
        for metric_name in sorted(set(baseline_section) | set(candidate_section)):
            baseline_value = baseline_section.get(metric_name)
            candidate_value = candidate_section.get(metric_name)
            comparison_rows.append(
                {
                    "section": section,
                    "metric": metric_name,
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                    "delta": _numeric_delta(baseline_value, candidate_value),
                }
            )

    metrics_payload = {
        "paper_id": PAPER_ID,
        "paper_pdf_path": cfg.REPORT.PAPER_PDF_PATH,
        "baseline_root": str(baseline_root),
        "candidate_root": str(candidate_root),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "comparisons": comparison_rows,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    with open(output_dir / "metrics.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "metric", "baseline", "candidate", "delta"])
        writer.writeheader()
        writer.writerows(comparison_rows)

    figure_manifest = {}
    for target in cfg.REPORT.TARGETS:
        target_dir = output_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)
        baseline_figures = _find_figure(baseline_root, target) if baseline_root.exists() else []
        candidate_figures = _find_figure(candidate_root, target) if candidate_root.exists() else []
        exported = []
        for label, paths in (("baseline", baseline_figures), ("candidate", candidate_figures)):
            if not paths:
                continue
            for path in paths:
                destination = target_dir / f"{label}_{path.name}"
                shutil.copy2(path, destination)
                exported.append(str(destination))
        figure_manifest[target] = {
            "category": PAPER_TARGETS.get(target, "unknown"),
            "baseline_count": len(baseline_figures),
            "candidate_count": len(candidate_figures),
            "exported": exported,
            "available": bool(exported),
        }

    summary_lines = [
        "# ECGTwin Paper Comparison",
        "",
        f"- paper_pdf_path: {cfg.REPORT.PAPER_PDF_PATH}",
        f"- baseline_root: {baseline_root}",
        f"- candidate_root: {candidate_root}",
        "",
        "## Metric Deltas",
        "",
    ]
    if comparison_rows:
        for row in comparison_rows:
            summary_lines.append(
                f"- {row['section']}.{row['metric']}: baseline={row['baseline']} candidate={row['candidate']} delta={row['delta']}"
            )
    else:
        summary_lines.append("- no comparable metrics found")
    summary_lines.append("")
    summary_lines.append("## Paper Targets")
    summary_lines.append("")
    for target in cfg.REPORT.TARGETS:
        manifest_row = figure_manifest[target]
        if manifest_row["available"]:
            summary_lines.append(f"- {target}: exported {len(manifest_row['exported'])} figure files")
        else:
            summary_lines.append(f"- {target}: unavailable from supplied artifacts")
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (output_dir / "figures.json").write_text(json.dumps(figure_manifest, indent=2), encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "metrics_path": str(output_dir / "metrics.json"),
        "metrics_csv_path": str(output_dir / "metrics.csv"),
        "summary_path": str(output_dir / "summary.md"),
    }
