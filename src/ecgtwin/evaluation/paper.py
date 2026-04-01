"""Static paper figure and table mapping used by report exporters."""

from __future__ import annotations


PAPER_ID = "lai2025_ecgtwin"

PAPER_TARGETS = {
    "table1": {
        "title": "Three-level evaluation result on MIMIC-IV-ECG",
        "metrics_group": "generation_metrics",
    },
    "table2": {
        "title": "Similarity score and silhouette coefficient result",
        "metrics_group": "personalization_metrics",
    },
    "table3": {
        "title": "ECG auto diagnosis test",
        "metrics_group": "pecg_monitor_metrics",
    },
    "table6": {
        "title": "Three-level evaluation result on PTB-XL",
        "metrics_group": "generation_metrics",
    },
    "figure3": {
        "title": "Base vector t-SNE embeddings",
        "figure_group": "personalization",
    },
    "figure4": {
        "title": "Personalized generation case study and attention map",
        "figure_group": "case_study",
    },
    "figure7": {
        "title": "Generated vs real latent scatter",
        "figure_group": "generation",
    },
    "figure8": {
        "title": "Generated vs target heart-rate scatter",
        "figure_group": "generation",
    },
    "figure9": {
        "title": "Scaling-up test on the conditioner",
        "figure_group": "personalization",
    },
    "figure10": {
        "title": "Prompt-to-prompt ECG editing example",
        "figure_group": "editing",
    },
}
