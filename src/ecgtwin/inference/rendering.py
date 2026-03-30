"""Rendering helpers for ECG visualizations."""

from pathlib import Path

from matplotlib import pyplot as plt
import ecg_plot


def save_ecg_plot(signal, target_path: Path, lead_index):
    """Render and save a single ECG plot to disk."""
    ecg_plot.plot(signal, 102.4, lead_index=lead_index, title=None, columns=1, row_height=4)
    plt.savefig(target_path)
    plt.close()
