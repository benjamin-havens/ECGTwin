"""Rendering helpers for ECG visualizations."""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import ecg_plot


def save_ecg_plot(signal, target_path: Path, lead_index):
    """Render and save a single ECG plot to disk."""
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise ValueError(f"ECG plots require a 2D signal array, received shape {signal.shape}")
    if signal.shape[0] != len(lead_index) and signal.shape[1] == len(lead_index):
        signal = signal.transpose(1, 0)
    if signal.shape[0] != len(lead_index):
        raise ValueError(f"ECG plots require {len(lead_index)} leads, received shape {signal.shape}")
    ecg_plot.plot(signal, 102.4, lead_index=lead_index, title=None, columns=1, row_height=4)
    plt.savefig(target_path)
    plt.close()
