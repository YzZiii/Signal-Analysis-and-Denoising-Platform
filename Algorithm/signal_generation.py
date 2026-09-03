"""Synthetic single-channel MEG-like signal generation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import SignalConfig

FloatArray = NDArray[np.float64]


def generate_signal(config: SignalConfig) -> tuple[FloatArray, FloatArray]:
    """Return the time axis and a deterministic clean synthetic signal."""
    config.validate()
    sample_count = int(round(config.sample_rate * config.duration))
    time = np.arange(sample_count, dtype=float) / config.sample_rate
    signal = (
        config.alpha_amplitude
        * np.sin(2.0 * np.pi * config.alpha_frequency * time)
        + config.beta_amplitude
        * np.sin(2.0 * np.pi * config.beta_frequency * time)
        + config.gamma_amplitude
        * np.sin(2.0 * np.pi * config.gamma_frequency * time)
    )
    return time, signal.astype(float, copy=False)
