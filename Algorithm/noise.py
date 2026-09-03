"""Controlled noise models for synthetic MEG experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import NoiseConfig

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class NoiseResult:
    noisy_signal: FloatArray
    combined_noise: FloatArray
    components: dict[str, FloatArray]


def add_noise(
    clean_signal: FloatArray,
    time: FloatArray,
    sample_rate: float,
    config: NoiseConfig,
) -> NoiseResult:
    """Add enabled noise sources and return each component for inspection."""
    config.validate(sample_rate)
    clean = np.asarray(clean_signal, dtype=float)
    time_axis = np.asarray(time, dtype=float)
    if clean.ndim != 1 or time_axis.ndim != 1 or clean.shape != time_axis.shape:
        raise ValueError("Clean signal and time must be one-dimensional and equal length.")

    components: dict[str, FloatArray] = {}
    if config.gaussian_enabled:
        rng = np.random.default_rng(config.seed)
        components["gaussian"] = rng.normal(0.0, config.gaussian_std, clean.size)
    if config.power_line_enabled:
        components["power_line"] = config.power_line_amplitude * np.sin(
            2.0 * np.pi * config.power_line_frequency * time_axis
        )
    if config.drift_enabled:
        components["drift"] = config.drift_amplitude * np.sin(
            2.0 * np.pi * config.drift_frequency * time_axis
        )

    if components:
        combined = np.sum(np.stack(tuple(components.values())), axis=0)
    else:
        combined = np.zeros_like(clean)
    return NoiseResult(clean + combined, combined, components)
