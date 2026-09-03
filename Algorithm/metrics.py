"""Objective full-reference signal-quality metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MetricSet:
    snr_db: float
    rmse: float
    correlation: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate_signal(clean_signal: FloatArray, candidate_signal: FloatArray) -> MetricSet:
    """Compare a candidate signal with the known clean reference."""
    clean = np.asarray(clean_signal, dtype=float)
    candidate = np.asarray(candidate_signal, dtype=float)
    if clean.ndim != 1 or candidate.ndim != 1 or clean.shape != candidate.shape:
        raise ValueError("Metric inputs must be one-dimensional and equal length.")
    if clean.size < 2 or not np.all(np.isfinite(clean)) or not np.all(np.isfinite(candidate)):
        raise ValueError("Metric inputs must contain at least two finite samples.")

    error = clean - candidate
    signal_power = float(np.mean(np.square(clean)))
    error_power = float(np.mean(np.square(error)))
    if error_power == 0.0:
        snr_db = float("inf")
    elif signal_power == 0.0:
        snr_db = float("-inf")
    else:
        snr_db = float(10.0 * np.log10(signal_power / error_power))
    rmse = float(np.sqrt(error_power))

    clean_std = float(np.std(clean))
    candidate_std = float(np.std(candidate))
    if clean_std == 0.0 or candidate_std == 0.0:
        correlation = 1.0 if np.array_equal(clean, candidate) else 0.0
    else:
        correlation = float(np.corrcoef(clean, candidate)[0, 1])
    return MetricSet(snr_db, rmse, correlation)
