"""Validated baseline filters for the MEG denoising comparison."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

from .config import DenoiseConfig

FloatArray = NDArray[np.float64]


def _as_signal(signal: FloatArray) -> FloatArray:
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size < 32:
        raise ValueError("The input signal must be one-dimensional with at least 32 samples.")
    if not np.all(np.isfinite(values)):
        raise ValueError("The input signal contains a non-finite value.")
    return values


def lowpass_filter(
    signal: FloatArray, sample_rate: float, cutoff: float, order: int = 4
) -> FloatArray:
    values = _as_signal(signal)
    sos = butter(order, cutoff, btype="lowpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, values)


def bandpass_filter(
    signal: FloatArray,
    sample_rate: float,
    low_cutoff: float,
    high_cutoff: float,
    order: int = 4,
) -> FloatArray:
    values = _as_signal(signal)
    sos = butter(
        order,
        (low_cutoff, high_cutoff),
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    return sosfiltfilt(sos, values)


def notch_filter(
    signal: FloatArray,
    sample_rate: float,
    line_frequency: float,
    quality_factor: float = 30.0,
) -> FloatArray:
    values = _as_signal(signal)
    numerator, denominator = iirnotch(line_frequency, quality_factor, fs=sample_rate)
    return filtfilt(numerator, denominator, values)


def denoise_signal(
    signal: FloatArray, sample_rate: float, config: DenoiseConfig
) -> FloatArray:
    """Apply the selected baseline method using zero-phase filtering."""
    config.validate(sample_rate)
    if config.method == "lowpass":
        return lowpass_filter(
            signal, sample_rate, config.high_cutoff, config.filter_order
        )
    if config.method == "bandpass":
        return bandpass_filter(
            signal,
            sample_rate,
            config.low_cutoff,
            config.high_cutoff,
            config.filter_order,
        )
    if config.method == "notch":
        return notch_filter(
            signal,
            sample_rate,
            config.line_frequency,
            config.notch_quality_factor,
        )

    notched = notch_filter(
        signal,
        sample_rate,
        config.line_frequency,
        config.notch_quality_factor,
    )
    return bandpass_filter(
        notched,
        sample_rate,
        config.low_cutoff,
        config.high_cutoff,
        config.filter_order,
    )
