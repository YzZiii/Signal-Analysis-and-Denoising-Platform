"""End-to-end interface used by the GUI and experiment runner."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray
import numpy as np

from .config import DenoiseConfig, NoiseConfig, SignalConfig
from .denoising import denoise_signal
from .metrics import MetricSet, evaluate_signal
from .noise import NoiseResult, add_noise
from .signal_generation import generate_signal

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ExperimentResult:
    time: FloatArray
    clean_signal: FloatArray
    noisy_signal: FloatArray
    denoised_signal: FloatArray
    noise: NoiseResult
    noisy_metrics: MetricSet
    denoised_metrics: MetricSet
    signal_config: SignalConfig
    noise_config: NoiseConfig
    denoise_config: DenoiseConfig


def run_experiment(
    signal_config: SignalConfig,
    noise_config: NoiseConfig,
    denoise_config: DenoiseConfig,
) -> ExperimentResult:
    """Run a complete, deterministic synthetic MEG experiment."""
    time, clean = generate_signal(signal_config)
    noise = add_noise(clean, time, signal_config.sample_rate, noise_config)
    denoised = denoise_signal(
        noise.noisy_signal, signal_config.sample_rate, denoise_config
    )
    return ExperimentResult(
        time=time,
        clean_signal=clean,
        noisy_signal=noise.noisy_signal,
        denoised_signal=denoised,
        noise=noise,
        noisy_metrics=evaluate_signal(clean, noise.noisy_signal),
        denoised_metrics=evaluate_signal(clean, denoised),
        signal_config=signal_config,
        noise_config=noise_config,
        denoise_config=denoise_config,
    )
