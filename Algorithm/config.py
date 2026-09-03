"""Validated configuration objects shared by the backend and GUI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalConfig:
    sample_rate: float = 1000.0
    duration: float = 5.0
    alpha_frequency: float = 10.0
    alpha_amplitude: float = 1.0
    beta_frequency: float = 20.0
    beta_amplitude: float = 0.5
    gamma_frequency: float = 40.0
    gamma_amplitude: float = 0.3

    def validate(self) -> None:
        if self.sample_rate <= 0 or self.duration <= 0:
            raise ValueError("Sampling rate and duration must be positive.")
        nyquist = self.sample_rate / 2.0
        frequencies = (
            self.alpha_frequency,
            self.beta_frequency,
            self.gamma_frequency,
        )
        if any(frequency <= 0 or frequency >= nyquist for frequency in frequencies):
            raise ValueError("Every signal frequency must be between 0 and Nyquist.")
        amplitudes = (
            self.alpha_amplitude,
            self.beta_amplitude,
            self.gamma_amplitude,
        )
        if any(amplitude < 0 for amplitude in amplitudes):
            raise ValueError("Signal amplitudes cannot be negative.")
        if round(self.sample_rate * self.duration) < 32:
            raise ValueError("The configuration must produce at least 32 samples.")


@dataclass(frozen=True)
class NoiseConfig:
    gaussian_enabled: bool = True
    gaussian_std: float = 0.4
    power_line_enabled: bool = True
    power_line_frequency: float = 50.0
    power_line_amplitude: float = 0.15
    drift_enabled: bool = True
    drift_frequency: float = 0.3
    drift_amplitude: float = 0.1
    seed: int = 42

    def validate(self, sample_rate: float) -> None:
        if self.gaussian_std < 0:
            raise ValueError("Gaussian standard deviation cannot be negative.")
        if self.power_line_amplitude < 0 or self.drift_amplitude < 0:
            raise ValueError("Noise amplitudes cannot be negative.")
        nyquist = sample_rate / 2.0
        if self.power_line_enabled and not 0 < self.power_line_frequency < nyquist:
            raise ValueError("Power-line frequency must be between 0 and Nyquist.")
        if self.drift_enabled and not 0 < self.drift_frequency < nyquist:
            raise ValueError("Drift frequency must be between 0 and Nyquist.")


@dataclass(frozen=True)
class DenoiseConfig:
    method: str = "combined"
    low_cutoff: float = 1.0
    high_cutoff: float = 45.0
    line_frequency: float = 50.0
    filter_order: int = 4
    notch_quality_factor: float = 30.0

    def validate(self, sample_rate: float) -> None:
        methods = {"lowpass", "bandpass", "notch", "combined"}
        if self.method not in methods:
            raise ValueError(f"Unknown denoising method: {self.method}.")
        nyquist = sample_rate / 2.0
        if not 0 < self.high_cutoff < nyquist:
            raise ValueError("High cutoff must be between 0 and Nyquist.")
        if not 0 < self.low_cutoff < self.high_cutoff:
            raise ValueError("Low cutoff must be positive and below the high cutoff.")
        if not 0 < self.line_frequency < nyquist:
            raise ValueError("Notch frequency must be between 0 and Nyquist.")
        if not 1 <= self.filter_order <= 12:
            raise ValueError("Filter order must be between 1 and 12.")
        if self.notch_quality_factor <= 0:
            raise ValueError("Notch quality factor must be positive.")
