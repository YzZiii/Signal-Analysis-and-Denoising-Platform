"""Synthetic spatial field projection for the interactive 3D MEG view.

This module does not perform source localisation. It projects the configured
oscillations onto a virtual helmet so the UI can explain spatial field behaviour.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import NoiseConfig, SignalConfig

FloatArray = NDArray[np.float64]


def generate_sensor_positions(sensor_count: int = 96) -> FloatArray:
    """Return evenly distributed unit-sphere positions on a virtual MEG helmet."""
    if sensor_count < 16:
        raise ValueError("At least 16 virtual sensors are required.")
    indices = np.arange(sensor_count, dtype=float)
    z = -0.32 + 1.30 * (indices + 0.5) / sensor_count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = indices * golden_angle
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))


def _lobe(sensor_directions: FloatArray, direction: FloatArray, sharpness: float) -> FloatArray:
    direction = direction / np.linalg.norm(direction)
    mirror = np.array((-direction[0], direction[1], direction[2]), dtype=float)
    mirror /= np.linalg.norm(mirror)
    positive = np.exp(sharpness * (sensor_directions @ direction - 1.0))
    negative = np.exp(sharpness * (sensor_directions @ mirror - 1.0))
    return positive - negative


def simulate_sensor_field(
    time_value: float,
    signal_config: SignalConfig,
    noise_config: NoiseConfig,
    mode: str = "clean",
    sensor_positions: FloatArray | None = None,
) -> FloatArray:
    """Return synthetic virtual-sensor field values in femtotesla.

    Modes are ``clean``, ``noisy``, and ``denoised``. Noise is deterministic for a
    fixed seed and time sample so the time slider remains reproducible.
    """
    signal_config.validate()
    noise_config.validate(signal_config.sample_rate)
    if mode not in {"clean", "noisy", "denoised"}:
        raise ValueError(f"Unknown spatial field mode: {mode}.")
    positions = (
        generate_sensor_positions()
        if sensor_positions is None
        else np.asarray(sensor_positions, dtype=float)
    )
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Sensor positions must have shape (n_sensors, 3).")
    norms = np.linalg.norm(positions, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Sensor positions cannot contain the origin.")
    directions = positions / norms

    source_directions = (
        np.array((-0.60, 0.30, 0.74)),
        np.array((0.48, 0.62, 0.62)),
        np.array((-0.20, -0.68, 0.71)),
    )
    components = (
        (signal_config.alpha_frequency, signal_config.alpha_amplitude, 0.0),
        (signal_config.beta_frequency, signal_config.beta_amplitude, 0.9),
        (signal_config.gamma_frequency, signal_config.gamma_amplitude, 1.7),
    )

    field = np.zeros(positions.shape[0], dtype=float)
    for direction, (frequency, amplitude, phase) in zip(source_directions, components):
        temporal = amplitude * np.sin(2.0 * np.pi * frequency * time_value + phase)
        field += temporal * _lobe(directions, direction, sharpness=7.0)
    clean_field = 115.0 * field

    if mode == "clean":
        return clean_field
    sample_index = int(round(time_value * signal_config.sample_rate))
    rng = np.random.default_rng(noise_config.seed + sample_index)
    random_noise = (
        rng.normal(0.0, 42.0 * noise_config.gaussian_std, positions.shape[0])
        if noise_config.gaussian_enabled
        else np.zeros(positions.shape[0])
    )
    line_phase = np.sin(2.0 * np.pi * noise_config.power_line_frequency * time_value)
    line_pattern = (
        60.0 * noise_config.power_line_amplitude * line_phase * directions[:, 0]
        if noise_config.power_line_enabled
        else np.zeros(positions.shape[0])
    )
    drift_phase = np.sin(2.0 * np.pi * noise_config.drift_frequency * time_value)
    drift_pattern = (
        75.0 * noise_config.drift_amplitude * drift_phase * directions[:, 2]
        if noise_config.drift_enabled
        else np.zeros(positions.shape[0])
    )
    noisy_field = clean_field + random_noise + line_pattern + drift_pattern
    if mode == "noisy":
        return noisy_field
    return clean_field + 0.22 * (noisy_field - clean_field)
