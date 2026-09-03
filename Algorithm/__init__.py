"""Signal-processing backend for the MEG denoising prototype."""

from .config import DenoiseConfig, NoiseConfig, SignalConfig
from .pipeline import ExperimentResult, run_experiment
from .spatial import generate_sensor_positions, simulate_sensor_field

__all__ = [
    "DenoiseConfig",
    "NoiseConfig",
    "SignalConfig",
    "ExperimentResult",
    "run_experiment",
    "generate_sensor_positions",
    "simulate_sensor_field",
]
