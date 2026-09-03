"""Signal-processing backend for the MEG denoising prototype."""

from .config import DenoiseConfig, NoiseConfig, SignalConfig
from .pipeline import ExperimentResult, run_experiment

__all__ = [
    "DenoiseConfig",
    "NoiseConfig",
    "SignalConfig",
    "ExperimentResult",
    "run_experiment",
]
