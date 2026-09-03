import numpy as np

from Algorithm.config import NoiseConfig, SignalConfig
from Algorithm.noise import add_noise
from Algorithm.signal_generation import generate_signal


def test_signal_shape_and_components() -> None:
    config = SignalConfig(sample_rate=500, duration=2)
    time, clean = generate_signal(config)
    assert time.shape == clean.shape == (1000,)
    assert time[0] == 0
    assert np.isclose(time[-1], 1.998)
    assert np.isclose(clean[0], 0)


def test_noise_is_repeatable_for_fixed_seed() -> None:
    signal_config = SignalConfig(sample_rate=500, duration=1)
    time, clean = generate_signal(signal_config)
    noise_config = NoiseConfig(seed=123)
    first = add_noise(clean, time, signal_config.sample_rate, noise_config)
    second = add_noise(clean, time, signal_config.sample_rate, noise_config)
    assert np.array_equal(first.noisy_signal, second.noisy_signal)
    assert set(first.components) == {"gaussian", "power_line", "drift"}


def test_all_noise_sources_can_be_disabled() -> None:
    signal_config = SignalConfig(sample_rate=500, duration=1)
    time, clean = generate_signal(signal_config)
    result = add_noise(
        clean,
        time,
        signal_config.sample_rate,
        NoiseConfig(
            gaussian_enabled=False,
            power_line_enabled=False,
            drift_enabled=False,
        ),
    )
    assert result.components == {}
    assert np.array_equal(result.noisy_signal, clean)
