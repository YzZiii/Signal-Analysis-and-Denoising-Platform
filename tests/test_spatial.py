import numpy as np

from Algorithm.config import NoiseConfig, SignalConfig
from Algorithm.spatial import generate_sensor_positions, simulate_sensor_field


def test_sensor_positions_form_a_virtual_helmet() -> None:
    positions = generate_sensor_positions(96)
    assert positions.shape == (96, 3)
    assert np.allclose(np.linalg.norm(positions, axis=1), 1.0)
    assert np.min(positions[:, 2]) > -0.33


def test_spatial_modes_are_finite_and_repeatable() -> None:
    signal_config = SignalConfig()
    noise_config = NoiseConfig(seed=42)
    positions = generate_sensor_positions(64)
    clean = simulate_sensor_field(0.123, signal_config, noise_config, "clean", positions)
    noisy_first = simulate_sensor_field(
        0.123, signal_config, noise_config, "noisy", positions
    )
    noisy_second = simulate_sensor_field(
        0.123, signal_config, noise_config, "noisy", positions
    )
    denoised = simulate_sensor_field(
        0.123, signal_config, noise_config, "denoised", positions
    )
    assert np.all(np.isfinite(clean))
    assert np.array_equal(noisy_first, noisy_second)
    assert not np.array_equal(clean, noisy_first)
    assert np.linalg.norm(denoised - clean) < np.linalg.norm(noisy_first - clean)


def test_disabled_noise_components_do_not_change_spatial_field() -> None:
    signal_config = SignalConfig()
    noise_config = NoiseConfig(
        gaussian_enabled=False,
        power_line_enabled=False,
        drift_enabled=False,
    )
    clean = simulate_sensor_field(0.211, signal_config, noise_config, "clean")
    noisy = simulate_sensor_field(0.211, signal_config, noise_config, "noisy")
    assert np.array_equal(clean, noisy)
