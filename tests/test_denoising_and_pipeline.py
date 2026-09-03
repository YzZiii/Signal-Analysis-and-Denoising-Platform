import numpy as np

from Algorithm.config import DenoiseConfig, NoiseConfig, SignalConfig
from Algorithm.denoising import notch_filter
from Algorithm.pipeline import run_experiment


def _spectral_amplitude(signal: np.ndarray, sample_rate: float, frequency: float) -> float:
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    spectrum = np.abs(np.fft.rfft(signal))
    return float(spectrum[np.argmin(np.abs(frequencies - frequency))])


def test_notch_filter_attenuates_line_frequency() -> None:
    sample_rate = 1000.0
    time = np.arange(5000) / sample_rate
    signal = np.sin(2 * np.pi * 10 * time) + np.sin(2 * np.pi * 50 * time)
    filtered = notch_filter(signal, sample_rate, 50, quality_factor=30)
    assert _spectral_amplitude(filtered, sample_rate, 50) < (
        0.1 * _spectral_amplitude(signal, sample_rate, 50)
    )
    assert _spectral_amplitude(filtered, sample_rate, 10) > (
        0.9 * _spectral_amplitude(signal, sample_rate, 10)
    )


def test_default_pipeline_is_repeatable_and_improves_metrics() -> None:
    arguments = (SignalConfig(), NoiseConfig(), DenoiseConfig())
    first = run_experiment(*arguments)
    second = run_experiment(*arguments)
    assert np.array_equal(first.denoised_signal, second.denoised_signal)
    assert first.denoised_signal.shape == first.clean_signal.shape
    assert first.denoised_metrics.snr_db > first.noisy_metrics.snr_db
    assert first.denoised_metrics.rmse < first.noisy_metrics.rmse
    assert first.denoised_metrics.correlation > first.noisy_metrics.correlation


def test_each_documented_method_runs() -> None:
    for method in ("lowpass", "bandpass", "notch", "combined"):
        result = run_experiment(
            SignalConfig(duration=1),
            NoiseConfig(seed=7),
            DenoiseConfig(method=method),
        )
        assert np.all(np.isfinite(result.denoised_signal))
