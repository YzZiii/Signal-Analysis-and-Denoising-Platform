import numpy as np

from Algorithm.metrics import evaluate_signal


def test_metrics_for_known_error() -> None:
    clean = np.array([1.0, -1.0, 1.0, -1.0])
    candidate = 0.5 * clean
    metrics = evaluate_signal(clean, candidate)
    assert np.isclose(metrics.snr_db, 10 * np.log10(4))
    assert np.isclose(metrics.rmse, 0.5)
    assert np.isclose(metrics.correlation, 1.0)


def test_identical_signal_has_perfect_metrics() -> None:
    clean = np.linspace(-1.0, 1.0, 100)
    metrics = evaluate_signal(clean, clean.copy())
    assert np.isinf(metrics.snr_db)
    assert metrics.rmse == 0
    assert np.isclose(metrics.correlation, 1)
