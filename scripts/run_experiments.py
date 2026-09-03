"""Run the fixed-seed comparison used as quantitative project evidence."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, stdev
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Algorithm.config import DenoiseConfig, NoiseConfig, SignalConfig
from Algorithm.pipeline import run_experiment


def main() -> None:
    output_dir = PROJECT_ROOT / "Results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "experiment_results.csv"
    summary_path = output_dir / "experiment_summary.csv"
    signal_config = SignalConfig()
    methods = ("lowpass", "bandpass", "notch", "combined")
    noise_levels = (0.2, 0.4, 0.6)
    seeds = (11, 23, 37, 41, 53)
    rows: list[dict[str, object]] = []

    for noise_level in noise_levels:
        for seed in seeds:
            noise_config = NoiseConfig(gaussian_std=noise_level, seed=seed)
            for method in methods:
                denoise_config = DenoiseConfig(method=method)
                result = run_experiment(signal_config, noise_config, denoise_config)
                rows.append(
                    {
                        "method": method,
                        "gaussian_std": noise_level,
                        "seed": seed,
                        "sample_rate_hz": signal_config.sample_rate,
                        "duration_s": signal_config.duration,
                        "low_cutoff_hz": denoise_config.low_cutoff,
                        "high_cutoff_hz": denoise_config.high_cutoff,
                        "line_frequency_hz": denoise_config.line_frequency,
                        "filter_order": denoise_config.filter_order,
                        "snr_before_db": result.noisy_metrics.snr_db,
                        "snr_after_db": result.denoised_metrics.snr_db,
                        "snr_change_db": (
                            result.denoised_metrics.snr_db - result.noisy_metrics.snr_db
                        ),
                        "rmse_before": result.noisy_metrics.rmse,
                        "rmse_after": result.denoised_metrics.rmse,
                        "correlation_before": result.noisy_metrics.correlation,
                        "correlation_after": result.denoised_metrics.correlation,
                    }
                )

    with results_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, object]] = []
    for noise_level in noise_levels:
        for method in methods:
            group = [
                row
                for row in rows
                if row["method"] == method and row["gaussian_std"] == noise_level
            ]

            def values(name: str) -> list[float]:
                return [float(row[name]) for row in group]

            summary_rows.append(
                {
                    "method": method,
                    "gaussian_std": noise_level,
                    "repeats": len(group),
                    "mean_snr_before_db": mean(values("snr_before_db")),
                    "mean_snr_after_db": mean(values("snr_after_db")),
                    "sd_snr_after_db": stdev(values("snr_after_db")),
                    "mean_snr_change_db": mean(values("snr_change_db")),
                    "mean_rmse_before": mean(values("rmse_before")),
                    "mean_rmse_after": mean(values("rmse_after")),
                    "sd_rmse_after": stdev(values("rmse_after")),
                    "mean_correlation_before": mean(values("correlation_before")),
                    "mean_correlation_after": mean(values("correlation_after")),
                    "sd_correlation_after": stdev(values("correlation_after")),
                }
            )

    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {len(rows)} experiment rows to {results_path}")
    print(f"Wrote {len(summary_rows)} aggregate rows to {summary_path}")


if __name__ == "__main__":
    main()
