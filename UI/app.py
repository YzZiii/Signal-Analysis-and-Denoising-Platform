"""Tkinter desktop interface for the MEG denoising prototype."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.signal import welch

from Algorithm.config import DenoiseConfig, NoiseConfig, SignalConfig
from Algorithm.denoising import denoise_signal
from Algorithm.metrics import MetricSet, evaluate_signal
from Algorithm.noise import NoiseResult, add_noise
from Algorithm.pipeline import ExperimentResult, run_experiment
from Algorithm.signal_generation import generate_signal


METHOD_LABELS = {
    "Butterworth low-pass": "lowpass",
    "Butterworth band-pass": "bandpass",
    "Power-line notch": "notch",
    "Notch + band-pass": "combined",
}


class MEGPlatformApp:
    """Stateful three-stage interface matching the documented project workflow."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MEG Signal Analysis and Denoising Platform")
        self.root.geometry("1280x800")
        self.root.minsize(1080, 680)

        self.time: np.ndarray | None = None
        self.clean_signal: np.ndarray | None = None
        self.noise_result: NoiseResult | None = None
        self.denoised_signal: np.ndarray | None = None
        self.noisy_metrics: MetricSet | None = None
        self.denoised_metrics: MetricSet | None = None
        self.last_result: ExperimentResult | None = None
        self.active_signal_config: SignalConfig | None = None
        self.active_noise_config: NoiseConfig | None = None

        self._build_variables()
        self._configure_style()
        self._build_layout()
        self._draw_signals()

    def _build_variables(self) -> None:
        self.sample_rate = tk.StringVar(value="1000")
        self.duration = tk.StringVar(value="5")
        self.alpha_frequency = tk.StringVar(value="10")
        self.alpha_amplitude = tk.StringVar(value="1.0")
        self.beta_frequency = tk.StringVar(value="20")
        self.beta_amplitude = tk.StringVar(value="0.5")
        self.gamma_frequency = tk.StringVar(value="40")
        self.gamma_amplitude = tk.StringVar(value="0.3")

        self.gaussian_enabled = tk.BooleanVar(value=True)
        self.gaussian_std = tk.StringVar(value="0.4")
        self.power_line_enabled = tk.BooleanVar(value=True)
        self.power_line_frequency = tk.StringVar(value="50")
        self.power_line_amplitude = tk.StringVar(value="0.15")
        self.drift_enabled = tk.BooleanVar(value=True)
        self.drift_frequency = tk.StringVar(value="0.3")
        self.drift_amplitude = tk.StringVar(value="0.1")
        self.seed = tk.StringVar(value="42")

        self.method_label = tk.StringVar(value="Notch + band-pass")
        self.low_cutoff = tk.StringVar(value="1")
        self.high_cutoff = tk.StringVar(value="45")
        self.filter_order = tk.StringVar(value="4")
        self.notch_quality_factor = tk.StringVar(value="30")
        self.view_seconds = tk.StringVar(value="0.5")

        self.snr_text = tk.StringVar(value="SNR: --")
        self.rmse_text = tk.StringVar(value="RMSE: --")
        self.correlation_text = tk.StringVar(value="Correlation: --")
        self.status_text = tk.StringVar(value="Ready. Generate a signal to begin.")

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        for candidate in ("vista", "clam"):
            if candidate in style.theme_names():
                style.theme_use(candidate)
                break
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Stage.TButton", font=("Segoe UI", 10, "bold"), padding=7)
        style.configure("Metric.TLabel", font=("Consolas", 10))
        style.configure("Status.TLabel", padding=(8, 5))

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            outer,
            text="MEG Signal Analysis and Denoising Platform",
            style="Title.TLabel",
        ).pack(anchor=tk.W, pady=(0, 8))

        body = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)
        controls = ttk.Frame(body, padding=(0, 0, 10, 0), width=330)
        visualisation = ttk.Frame(body)
        body.add(controls, weight=0)
        body.add(visualisation, weight=1)

        notebook = ttk.Notebook(controls)
        notebook.pack(fill=tk.BOTH, expand=True)
        signal_tab = ttk.Frame(notebook, padding=10)
        noise_tab = ttk.Frame(notebook, padding=10)
        filter_tab = ttk.Frame(notebook, padding=10)
        notebook.add(signal_tab, text="Signal")
        notebook.add(noise_tab, text="Noise")
        notebook.add(filter_tab, text="Denoising")
        self._build_signal_controls(signal_tab)
        self._build_noise_controls(noise_tab)
        self._build_filter_controls(filter_tab)

        workflow = ttk.LabelFrame(controls, text="Workflow", padding=8)
        workflow.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(
            workflow,
            text="1. Generate Signal",
            command=self.generate_stage,
            style="Stage.TButton",
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            workflow,
            text="2. Add Noise",
            command=self.noise_stage,
            style="Stage.TButton",
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            workflow,
            text="3. Denoise Signal",
            command=self.denoise_stage,
            style="Stage.TButton",
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            workflow,
            text="Run Full Pipeline",
            command=self.run_full_pipeline,
        ).pack(fill=tk.X, pady=(8, 2))
        self.export_button = ttk.Button(
            workflow,
            text="Export Result",
            command=self.export_result,
            state=tk.DISABLED,
        )
        self.export_button.pack(fill=tk.X, pady=2)

        metrics = ttk.LabelFrame(visualisation, text="Evaluation metrics", padding=8)
        metrics.pack(fill=tk.X)
        for column, variable in enumerate(
            (self.snr_text, self.rmse_text, self.correlation_text)
        ):
            metrics.columnconfigure(column, weight=1)
            ttk.Label(metrics, textvariable=variable, style="Metric.TLabel").grid(
                row=0, column=column, sticky=tk.W, padx=8
            )

        figure_frame = ttk.LabelFrame(
            visualisation, text="Signal visualisation", padding=4
        )
        figure_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.figure = Figure(figsize=(8.5, 6), dpi=100, constrained_layout=True)
        self.time_axis = self.figure.add_subplot(211)
        self.spectrum_axis = self.figure.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.figure, master=figure_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, figure_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=tk.X)

        ttk.Label(
            outer, textvariable=self.status_text, style="Status.TLabel", anchor=tk.W
        ).pack(fill=tk.X, pady=(6, 0))

    def _entry(self, parent: ttk.Frame, row: int, label: str, variable: tk.Variable) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
        ttk.Entry(parent, textvariable=variable, width=12).grid(
            row=row, column=1, sticky=tk.EW, pady=3
        )
        parent.columnconfigure(1, weight=1)

    def _build_signal_controls(self, parent: ttk.Frame) -> None:
        fields = (
            ("Sampling rate (Hz)", self.sample_rate),
            ("Duration (s)", self.duration),
            ("Alpha frequency (Hz)", self.alpha_frequency),
            ("Alpha amplitude", self.alpha_amplitude),
            ("Beta frequency (Hz)", self.beta_frequency),
            ("Beta amplitude", self.beta_amplitude),
            ("Gamma frequency (Hz)", self.gamma_frequency),
            ("Gamma amplitude", self.gamma_amplitude),
            ("Plot window (s)", self.view_seconds),
        )
        for row, (label, variable) in enumerate(fields):
            self._entry(parent, row, label, variable)

    def _build_noise_controls(self, parent: ttk.Frame) -> None:
        ttk.Checkbutton(
            parent, text="Gaussian sensor noise", variable=self.gaussian_enabled
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        self._entry(parent, 1, "Standard deviation", self.gaussian_std)
        ttk.Separator(parent).grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=7)
        ttk.Checkbutton(
            parent, text="Power-line interference", variable=self.power_line_enabled
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W)
        self._entry(parent, 4, "Frequency (Hz)", self.power_line_frequency)
        self._entry(parent, 5, "Amplitude", self.power_line_amplitude)
        ttk.Separator(parent).grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=7)
        ttk.Checkbutton(
            parent, text="Low-frequency drift", variable=self.drift_enabled
        ).grid(row=7, column=0, columnspan=2, sticky=tk.W)
        self._entry(parent, 8, "Frequency (Hz)", self.drift_frequency)
        self._entry(parent, 9, "Amplitude", self.drift_amplitude)
        ttk.Separator(parent).grid(row=10, column=0, columnspan=2, sticky=tk.EW, pady=7)
        self._entry(parent, 11, "Random seed", self.seed)

    def _build_filter_controls(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Method").grid(row=0, column=0, sticky=tk.W, pady=3)
        ttk.Combobox(
            parent,
            textvariable=self.method_label,
            values=tuple(METHOD_LABELS),
            state="readonly",
            width=22,
        ).grid(row=0, column=1, sticky=tk.EW, pady=3)
        fields = (
            ("Low cutoff (Hz)", self.low_cutoff),
            ("High cutoff (Hz)", self.high_cutoff),
            ("Filter order", self.filter_order),
            ("Notch Q factor", self.notch_quality_factor),
        )
        for row, (label, variable) in enumerate(fields, start=1):
            self._entry(parent, row, label, variable)

    @staticmethod
    def _number(variable: tk.StringVar, label: str, integer: bool = False) -> float | int:
        try:
            return int(variable.get()) if integer else float(variable.get())
        except ValueError as exc:
            raise ValueError(f"{label} must be a valid number.") from exc

    def _signal_config(self) -> SignalConfig:
        return SignalConfig(
            sample_rate=float(self._number(self.sample_rate, "Sampling rate")),
            duration=float(self._number(self.duration, "Duration")),
            alpha_frequency=float(self._number(self.alpha_frequency, "Alpha frequency")),
            alpha_amplitude=float(self._number(self.alpha_amplitude, "Alpha amplitude")),
            beta_frequency=float(self._number(self.beta_frequency, "Beta frequency")),
            beta_amplitude=float(self._number(self.beta_amplitude, "Beta amplitude")),
            gamma_frequency=float(self._number(self.gamma_frequency, "Gamma frequency")),
            gamma_amplitude=float(self._number(self.gamma_amplitude, "Gamma amplitude")),
        )

    def _noise_config(self) -> NoiseConfig:
        return NoiseConfig(
            gaussian_enabled=self.gaussian_enabled.get(),
            gaussian_std=float(self._number(self.gaussian_std, "Gaussian deviation")),
            power_line_enabled=self.power_line_enabled.get(),
            power_line_frequency=float(
                self._number(self.power_line_frequency, "Power-line frequency")
            ),
            power_line_amplitude=float(
                self._number(self.power_line_amplitude, "Power-line amplitude")
            ),
            drift_enabled=self.drift_enabled.get(),
            drift_frequency=float(self._number(self.drift_frequency, "Drift frequency")),
            drift_amplitude=float(self._number(self.drift_amplitude, "Drift amplitude")),
            seed=int(self._number(self.seed, "Random seed", integer=True)),
        )

    def _denoise_config(self) -> DenoiseConfig:
        return DenoiseConfig(
            method=METHOD_LABELS[self.method_label.get()],
            low_cutoff=float(self._number(self.low_cutoff, "Low cutoff")),
            high_cutoff=float(self._number(self.high_cutoff, "High cutoff")),
            line_frequency=float(
                self._number(self.power_line_frequency, "Notch frequency")
            ),
            filter_order=int(self._number(self.filter_order, "Filter order", integer=True)),
            notch_quality_factor=float(
                self._number(self.notch_quality_factor, "Notch quality factor")
            ),
        )

    def _run_action(self, action) -> None:
        try:
            action()
        except Exception as exc:
            self.status_text.set(f"Error: {exc}")
            messagebox.showerror("MEG platform", str(exc), parent=self.root)

    def generate_stage(self) -> None:
        self._run_action(self._generate_stage)

    def _generate_stage(self) -> None:
        config = self._signal_config()
        self.time, self.clean_signal = generate_signal(config)
        self.active_signal_config = config
        self.active_noise_config = None
        self.noise_result = None
        self.denoised_signal = None
        self.noisy_metrics = None
        self.denoised_metrics = None
        self.last_result = None
        self.export_button.configure(state=tk.DISABLED)
        self._update_metric_text()
        self._draw_signals()
        self.status_text.set(
            f"Generated {self.clean_signal.size:,} samples at {config.sample_rate:g} Hz."
        )

    def noise_stage(self) -> None:
        self._run_action(self._noise_stage)

    def _noise_stage(self) -> None:
        if self.clean_signal is None or self.time is None:
            raise ValueError("Generate a clean signal before adding noise.")
        signal_config = self._signal_config()
        if signal_config != self.active_signal_config:
            raise ValueError("Signal settings changed. Generate the clean signal again.")
        noise_config = self._noise_config()
        self.noise_result = add_noise(
            self.clean_signal, self.time, signal_config.sample_rate, noise_config
        )
        self.noisy_metrics = evaluate_signal(
            self.clean_signal, self.noise_result.noisy_signal
        )
        self.active_noise_config = noise_config
        self.denoised_signal = None
        self.denoised_metrics = None
        self.last_result = None
        self.export_button.configure(state=tk.DISABLED)
        self._update_metric_text()
        self._draw_signals()
        names = ", ".join(self.noise_result.components) or "none"
        self.status_text.set(f"Added noise components: {names}. Seed: {noise_config.seed}.")

    def denoise_stage(self) -> None:
        self._run_action(self._denoise_stage)

    def _denoise_stage(self) -> None:
        if self.clean_signal is None or self.time is None or self.noise_result is None:
            raise ValueError("Generate a signal and add noise before denoising.")
        signal_config = self._signal_config()
        noise_config = self._noise_config()
        if signal_config != self.active_signal_config:
            raise ValueError("Signal settings changed. Generate the clean signal again.")
        if noise_config != self.active_noise_config:
            raise ValueError("Noise settings changed. Add noise again before denoising.")
        denoise_config = self._denoise_config()
        self.denoised_signal = denoise_signal(
            self.noise_result.noisy_signal,
            signal_config.sample_rate,
            denoise_config,
        )
        self.denoised_metrics = evaluate_signal(
            self.clean_signal, self.denoised_signal
        )
        self.last_result = ExperimentResult(
            time=self.time,
            clean_signal=self.clean_signal,
            noisy_signal=self.noise_result.noisy_signal,
            denoised_signal=self.denoised_signal,
            noise=self.noise_result,
            noisy_metrics=self.noisy_metrics
            or evaluate_signal(self.clean_signal, self.noise_result.noisy_signal),
            denoised_metrics=self.denoised_metrics,
            signal_config=signal_config,
            noise_config=noise_config,
            denoise_config=denoise_config,
        )
        self.export_button.configure(state=tk.NORMAL)
        self._update_metric_text()
        self._draw_signals()
        change = self.denoised_metrics.snr_db - self.last_result.noisy_metrics.snr_db
        self.status_text.set(
            f"Denoising complete with {self.method_label.get()}. SNR change: {change:+.2f} dB."
        )

    def run_full_pipeline(self) -> None:
        self._run_action(self._run_full_pipeline)

    def _run_full_pipeline(self) -> None:
        result = run_experiment(
            self._signal_config(), self._noise_config(), self._denoise_config()
        )
        self.last_result = result
        self.active_signal_config = result.signal_config
        self.active_noise_config = result.noise_config
        self.time = result.time
        self.clean_signal = result.clean_signal
        self.noise_result = result.noise
        self.denoised_signal = result.denoised_signal
        self.noisy_metrics = result.noisy_metrics
        self.denoised_metrics = result.denoised_metrics
        self.export_button.configure(state=tk.NORMAL)
        self._update_metric_text()
        self._draw_signals()
        change = result.denoised_metrics.snr_db - result.noisy_metrics.snr_db
        self.status_text.set(
            f"Full pipeline complete. SNR change: {change:+.2f} dB; seed: {result.noise_config.seed}."
        )

    def _update_metric_text(self) -> None:
        if self.noisy_metrics is None:
            self.snr_text.set("SNR: --")
            self.rmse_text.set("RMSE: --")
            self.correlation_text.set("Correlation: --")
            return
        before = self.noisy_metrics
        after = self.denoised_metrics
        self.snr_text.set(
            f"SNR: {before.snr_db:.2f} dB → "
            + (f"{after.snr_db:.2f} dB" if after else "--")
        )
        self.rmse_text.set(
            f"RMSE: {before.rmse:.4f} → " + (f"{after.rmse:.4f}" if after else "--")
        )
        self.correlation_text.set(
            f"Correlation: {before.correlation:.4f} → "
            + (f"{after.correlation:.4f}" if after else "--")
        )

    def _draw_signals(self) -> None:
        self.time_axis.clear()
        self.spectrum_axis.clear()
        if self.clean_signal is None or self.time is None:
            self.time_axis.text(
                0.5,
                0.5,
                "Generate a signal to begin",
                ha="center",
                va="center",
                transform=self.time_axis.transAxes,
            )
            self.time_axis.set_axis_off()
            self.spectrum_axis.set_axis_off()
            self.canvas.draw_idle()
            return

        self.time_axis.set_axis_on()
        self.spectrum_axis.set_axis_on()
        sample_rate = float(self.sample_rate.get())
        view_seconds = float(self._number(self.view_seconds, "Plot window"))
        if view_seconds <= 0:
            raise ValueError("Plot window must be positive.")
        visible = self.time <= min(view_seconds, self.time[-1])
        series: list[tuple[str, np.ndarray, str, float]] = [
            ("Clean", self.clean_signal, "#1f77b4", 1.6)
        ]
        if self.noise_result is not None:
            series.append(("Noisy", self.noise_result.noisy_signal, "#d62728", 0.9))
        if self.denoised_signal is not None:
            series.append(("Denoised", self.denoised_signal, "#2ca02c", 1.3))

        for label, values, colour, width in series:
            self.time_axis.plot(
                self.time[visible], values[visible], label=label, color=colour, linewidth=width
            )
            segment = min(2048, values.size)
            frequencies, power = welch(values, fs=sample_rate, nperseg=segment)
            keep = frequencies <= min(100.0, sample_rate / 2.0)
            self.spectrum_axis.semilogy(
                frequencies[keep],
                np.maximum(power[keep], np.finfo(float).tiny),
                label=label,
                color=colour,
                linewidth=width,
            )

        self.time_axis.set_title("Time-domain comparison")
        self.time_axis.set_xlabel("Time (s)")
        self.time_axis.set_ylabel("Normalised amplitude")
        self.time_axis.grid(True, alpha=0.25)
        self.time_axis.legend(loc="upper right")
        self.spectrum_axis.set_title("Power spectral density")
        self.spectrum_axis.set_xlabel("Frequency (Hz)")
        self.spectrum_axis.set_ylabel("PSD")
        self.spectrum_axis.grid(True, alpha=0.25)
        self.spectrum_axis.legend(loc="upper right")
        self.canvas.draw_idle()

    def export_result(self) -> None:
        if self.last_result is None:
            messagebox.showinfo(
                "MEG platform", "Run denoising before exporting.", parent=self.root
            )
            return
        output_dir = Path(__file__).resolve().parents[1] / "Results"
        output_dir.mkdir(parents=True, exist_ok=True)
        path_text = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export MEG experiment",
            initialdir=output_dir,
            initialfile=f"meg_experiment_seed_{self.last_result.noise_config.seed}.csv",
            defaultextension=".csv",
            filetypes=(("CSV file", "*.csv"),),
        )
        if not path_text:
            return
        csv_path = Path(path_text)
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("time_s", "clean", "noisy", "denoised"))
            writer.writerows(
                zip(
                    self.last_result.time,
                    self.last_result.clean_signal,
                    self.last_result.noisy_signal,
                    self.last_result.denoised_signal,
                )
            )

        summary = {
            "signal_config": asdict(self.last_result.signal_config),
            "noise_config": asdict(self.last_result.noise_config),
            "denoise_config": asdict(self.last_result.denoise_config),
            "metrics_before": self.last_result.noisy_metrics.as_dict(),
            "metrics_after": self.last_result.denoised_metrics.as_dict(),
        }
        json_path = csv_path.with_suffix(".json")
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.status_text.set(f"Exported {csv_path.name} and {json_path.name}.")


def launch() -> None:
    root = tk.Tk()
    MEGPlatformApp(root)
    root.mainloop()
