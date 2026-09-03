"""Modern Tkinter desktop interface for the MEG denoising prototype."""

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
from UI.brain_view import BrainFieldView


COLORS = {
    "background": "#06131F",
    "rail": "#081A28",
    "card": "#0C2030",
    "inset": "#102838",
    "border": "#244256",
    "text": "#E6F1F7",
    "muted": "#8CA6B5",
    "teal": "#2DD4BF",
    "cyan": "#38BDF8",
    "green": "#4ADE80",
    "coral": "#FB7185",
    "amber": "#FBBF24",
}

METHOD_LABELS = {
    "Butterworth low-pass": "lowpass",
    "Butterworth band-pass": "bandpass",
    "Power-line notch": "notch",
    "Notch + band-pass": "combined",
}


class MEGPlatformApp:
    """Stateful desktop interface for a reproducible synthetic MEG workflow."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MEG Signal Analysis Platform")
        self.root.geometry("1440x900")
        self.root.minsize(1180, 720)
        self.root.configure(background=COLORS["background"])

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
        self.show_page("analysis")

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

        self.snr_text = tk.StringVar(value="--")
        self.snr_change_text = tk.StringVar(value="Waiting for data")
        self.rmse_text = tk.StringVar(value="--")
        self.rmse_change_text = tk.StringVar(value="No comparison")
        self.correlation_text = tk.StringVar(value="--")
        self.correlation_change_text = tk.StringVar(value="No comparison")
        self.status_text = tk.StringVar(value="Ready — generate a signal to begin")
        self.sample_rate_chip = tk.StringVar(value="1000 Hz")
        self.duration_chip = tk.StringVar(value="5.0 s")

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        self.root.option_add("*Font", "{Segoe UI} 10")

        style.configure("App.TFrame", background=COLORS["background"])
        style.configure("Rail.TFrame", background=COLORS["rail"])
        style.configure("Card.TFrame", background=COLORS["card"])
        style.configure("Inset.TFrame", background=COLORS["inset"])
        style.configure("TSeparator", background=COLORS["border"])

        style.configure(
            "Header.TLabel",
            background=COLORS["background"],
            foreground=COLORS["text"],
            font=("Segoe UI", 18, "bold"),
        )
        style.configure(
            "PageTitle.TLabel",
            background=COLORS["background"],
            foreground=COLORS["text"],
            font=("Segoe UI", 16, "bold"),
        )
        style.configure(
            "RailBrand.TLabel",
            background=COLORS["rail"],
            foreground=COLORS["teal"],
            font=("Segoe UI", 14, "bold"),
        )
        style.configure(
            "Body.TLabel", background=COLORS["card"], foreground=COLORS["text"]
        )
        style.configure(
            "Muted.TLabel", background=COLORS["card"], foreground=COLORS["muted"]
        )
        style.configure(
            "AppMuted.TLabel",
            background=COLORS["background"],
            foreground=COLORS["muted"],
        )
        style.configure(
            "RailMuted.TLabel",
            background=COLORS["rail"],
            foreground=COLORS["muted"],
        )
        style.configure(
            "CardTitle.TLabel",
            background=COLORS["card"],
            foreground=COLORS["text"],
            font=("Segoe UI", 12, "bold"),
        )
        style.configure(
            "MetricTitle.TLabel",
            background=COLORS["card"],
            foreground=COLORS["muted"],
            font=("Segoe UI", 9, "bold"),
        )
        style.configure(
            "MetricValue.TLabel",
            background=COLORS["inset"],
            foreground=COLORS["text"],
            font=("Segoe UI", 15, "bold"),
        )
        style.configure(
            "MetricSmall.TLabel",
            background=COLORS["inset"],
            foreground=COLORS["muted"],
            font=("Segoe UI", 9),
        )
        style.configure(
            "Chip.TLabel",
            background=COLORS["inset"],
            foreground=COLORS["cyan"],
            padding=(10, 5),
        )
        style.configure(
            "WarningChip.TLabel",
            background=COLORS["inset"],
            foreground=COLORS["amber"],
            font=("Segoe UI", 8, "bold"),
            padding=(9, 4),
        )
        style.configure(
            "Status.TLabel",
            background=COLORS["rail"],
            foreground=COLORS["muted"],
            padding=(14, 7),
        )

        style.configure(
            "Nav.TButton",
            background=COLORS["rail"],
            foreground=COLORS["muted"],
            borderwidth=0,
            padding=(16, 12),
            anchor=tk.W,
        )
        style.map(
            "Nav.TButton",
            background=[("active", COLORS["inset"])],
            foreground=[("active", COLORS["text"])],
        )
        style.configure(
            "NavActive.TButton",
            background=COLORS["inset"],
            foreground=COLORS["teal"],
            borderwidth=0,
            padding=(16, 12),
            anchor=tk.W,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "Primary.TButton",
            background="#149C9A",
            foreground="#FFFFFF",
            bordercolor="#2DD4BF",
            padding=(12, 10),
            font=("Segoe UI", 10, "bold"),
        )
        style.map("Primary.TButton", background=[("active", "#18B7B1")])
        style.configure(
            "Secondary.TButton",
            background=COLORS["inset"],
            foreground=COLORS["text"],
            bordercolor=COLORS["border"],
            padding=(10, 7),
        )
        style.map("Secondary.TButton", background=[("active", "#17384B")])

        style.configure(
            "TNotebook", background=COLORS["card"], borderwidth=0, tabmargins=0
        )
        style.configure(
            "TNotebook.Tab",
            background=COLORS["card"],
            foreground=COLORS["muted"],
            padding=(16, 8),
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", COLORS["inset"])],
            foreground=[("selected", COLORS["teal"])],
        )
        style.configure(
            "Panel.TLabelframe",
            background=COLORS["card"],
            bordercolor=COLORS["border"],
            relief=tk.SOLID,
        )
        style.configure(
            "Panel.TLabelframe.Label",
            background=COLORS["card"],
            foreground=COLORS["muted"],
        )
        style.configure(
            "TEntry",
            fieldbackground=COLORS["inset"],
            foreground=COLORS["text"],
            bordercolor=COLORS["border"],
            insertcolor=COLORS["text"],
            padding=5,
        )
        style.configure(
            "TCombobox",
            fieldbackground=COLORS["inset"],
            background=COLORS["inset"],
            foreground=COLORS["text"],
            arrowcolor=COLORS["muted"],
            padding=5,
        )
        style.configure(
            "TCheckbutton",
            background=COLORS["card"],
            foreground=COLORS["text"],
        )
        style.map("TCheckbutton", background=[("active", COLORS["card"])])
        style.configure(
            "StagePending.TLabel",
            background=COLORS["inset"],
            foreground=COLORS["muted"],
            padding=(12, 7),
        )
        style.configure(
            "StageDone.TLabel",
            background="#123A3E",
            foreground=COLORS["teal"],
            padding=(12, 7),
            font=("Segoe UI", 9, "bold"),
        )

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, style="App.TFrame")
        outer.pack(fill=tk.BOTH, expand=True)
        self._build_header(outer)

        content = ttk.Frame(outer, style="App.TFrame")
        content.pack(fill=tk.BOTH, expand=True)
        rail = ttk.Frame(content, style="Rail.TFrame", width=172)
        rail.pack(side=tk.LEFT, fill=tk.Y)
        rail.pack_propagate(False)
        self.page_host = ttk.Frame(content, style="App.TFrame")
        self.page_host.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(rail, text="◉  MEG LAB", style="RailBrand.TLabel").pack(
            anchor=tk.W, padx=16, pady=(20, 22)
        )
        self.nav_buttons: dict[str, ttk.Button] = {}
        for key, label in (
            ("analysis", "▦   Signal Analysis"),
            ("brain", "◌   3D MEG Field"),
        ):
            button = ttk.Button(
                rail,
                text=label,
                style="Nav.TButton",
                command=lambda page=key: self.show_page(page),
            )
            button.pack(fill=tk.X, pady=2)
            self.nav_buttons[key] = button
        ttk.Separator(rail).pack(fill=tk.X, padx=16, pady=18)
        ttk.Label(
            rail,
            text="RESEARCH PROTOTYPE",
            style="WarningChip.TLabel",
        ).pack(anchor=tk.W, padx=14)
        ttk.Label(
            rail,
            text="Synthetic signals only\nNot for clinical use",
            style="RailMuted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=16, pady=10)

        self.pages: dict[str, ttk.Frame] = {}
        analysis_page = ttk.Frame(self.page_host, style="App.TFrame", padding=12)
        analysis_page.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.pages["analysis"] = analysis_page
        self._build_analysis_page(analysis_page)

        brain_page = ttk.Frame(self.page_host, style="App.TFrame")
        brain_page.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.pages["brain"] = brain_page
        self.brain_view = BrainFieldView(
            brain_page,
            self._signal_config,
            self._noise_config,
            self.status_text.set,
        )
        self.brain_view.pack(fill=tk.BOTH, expand=True)

        ttk.Label(outer, textvariable=self.status_text, style="Status.TLabel").pack(
            fill=tk.X, side=tk.BOTTOM
        )

    def _build_header(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent, style="App.TFrame", padding=(18, 12))
        header.pack(fill=tk.X)
        ttk.Label(header, text="MEG Signal Analysis Platform", style="Header.TLabel").pack(
            side=tk.LEFT
        )
        ttk.Label(header, text="●  Ready", style="Chip.TLabel").pack(
            side=tk.RIGHT, padx=(8, 0)
        )
        ttk.Label(header, textvariable=self.duration_chip, style="Chip.TLabel").pack(
            side=tk.RIGHT, padx=(8, 0)
        )
        ttk.Label(header, textvariable=self.sample_rate_chip, style="Chip.TLabel").pack(
            side=tk.RIGHT, padx=(8, 0)
        )
        ttk.Label(header, text="Simulation Mode", style="Chip.TLabel").pack(side=tk.RIGHT)
        ttk.Separator(parent).pack(fill=tk.X)

    def _build_analysis_page(self, parent: ttk.Frame) -> None:
        title_row = ttk.Frame(parent, style="App.TFrame")
        title_row.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(title_row, text="Signal Analysis", style="PageTitle.TLabel").pack(
            side=tk.LEFT
        )
        ttk.Label(
            title_row,
            text="Controlled generation, contamination and denoising",
            style="AppMuted.TLabel",
        ).pack(side=tk.LEFT, padx=12, pady=(4, 0))
        stage_bar = ttk.Frame(title_row, style="App.TFrame")
        stage_bar.pack(side=tk.RIGHT)
        self.stage_labels: list[ttk.Label] = []
        for text in ("1  Generate Signal", "2  Add Noise", "3  Denoise"):
            label = ttk.Label(stage_bar, text=text, style="StagePending.TLabel")
            label.pack(side=tk.LEFT, padx=3)
            self.stage_labels.append(label)

        body = ttk.Frame(parent, style="App.TFrame")
        body.pack(fill=tk.BOTH, expand=True)
        inspector = ttk.Frame(body, style="Card.TFrame", width=320, padding=12)
        inspector.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        inspector.pack_propagate(False)
        visual = ttk.Frame(body, style="Card.TFrame", padding=8)
        visual.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        chart_tabs = ttk.Notebook(visual)
        chart_tabs.pack(fill=tk.BOTH, expand=True)
        time_tab = ttk.Frame(chart_tabs, style="Card.TFrame")
        spectrum_tab = ttk.Frame(chart_tabs, style="Card.TFrame")
        chart_tabs.add(time_tab, text="Time Domain")
        chart_tabs.add(spectrum_tab, text="Power Spectrum")

        self.time_figure = Figure(figsize=(8.4, 5.8), dpi=100, facecolor=COLORS["card"])
        self.time_axes = [self.time_figure.add_subplot(311 + index) for index in range(3)]
        self.time_canvas = FigureCanvasTkAgg(self.time_figure, master=time_tab)
        self.time_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        time_toolbar = NavigationToolbar2Tk(self.time_canvas, time_tab, pack_toolbar=False)
        time_toolbar.configure(background=COLORS["card"])
        time_toolbar.update()
        time_toolbar.pack(fill=tk.X)

        self.spectrum_figure = Figure(
            figsize=(8.4, 5.8), dpi=100, facecolor=COLORS["card"]
        )
        self.spectrum_axis = self.spectrum_figure.add_subplot(111)
        self.spectrum_canvas = FigureCanvasTkAgg(self.spectrum_figure, master=spectrum_tab)
        self.spectrum_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        spectrum_toolbar = NavigationToolbar2Tk(
            self.spectrum_canvas, spectrum_tab, pack_toolbar=False
        )
        spectrum_toolbar.configure(background=COLORS["card"])
        spectrum_toolbar.update()
        spectrum_toolbar.pack(fill=tk.X)

        metrics = ttk.Frame(visual, style="Card.TFrame")
        metrics.pack(fill=tk.X, pady=(8, 0))
        self._metric_card(
            metrics, "SNR", self.snr_text, self.snr_change_text, COLORS["cyan"]
        ).pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self._metric_card(
            metrics, "RMSE", self.rmse_text, self.rmse_change_text, COLORS["coral"]
        ).pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        self._metric_card(
            metrics,
            "CORRELATION",
            self.correlation_text,
            self.correlation_change_text,
            COLORS["green"],
        ).pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        ttk.Label(inspector, text="Pipeline Parameters", style="CardTitle.TLabel").pack(
            anchor=tk.W, pady=(0, 8)
        )
        controls = ttk.Notebook(inspector)
        controls.pack(fill=tk.BOTH, expand=True)
        signal_tab = ttk.Frame(controls, style="Card.TFrame", padding=8)
        noise_tab = ttk.Frame(controls, style="Card.TFrame", padding=8)
        filter_tab = ttk.Frame(controls, style="Card.TFrame", padding=8)
        controls.add(signal_tab, text="Signal")
        controls.add(noise_tab, text="Noise")
        controls.add(filter_tab, text="Filter")
        self._build_signal_controls(signal_tab)
        self._build_noise_controls(noise_tab)
        self._build_filter_controls(filter_tab)

        ttk.Button(
            inspector,
            text="▶  Run Full Pipeline",
            command=self.run_full_pipeline,
            style="Primary.TButton",
        ).pack(fill=tk.X, pady=(10, 4))
        stage_actions = ttk.Frame(inspector, style="Card.TFrame")
        stage_actions.pack(fill=tk.X)
        for text, command in (
            ("Generate", self.generate_stage),
            ("Add Noise", self.noise_stage),
            ("Denoise", self.denoise_stage),
        ):
            ttk.Button(
                stage_actions, text=text, command=command, style="Secondary.TButton"
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.export_button = ttk.Button(
            inspector,
            text="⇩  Export Results",
            command=self.export_result,
            style="Secondary.TButton",
            state=tk.DISABLED,
        )
        self.export_button.pack(fill=tk.X, pady=(6, 0))

    def _metric_card(
        self,
        parent: ttk.Frame,
        title: str,
        value: tk.StringVar,
        detail: tk.StringVar,
        accent: str,
    ) -> ttk.Frame:
        card = ttk.Frame(parent, style="Inset.TFrame", padding=(14, 10))
        ttk.Label(card, text=title, style="MetricSmall.TLabel", foreground=accent).pack(
            anchor=tk.W
        )
        ttk.Label(card, textvariable=value, style="MetricValue.TLabel").pack(anchor=tk.W)
        ttk.Label(card, textvariable=detail, style="MetricSmall.TLabel").pack(anchor=tk.W)
        return card

    def _entry(self, parent: ttk.Frame, row: int, label: str, variable: tk.Variable) -> None:
        ttk.Label(parent, text=label, style="Body.TLabel").grid(
            row=row, column=0, sticky=tk.W, pady=3
        )
        ttk.Entry(parent, textvariable=variable, width=11).grid(
            row=row, column=1, sticky=tk.EW, pady=3, padx=(8, 0)
        )
        parent.columnconfigure(1, weight=1)

    def _build_signal_controls(self, parent: ttk.Frame) -> None:
        fields = (
            ("Sampling (Hz)", self.sample_rate),
            ("Duration (s)", self.duration),
            ("Alpha Hz", self.alpha_frequency),
            ("Alpha amp", self.alpha_amplitude),
            ("Beta Hz", self.beta_frequency),
            ("Beta amp", self.beta_amplitude),
            ("Gamma Hz", self.gamma_frequency),
            ("Gamma amp", self.gamma_amplitude),
            ("Plot window (s)", self.view_seconds),
        )
        for row, (label, variable) in enumerate(fields):
            self._entry(parent, row, label, variable)

    def _build_noise_controls(self, parent: ttk.Frame) -> None:
        ttk.Checkbutton(
            parent, text="Gaussian noise", variable=self.gaussian_enabled
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        self._entry(parent, 1, "Std deviation", self.gaussian_std)
        ttk.Separator(parent).grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=6)
        ttk.Checkbutton(
            parent, text="Power-line", variable=self.power_line_enabled
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W)
        self._entry(parent, 4, "Frequency (Hz)", self.power_line_frequency)
        self._entry(parent, 5, "Amplitude", self.power_line_amplitude)
        ttk.Separator(parent).grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=6)
        ttk.Checkbutton(parent, text="Drift", variable=self.drift_enabled).grid(
            row=7, column=0, columnspan=2, sticky=tk.W
        )
        self._entry(parent, 8, "Frequency (Hz)", self.drift_frequency)
        self._entry(parent, 9, "Amplitude", self.drift_amplitude)
        self._entry(parent, 10, "Random seed", self.seed)

    def _build_filter_controls(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Method", style="Body.TLabel").grid(
            row=0, column=0, sticky=tk.W, pady=3
        )
        ttk.Combobox(
            parent,
            textvariable=self.method_label,
            values=tuple(METHOD_LABELS),
            state="readonly",
            width=21,
        ).grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(0, 8))
        for row, (label, variable) in enumerate(
            (
                ("Low cutoff (Hz)", self.low_cutoff),
                ("High cutoff (Hz)", self.high_cutoff),
                ("Filter order", self.filter_order),
                ("Notch Q", self.notch_quality_factor),
            ),
            start=2,
        ):
            self._entry(parent, row, label, variable)

    def show_page(self, name: str) -> None:
        self.pages[name].tkraise()
        for key, button in self.nav_buttons.items():
            button.configure(style="NavActive.TButton" if key == name else "Nav.TButton")
        if name == "brain":
            self.brain_view.update_field()

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
            self.status_text.set(f"Error — {exc}")
            messagebox.showerror("MEG platform", str(exc), parent=self.root)

    def _set_stage(self, completed: int) -> None:
        for index, label in enumerate(self.stage_labels, start=1):
            label.configure(
                style="StageDone.TLabel" if index <= completed else "StagePending.TLabel"
            )

    def _sync_header(self, config: SignalConfig) -> None:
        self.sample_rate_chip.set(f"{config.sample_rate:g} Hz")
        self.duration_chip.set(f"{config.duration:g} s")
        self.brain_view.set_duration(config.duration)

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
        self._sync_header(config)
        self._set_stage(1)
        self._update_metric_text()
        self._draw_signals()
        self.brain_view.update_field()
        self.status_text.set(
            f"Generated {self.clean_signal.size:,} samples at {config.sample_rate:g} Hz"
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
        self._set_stage(2)
        self._update_metric_text()
        self._draw_signals()
        self.brain_view.update_field()
        names = ", ".join(self.noise_result.components) or "none"
        self.status_text.set(f"Added {names} noise — seed {noise_config.seed}")

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
            self.noise_result.noisy_signal, signal_config.sample_rate, denoise_config
        )
        self.denoised_metrics = evaluate_signal(self.clean_signal, self.denoised_signal)
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
        self._set_stage(3)
        self._update_metric_text()
        self._draw_signals()
        self.brain_view.update_field()
        change = self.denoised_metrics.snr_db - self.last_result.noisy_metrics.snr_db
        self.status_text.set(
            f"Processing complete — {self.method_label.get()} — SNR {change:+.2f} dB"
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
        self._sync_header(result.signal_config)
        self._set_stage(3)
        self._update_metric_text()
        self._draw_signals()
        self.brain_view.update_field()
        change = result.denoised_metrics.snr_db - result.noisy_metrics.snr_db
        self.status_text.set(
            f"Processing complete  •  Seed {result.noise_config.seed}  •  "
            f"{result.clean_signal.size:,} samples  •  SNR {change:+.2f} dB"
        )

    def _update_metric_text(self) -> None:
        if self.noisy_metrics is None:
            self.snr_text.set("--")
            self.snr_change_text.set("Waiting for data")
            self.rmse_text.set("--")
            self.rmse_change_text.set("No comparison")
            self.correlation_text.set("--")
            self.correlation_change_text.set("No comparison")
            return
        before = self.noisy_metrics
        after = self.denoised_metrics
        if after is None:
            self.snr_text.set(f"{before.snr_db:.2f} dB → --")
            self.snr_change_text.set("Denoising pending")
            self.rmse_text.set(f"{before.rmse:.4f} → --")
            self.rmse_change_text.set("Denoising pending")
            self.correlation_text.set(f"{before.correlation:.4f} → --")
            self.correlation_change_text.set("Denoising pending")
            return
        self.snr_text.set(f"{before.snr_db:.2f} → {after.snr_db:.2f} dB")
        self.snr_change_text.set(f"{after.snr_db - before.snr_db:+.2f} dB improvement")
        self.rmse_text.set(f"{before.rmse:.4f} → {after.rmse:.4f}")
        self.rmse_change_text.set(f"{before.rmse - after.rmse:+.4f} error reduction")
        self.correlation_text.set(
            f"{before.correlation:.4f} → {after.correlation:.4f}"
        )
        self.correlation_change_text.set("Closer to clean reference")

    @staticmethod
    def _style_plot_axis(axis, title: str) -> None:
        axis.set_facecolor(COLORS["card"])
        axis.set_title(title, loc="left", color=COLORS["text"], fontsize=10, pad=7)
        axis.tick_params(colors=COLORS["muted"], labelsize=8)
        axis.grid(True, color=COLORS["border"], alpha=0.55, linestyle="--", linewidth=0.6)
        for spine in axis.spines.values():
            spine.set_color(COLORS["border"])

    def _draw_signals(self) -> None:
        for axis in self.time_axes:
            axis.clear()
        self.spectrum_axis.clear()
        if self.clean_signal is None or self.time is None:
            first = self.time_axes[0]
            first.text(
                0.5,
                0.5,
                "Generate a signal to begin",
                ha="center",
                va="center",
                color=COLORS["muted"],
                transform=first.transAxes,
                fontsize=12,
            )
            for axis in self.time_axes:
                axis.set_axis_off()
            self.spectrum_axis.set_axis_off()
            self.time_canvas.draw_idle()
            self.spectrum_canvas.draw_idle()
            return

        sample_rate = float(self.sample_rate.get())
        view_seconds = float(self._number(self.view_seconds, "Plot window"))
        if view_seconds <= 0:
            raise ValueError("Plot window must be positive.")
        visible = self.time <= min(view_seconds, self.time[-1])
        series = [
            ("Clean Reference", self.clean_signal, COLORS["cyan"]),
            (
                "Noisy Signal",
                self.noise_result.noisy_signal if self.noise_result else None,
                COLORS["coral"],
            ),
            ("Denoised Signal", self.denoised_signal, COLORS["green"]),
        ]
        available = [values for _title, values, _color in series if values is not None]
        amplitude = max(
            1.0,
            max(float(np.max(np.abs(values[visible]))) for values in available) * 1.12,
        )
        for index, (title, values, color) in enumerate(series):
            axis = self.time_axes[index]
            axis.set_axis_on()
            self._style_plot_axis(axis, title)
            axis.set_ylim(-amplitude, amplitude)
            axis.set_ylabel("Amplitude", color=COLORS["muted"], fontsize=8)
            if values is not None:
                axis.plot(self.time[visible], values[visible], color=color, linewidth=1.05)
            else:
                axis.text(
                    0.5,
                    0.5,
                    "Pending",
                    ha="center",
                    va="center",
                    color=COLORS["muted"],
                    transform=axis.transAxes,
                )
            if index < 2:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel("Time (s)", color=COLORS["muted"], fontsize=8)
        self.time_figure.subplots_adjust(left=0.08, right=0.985, top=0.965, bottom=0.09, hspace=0.38)

        self.spectrum_axis.set_axis_on()
        self._style_plot_axis(self.spectrum_axis, "Power Spectral Density Comparison")
        for title, values, color in series:
            if values is None:
                continue
            segment = min(2048, values.size)
            frequencies, power = welch(values, fs=sample_rate, nperseg=segment)
            keep = frequencies <= min(100.0, sample_rate / 2.0)
            self.spectrum_axis.semilogy(
                frequencies[keep],
                np.maximum(power[keep], np.finfo(float).tiny),
                label=title,
                color=color,
                linewidth=1.35,
            )
        self.spectrum_axis.set_xlabel("Frequency (Hz)", color=COLORS["muted"])
        self.spectrum_axis.set_ylabel("PSD", color=COLORS["muted"])
        legend = self.spectrum_axis.legend(
            loc="upper right", facecolor=COLORS["inset"], edgecolor=COLORS["border"]
        )
        if legend:
            for text in legend.get_texts():
                text.set_color(COLORS["text"])
        self.spectrum_figure.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.12)
        self.time_canvas.draw_idle()
        self.spectrum_canvas.draw_idle()

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
        self.status_text.set(f"Exported {csv_path.name} and {json_path.name}")


def launch() -> None:
    root = tk.Tk()
    MEGPlatformApp(root)
    root.mainloop()
