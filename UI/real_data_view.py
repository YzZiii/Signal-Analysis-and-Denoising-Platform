"""Tkinter workspace for importing and analysing real MEG recordings."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from collections.abc import Callable

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.signal import welch

from Algorithm.config import DenoiseConfig
from Algorithm.real_data import (
    RealRecording,
    RealSpatialFrame,
    denoise_recording,
    load_real_recording,
    summarise_real_channel,
)


COLORS = {
    "background": "#06131F",
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


class RealDataView(ttk.Frame):
    """Independent real-recording state and analysis UI."""

    def __init__(
        self,
        parent: tk.Misc,
        status_callback: Callable[[str], None],
        recording_callback: Callable[[RealRecording], None],
        brain_refresh_callback: Callable[[], None],
    ) -> None:
        super().__init__(parent, style="App.TFrame", padding=12)
        self.status_callback = status_callback
        self.recording_callback = recording_callback
        self.brain_refresh_callback = brain_refresh_callback
        self.recording: RealRecording | None = None
        self.processed_data: np.ndarray | None = None
        self.processed_config: DenoiseConfig | None = None

        self.fallback_sample_rate = tk.StringVar(value="1000")
        self.numpy_layout = tk.StringVar(value="Channels × Samples")
        self.channel_name = tk.StringVar(value="")
        self.window_start = tk.DoubleVar(value=0.0)
        self.window_seconds = tk.StringVar(value="2.0")
        self.method_label = tk.StringVar(value="Notch + band-pass")
        self.low_cutoff = tk.StringVar(value="1")
        self.high_cutoff = tk.StringVar(value="45")
        self.line_frequency = tk.StringVar(value="50")
        self.filter_order = tk.StringVar(value="4")
        self.notch_quality = tk.StringVar(value="30")
        self.file_text = tk.StringVar(value="No recording loaded")
        self.format_text = tk.StringVar(value="Format  —")
        self.channel_text = tk.StringVar(value="Channels  —")
        self.duration_text = tk.StringVar(value="Duration  —")
        self.unit_text = tk.StringVar(value="Unit  —")
        self.sensor_text = tk.StringVar(value="Sensor layout  —")
        self.preview_text = tk.StringVar(value="Source file remains unchanged")
        self.rms_text = tk.StringVar(value="—")
        self.rms_detail = tk.StringVar(value="Load data to calculate")
        self.ptp_text = tk.StringVar(value="—")
        self.ptp_detail = tk.StringVar(value="Peak-to-peak amplitude")
        self.line_text = tk.StringVar(value="—")
        self.line_detail = tk.StringVar(value="Apply denoising to calculate")
        self.frequency_text = tk.StringVar(value="—")
        self.frequency_detail = tk.StringVar(value="Dominant input frequency")
        self._draw_after_id: str | None = None
        self._build()
        self._draw_empty()

    def _build(self) -> None:
        title_row = ttk.Frame(self, style="App.TFrame")
        title_row.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(title_row, text="Real / Imported Data", style="PageTitle.TLabel").pack(
            side=tk.LEFT
        )
        ttk.Label(
            title_row,
            text="Recording analysis · no clean-reference assumptions",
            style="AppMuted.TLabel",
        ).pack(side=tk.LEFT, padx=12, pady=(4, 0))
        ttk.Label(
            title_row,
            text="RESEARCH ANALYSIS · NOT FOR DIAGNOSIS",
            style="WarningChip.TLabel",
        ).pack(side=tk.RIGHT)

        body = ttk.Frame(self, style="App.TFrame")
        body.pack(fill=tk.BOTH, expand=True)
        inspector = ttk.Frame(body, style="Card.TFrame", width=320, padding=12)
        inspector.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        inspector.pack_propagate(False)
        visual = ttk.Frame(body, style="Card.TFrame", padding=8)
        visual.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tabs = ttk.Notebook(visual)
        tabs.pack(fill=tk.BOTH, expand=True)
        time_tab = ttk.Frame(tabs, style="Card.TFrame")
        spectrum_tab = ttk.Frame(tabs, style="Card.TFrame")
        tabs.add(time_tab, text="Channel Time Series")
        tabs.add(spectrum_tab, text="Power Spectrum")

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
        self.spectrum_canvas = FigureCanvasTkAgg(
            self.spectrum_figure, master=spectrum_tab
        )
        self.spectrum_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        spectrum_toolbar = NavigationToolbar2Tk(
            self.spectrum_canvas, spectrum_tab, pack_toolbar=False
        )
        spectrum_toolbar.configure(background=COLORS["card"])
        spectrum_toolbar.update()
        spectrum_toolbar.pack(fill=tk.X)

        metrics = ttk.Frame(visual, style="Card.TFrame")
        metrics.pack(fill=tk.X, pady=(8, 0))
        for index, (title, value, detail, accent) in enumerate(
            (
                ("RMS", self.rms_text, self.rms_detail, COLORS["cyan"]),
                ("PEAK TO PEAK", self.ptp_text, self.ptp_detail, COLORS["coral"]),
                ("LINE ATTENUATION", self.line_text, self.line_detail, COLORS["green"]),
                ("DOMINANT FREQ", self.frequency_text, self.frequency_detail, COLORS["amber"]),
            )
        ):
            card = ttk.Frame(metrics, style="Inset.TFrame", padding=(12, 9))
            card.pack(
                side=tk.LEFT,
                fill=tk.BOTH,
                expand=True,
                padx=(0 if index == 0 else 3, 0 if index == 3 else 3),
            )
            ttk.Label(
                card, text=title, style="MetricSmall.TLabel", foreground=accent
            ).pack(anchor=tk.W)
            ttk.Label(card, textvariable=value, style="MetricValue.TLabel").pack(
                anchor=tk.W
            )
            ttk.Label(card, textvariable=detail, style="MetricSmall.TLabel").pack(
                anchor=tk.W
            )

        ttk.Label(inspector, text="Recording Input", style="CardTitle.TLabel").pack(
            anchor=tk.W, pady=(0, 8)
        )
        ttk.Label(
            inspector,
            text="Fallback sampling rate (CSV/NumPy)",
            style="Muted.TLabel",
        ).pack(anchor=tk.W)
        ttk.Entry(inspector, textvariable=self.fallback_sample_rate).pack(
            fill=tk.X, pady=(4, 8)
        )
        ttk.Label(inspector, text="NumPy 2-D layout", style="Muted.TLabel").pack(
            anchor=tk.W
        )
        ttk.Combobox(
            inspector,
            textvariable=self.numpy_layout,
            values=("Channels × Samples", "Samples × Channels"),
            state="readonly",
        ).pack(fill=tk.X, pady=(4, 8))
        ttk.Button(
            inspector,
            text="＋  Load MEG File",
            style="Primary.TButton",
            command=self.load_file,
        ).pack(fill=tk.X, pady=3)
        ttk.Button(
            inspector,
            text="＋  Load CTF .ds Folder",
            style="Secondary.TButton",
            command=self.load_ctf_folder,
        ).pack(fill=tk.X, pady=3)

        metadata = ttk.Frame(inspector, style="Inset.TFrame", padding=10)
        metadata.pack(fill=tk.X, pady=10)
        ttk.Label(
            metadata,
            textvariable=self.file_text,
            style="MetricValue.TLabel",
            wraplength=260,
        ).pack(anchor=tk.W)
        for variable in (
            self.format_text,
            self.channel_text,
            self.duration_text,
            self.unit_text,
            self.sensor_text,
            self.preview_text,
        ):
            ttk.Label(
                metadata,
                textvariable=variable,
                style="MetricSmall.TLabel",
                wraplength=260,
            ).pack(anchor=tk.W, pady=(3, 0))

        ttk.Label(inspector, text="Selected channel", style="Muted.TLabel").pack(
            anchor=tk.W
        )
        self.channel_box = ttk.Combobox(
            inspector, textvariable=self.channel_name, values=(), state="disabled"
        )
        self.channel_box.pack(fill=tk.X, pady=(4, 8))
        self.channel_box.bind("<<ComboboxSelected>>", lambda _event: self._draw())

        ttk.Label(inspector, text="Window start", style="Muted.TLabel").pack(
            anchor=tk.W
        )
        self.start_scale = ttk.Scale(
            inspector,
            from_=0.0,
            to=1.0,
            variable=self.window_start,
            command=lambda _value: self._schedule_draw(),
        )
        self.start_scale.pack(fill=tk.X, pady=(4, 8))
        ttk.Label(inspector, text="Plot window (seconds)", style="Muted.TLabel").pack(
            anchor=tk.W
        )
        window_entry = ttk.Entry(inspector, textvariable=self.window_seconds)
        window_entry.pack(fill=tk.X, pady=(4, 8))
        window_entry.bind("<Return>", lambda _event: self._draw())

        ttk.Label(inspector, text="Real-data filter", style="Muted.TLabel").pack(
            anchor=tk.W
        )
        ttk.Combobox(
            inspector,
            textvariable=self.method_label,
            values=tuple(METHOD_LABELS),
            state="readonly",
        ).pack(fill=tk.X, pady=(4, 5))
        filter_grid = ttk.Frame(inspector, style="Card.TFrame")
        filter_grid.pack(fill=tk.X)
        for row, (label, variable) in enumerate(
            (
                ("Low Hz", self.low_cutoff),
                ("High Hz", self.high_cutoff),
                ("Line Hz", self.line_frequency),
                ("Order", self.filter_order),
                ("Notch Q", self.notch_quality),
            )
        ):
            ttk.Label(filter_grid, text=label, style="Body.TLabel").grid(
                row=row // 2,
                column=(row % 2) * 2,
                sticky=tk.W,
                padx=(0 if row % 2 == 0 else 8, 3),
                pady=2,
            )
            ttk.Entry(filter_grid, textvariable=variable, width=7).grid(
                row=row // 2,
                column=(row % 2) * 2 + 1,
                sticky=tk.EW,
                pady=2,
            )
        filter_grid.columnconfigure(1, weight=1)
        filter_grid.columnconfigure(3, weight=1)

        self.process_button = ttk.Button(
            inspector,
            text="▶  Apply Denoising to Preview",
            style="Primary.TButton",
            state=tk.DISABLED,
            command=self.process_recording,
        )
        self.process_button.pack(fill=tk.X, pady=(8, 3))
        self.export_button = ttk.Button(
            inspector,
            text="⇩  Export Selected Channel",
            style="Secondary.TButton",
            state=tk.DISABLED,
            command=self.export_selected_channel,
        )
        self.export_button.pack(fill=tk.X, pady=3)
        ttk.Label(
            inspector,
            text=(
                "Supported: FIF/FIF.GZ, KIT SQD/CON, CTF .ds, CSV/TSV/TXT, "
                "NPY and NPZ. Imports are read-only and capped to a 60 s preview."
            ),
            style="Muted.TLabel",
            wraplength=280,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(10, 0))

    @staticmethod
    def _style_axis(axis, title: str) -> None:
        axis.set_facecolor(COLORS["card"])
        axis.set_title(title, loc="left", color=COLORS["text"], fontsize=10, pad=7)
        axis.tick_params(colors=COLORS["muted"], labelsize=8)
        axis.grid(True, color=COLORS["border"], alpha=0.55, linestyle="--", linewidth=0.6)
        for spine in axis.spines.values():
            spine.set_color(COLORS["border"])

    def _run_action(self, action: Callable[[], None]) -> None:
        try:
            action()
        except Exception as exc:
            self.status_callback(f"Real-data error — {exc}")
            messagebox.showerror("Real MEG data", str(exc), parent=self.winfo_toplevel())

    def _denoise_config(self) -> DenoiseConfig:
        try:
            return DenoiseConfig(
                method=METHOD_LABELS[self.method_label.get()],
                low_cutoff=float(self.low_cutoff.get()),
                high_cutoff=float(self.high_cutoff.get()),
                line_frequency=float(self.line_frequency.get()),
                filter_order=int(self.filter_order.get()),
                notch_quality_factor=float(self.notch_quality.get()),
            )
        except (ValueError, KeyError) as exc:
            raise ValueError("Real-data filter settings must be valid numbers.") from exc

    def load_file(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.winfo_toplevel(),
            title="Open a real MEG recording or converted array",
            filetypes=(
                ("Supported MEG data", "*.fif *.fif.gz *.sqd *.con *.csv *.tsv *.txt *.npy *.npz"),
                ("MNE recordings", "*.fif *.fif.gz *.sqd *.con"),
                ("Tabular or NumPy", "*.csv *.tsv *.txt *.npy *.npz"),
                ("All files", "*.*"),
            ),
        )
        if path:
            self._run_action(lambda: self.load_path(path))

    def load_ctf_folder(self) -> None:
        path = filedialog.askdirectory(
            parent=self.winfo_toplevel(), title="Choose a CTF .ds recording folder"
        )
        if path:
            self._run_action(lambda: self.load_path(path))

    def load_path(self, path: str | Path) -> None:
        try:
            sample_rate = float(self.fallback_sample_rate.get())
        except ValueError as exc:
            raise ValueError("Fallback sampling rate must be a valid number.") from exc
        layout = (
            "channels_by_samples"
            if self.numpy_layout.get() == "Channels × Samples"
            else "samples_by_channels"
        )
        recording = load_real_recording(path, sample_rate, numpy_layout=layout)
        self.recording = recording
        self.processed_data = None
        self.processed_config = None
        self.channel_box.configure(values=recording.channel_names, state="readonly")
        self.channel_name.set(recording.channel_names[0])
        self.window_start.set(0.0)
        self.start_scale.configure(to=max(0.001, recording.duration))
        self.process_button.configure(state=tk.NORMAL)
        self.export_button.configure(state=tk.DISABLED)
        self.file_text.set(Path(recording.source_path).name)
        self.format_text.set(f"Format  {recording.source_format}")
        self.channel_text.set(f"Channels  {recording.channel_count}")
        self.duration_text.set(
            f"Duration  {recording.duration:.3f} s  ·  {recording.sample_rate:g} Hz"
        )
        self.unit_text.set(f"Display unit  {recording.unit}")
        self.sensor_text.set(recording.position_kind)
        self.preview_text.set(
            "Preview limited to 60 s / 128 channels"
            if recording.preview_limited
            else "Complete imported array · source unchanged"
        )
        self.recording_callback(recording)
        self._draw()
        self.brain_refresh_callback()
        self.status_callback(
            f"Loaded {Path(recording.source_path).name} · "
            f"{recording.channel_count} channels · {recording.duration:.2f} s preview"
        )

    def process_recording(self) -> None:
        self._run_action(self._process_recording)

    def _process_recording(self) -> None:
        if self.recording is None:
            raise ValueError("Load a recording before applying denoising.")
        config = self._denoise_config()
        self.processed_data = denoise_recording(self.recording, config)
        self.processed_config = config
        self.export_button.configure(state=tk.NORMAL)
        self._draw()
        self.brain_refresh_callback()
        self.status_callback(
            f"Processed {self.recording.channel_count} imported recording channels · "
            f"{config.method}"
        )

    def _channel_index(self) -> int:
        if self.recording is None:
            raise ValueError("No recording is loaded.")
        try:
            return self.recording.channel_names.index(self.channel_name.get())
        except ValueError as exc:
            raise ValueError("Choose a valid channel.") from exc

    def _schedule_draw(self) -> None:
        if self._draw_after_id is not None:
            self.after_cancel(self._draw_after_id)
        self._draw_after_id = self.after(80, self._draw)

    def _draw_empty(self) -> None:
        for axis in self.time_axes:
            axis.clear()
            axis.set_axis_off()
        self.time_axes[0].text(
            0.5,
            0.5,
            "Load a recording to begin real-data analysis",
            ha="center",
            va="center",
            color=COLORS["muted"],
            transform=self.time_axes[0].transAxes,
            fontsize=12,
        )
        self.spectrum_axis.clear()
        self.spectrum_axis.set_axis_off()
        self.time_canvas.draw_idle()
        self.spectrum_canvas.draw_idle()

    def _draw(self) -> None:
        self._draw_after_id = None
        if self.recording is None:
            self._draw_empty()
            return
        try:
            window_seconds = float(self.window_seconds.get())
        except ValueError:
            self.status_callback("Real-data error — plot window must be numeric")
            return
        if window_seconds <= 0:
            self.status_callback("Real-data error — plot window must be positive")
            return
        index = self._channel_index()
        raw_values = self.recording.data[index]
        output = None if self.processed_data is None else self.processed_data[index]
        maximum_start = max(
            0.0, self.recording.duration - min(window_seconds, self.recording.duration)
        )
        self.start_scale.configure(to=max(0.001, maximum_start))
        start = min(max(0.0, self.window_start.get()), maximum_start)
        if self.window_start.get() != start:
            self.window_start.set(start)
        stop = min(self.recording.duration, start + window_seconds)
        first = min(int(round(start * self.recording.sample_rate)), raw_values.size - 1)
        last = min(
            raw_values.size,
            max(first + 2, int(round(stop * self.recording.sample_rate))),
        )
        times = self.recording.times[first:last]
        raw_window = raw_values[first:last]
        output_window = None if output is None else output[first:last]
        residual_window = None if output is None else raw_window - output_window

        series = (
            ("Original Recording", raw_window, COLORS["cyan"]),
            ("Denoised Recording", output_window, COLORS["green"]),
            ("Removed Component", residual_window, COLORS["coral"]),
        )
        available = [values for _, values, _ in series if values is not None]
        amplitude = max(
            np.finfo(float).eps,
            max(float(np.max(np.abs(values))) for values in available) * 1.1,
        )
        for plot_index, (title, values, color) in enumerate(series):
            axis = self.time_axes[plot_index]
            axis.clear()
            axis.set_axis_on()
            self._style_axis(axis, title)
            axis.set_ylim(-amplitude, amplitude)
            axis.set_ylabel(self.recording.unit, color=COLORS["muted"], fontsize=8)
            if values is None:
                axis.text(
                    0.5,
                    0.5,
                    "Denoising pending",
                    ha="center",
                    va="center",
                    color=COLORS["muted"],
                    transform=axis.transAxes,
                )
            else:
                axis.plot(times, values, color=color, linewidth=1.0)
            if plot_index < 2:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel("Time (s)", color=COLORS["muted"], fontsize=8)
        self.time_figure.subplots_adjust(
            left=0.09, right=0.985, top=0.965, bottom=0.09, hspace=0.38
        )

        self.spectrum_axis.clear()
        self.spectrum_axis.set_axis_on()
        self._style_axis(self.spectrum_axis, f"PSD · {self.channel_name.get()}")
        segment = min(4096, raw_values.size)
        for label, values, color in (
            ("Original", raw_values, COLORS["cyan"]),
            ("Denoised", output, COLORS["green"]),
        ):
            if values is None:
                continue
            frequencies, psd = welch(
                values, fs=self.recording.sample_rate, nperseg=segment
            )
            keep = frequencies <= min(100.0, self.recording.sample_rate / 2.0)
            self.spectrum_axis.semilogy(
                frequencies[keep],
                np.maximum(psd[keep], np.finfo(float).tiny),
                color=color,
                linewidth=1.3,
                label=label,
            )
        self.spectrum_axis.set_xlabel("Frequency (Hz)", color=COLORS["muted"])
        self.spectrum_axis.set_ylabel(
            f"PSD ({self.recording.unit}²/Hz)", color=COLORS["muted"]
        )
        legend = self.spectrum_axis.legend(
            loc="upper right", facecolor=COLORS["inset"], edgecolor=COLORS["border"]
        )
        if legend:
            for text in legend.get_texts():
                text.set_color(COLORS["text"])
        self.spectrum_figure.subplots_adjust(
            left=0.11, right=0.98, top=0.93, bottom=0.12
        )

        try:
            line_frequency = (
                self.processed_config.line_frequency
                if self.processed_config is not None
                else float(self.line_frequency.get())
            )
        except ValueError:
            line_frequency = float("nan")
        metrics = summarise_real_channel(
            raw_values, output, self.recording.sample_rate, line_frequency
        )
        unit = self.recording.unit
        self.rms_text.set(
            f"{metrics.input_rms:.3g} → "
            + ("—" if metrics.output_rms is None else f"{metrics.output_rms:.3g}")
        )
        self.rms_detail.set(f"Input → output ({unit})")
        self.ptp_text.set(
            f"{metrics.input_peak_to_peak:.3g} → "
            + (
                "—"
                if metrics.output_peak_to_peak is None
                else f"{metrics.output_peak_to_peak:.3g}"
            )
        )
        self.ptp_detail.set(f"Input → output ({unit})")
        self.line_text.set(
            "—"
            if metrics.line_attenuation_db is None
            else f"{metrics.line_attenuation_db:+.2f} dB"
        )
        self.line_detail.set(
            f"At {line_frequency:g} Hz · descriptive only"
            if np.isfinite(line_frequency)
            and 0 < line_frequency < self.recording.sample_rate / 2.0
            else "Line frequency outside valid range"
        )
        self.frequency_text.set(f"{metrics.dominant_frequency:.2f} Hz")
        self.frequency_detail.set("Dominant input frequency")
        self.time_canvas.draw_idle()
        self.spectrum_canvas.draw_idle()

    def spatial_frame(self, time_value: float, processed: bool) -> RealSpatialFrame | None:
        """Return real sensor-space data only when valid coordinates are present."""
        if self.recording is None or self.recording.sensor_positions is None:
            return None
        values = self.processed_data if processed else self.recording.data
        if values is None:
            return None
        sample_index = min(
            values.shape[1] - 1,
            max(0, int(round(float(time_value) * self.recording.sample_rate))),
        )
        return RealSpatialFrame(
            values=np.asarray(values[:, sample_index], dtype=float),
            sensor_positions=np.asarray(self.recording.sensor_positions, dtype=float),
            unit=self.recording.unit,
            position_kind=self.recording.position_kind,
            source_name="Imported coordinate-aware recording",
        )

    def export_selected_channel(self) -> None:
        self._run_action(self._export_selected_channel)

    def _export_selected_channel(self) -> None:
        if self.recording is None or self.processed_data is None:
            raise ValueError("Apply denoising before exporting a real-data channel.")
        index = self._channel_index()
        if self.processed_config is None:
            raise ValueError("The processing configuration is unavailable; run denoising again.")
        documents = Path.home() / "Documents"
        output_dir = documents if documents.exists() else Path.home()
        path_text = filedialog.asksaveasfilename(
            parent=self.winfo_toplevel(),
            title="Export processed real MEG channel",
            initialdir=output_dir,
            initialfile="processed_recording_channel.csv",
            defaultextension=".csv",
            filetypes=(("CSV file", "*.csv"),),
        )
        if not path_text:
            return
        csv_path = Path(path_text)
        project_root = Path(__file__).resolve().parents[1]
        if csv_path.resolve().is_relative_to(project_root):
            raise ValueError(
                "For privacy, save real-data exports outside the project folder."
            )
        raw_values = self.recording.data[index]
        processed = self.processed_data[index]
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("time_s", "original", "denoised", "removed"))
            writer.writerows(
                zip(self.recording.times, raw_values, processed, raw_values - processed)
            )
        config = self.processed_config
        metrics = summarise_real_channel(
            raw_values, processed, self.recording.sample_rate, config.line_frequency
        )
        summary = {
            "mode": "real_data",
            "source_file": "withheld_for_privacy",
            "selected_channel_index": index,
            "sample_rate": self.recording.sample_rate,
            "display_unit": self.recording.unit,
            "filter_config": asdict(config),
            "descriptive_metrics": asdict(metrics),
            "note": "No clean reference is available; these are not clinical quality scores.",
        }
        csv_path.with_suffix(".json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        self.status_callback(
            "Exported processed channel outside the project by default; review it for "
            "identifiable information before sharing"
        )
