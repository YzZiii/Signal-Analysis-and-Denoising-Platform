"""Interactive anatomical 3D MEG sensor-space view embedded in Tkinter."""

from __future__ import annotations

from collections.abc import Callable
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from Algorithm.config import NoiseConfig, SignalConfig
from Algorithm.real_data import RealSpatialFrame
from Algorithm.spatial import generate_sensor_positions, simulate_sensor_field


SIMULATION_MODES = {
    "Clean Reference": "clean",
    "Noisy Field": "noisy",
    "Denoised Estimate": "denoised",
}
REAL_MODES = ("Raw Recording", "Processed Recording")

VIEW_PRESETS = {
    "Oblique": (24.0, -54.0, "OBLIQUE ORIENTATION", ("", "", "", "")),
    "Superior": (
        90.0,
        -90.0,
        "SUPERIOR VIEW · LOOKING TOWARD INFERIOR",
        ("ANTERIOR", "POSTERIOR", "LEFT", "RIGHT"),
    ),
    "Inferior": (
        -90.0,
        90.0,
        "INFERIOR VIEW · LOOKING TOWARD SUPERIOR",
        ("ANTERIOR", "POSTERIOR", "RIGHT", "LEFT"),
    ),
    "Anterior": (
        0.0,
        90.0,
        "ANTERIOR VIEW · LOOKING TOWARD POSTERIOR",
        ("SUPERIOR", "INFERIOR", "PATIENT RIGHT", "PATIENT LEFT"),
    ),
    "Posterior": (
        0.0,
        -90.0,
        "POSTERIOR VIEW · LOOKING TOWARD ANTERIOR",
        ("SUPERIOR", "INFERIOR", "PATIENT LEFT", "PATIENT RIGHT"),
    ),
    "Left": (
        0.0,
        180.0,
        "LEFT LATERAL VIEW · LOOKING RIGHT",
        ("SUPERIOR", "INFERIOR", "ANTERIOR", "POSTERIOR"),
    ),
    "Right": (
        0.0,
        0.0,
        "RIGHT LATERAL VIEW · LOOKING LEFT",
        ("SUPERIOR", "INFERIOR", "POSTERIOR", "ANTERIOR"),
    ),
}


class BrainFieldView(ttk.Frame):
    """Rotatable brain anatomy with synthetic or recorded MEG sensor fields."""

    def __init__(
        self,
        parent: tk.Misc,
        signal_config_provider: Callable[[], SignalConfig],
        noise_config_provider: Callable[[], NoiseConfig],
        status_callback: Callable[[str], None],
        real_field_provider: Callable[[float, bool], RealSpatialFrame | None] | None = None,
    ) -> None:
        super().__init__(parent, style="App.TFrame", padding=12)
        self.signal_config_provider = signal_config_provider
        self.noise_config_provider = noise_config_provider
        self.status_callback = status_callback
        self.real_field_provider = real_field_provider
        self.data_mode = "simulation"
        self.sensor_positions = generate_sensor_positions(96)
        self.field_mode = tk.StringVar(value="Denoised Estimate")
        self.time_value = tk.DoubleVar(value=0.125)
        self.time_text = tk.StringVar(value="Time 0.125 s")
        self.peak_text = tk.StringVar(value="Peak field — fT")
        self.rms_text = tk.StringVar(value="RMS field — fT")
        self.context_chip_text = tk.StringVar(value="SYNTHETIC VALIDATION")
        self.source_text = tk.StringVar(value="96 virtual sensors · generated field")
        self.view_direction_text = tk.StringVar(value="Oblique orientation")
        self.disclaimer_text = tk.StringVar()
        self._build()
        self.set_data_mode("simulation", self.signal_config_provider().duration)

    def _build(self) -> None:
        header = ttk.Frame(self, style="App.TFrame")
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="3D MEG Sensor Field", style="PageTitle.TLabel").pack(
            side=tk.LEFT
        )
        ttk.Label(
            header,
            textvariable=self.context_chip_text,
            style="WarningChip.TLabel",
        ).pack(side=tk.LEFT, padx=12)
        ttk.Label(
            header,
            text="Drag to rotate  •  Scroll to zoom",
            style="AppMuted.TLabel",
        ).pack(side=tk.RIGHT)

        content = ttk.Frame(self, style="App.TFrame")
        content.pack(fill=tk.BOTH, expand=True)
        plot_card = ttk.Frame(content, style="Card.TFrame", padding=8)
        plot_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        inspector = ttk.Frame(content, style="Card.TFrame", width=292, padding=16)
        inspector.pack(side=tk.RIGHT, fill=tk.Y)
        inspector.pack_propagate(False)

        self.figure = Figure(figsize=(8.5, 7.0), dpi=100, facecolor="#0C2030")
        self.axis = self.figure.add_subplot(111, projection="3d")
        self.axis.set_proj_type("ortho")
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_card)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_card, pack_toolbar=False)
        toolbar.configure(background="#0C2030")
        toolbar.update()
        toolbar.pack(fill=tk.X)
        self._draw_static_scene()
        self.canvas.mpl_connect("button_release_event", self._on_manual_rotation)

        ttk.Label(inspector, text="Field Controls", style="CardTitle.TLabel").pack(
            anchor=tk.W, pady=(0, 14)
        )
        ttk.Label(inspector, text="Display data", style="Muted.TLabel").pack(anchor=tk.W)
        self.mode_box = ttk.Combobox(
            inspector,
            textvariable=self.field_mode,
            values=tuple(SIMULATION_MODES),
            state="readonly",
        )
        self.mode_box.pack(fill=tk.X, pady=(4, 12))
        self.mode_box.bind("<<ComboboxSelected>>", lambda _event: self.update_field())

        ttk.Label(inspector, textvariable=self.time_text, style="Body.TLabel").pack(
            anchor=tk.W
        )
        self.time_slider = ttk.Scale(
            inspector,
            from_=0.0,
            to=1.0,
            variable=self.time_value,
            command=lambda _value: self.update_field(),
        )
        self.time_slider.pack(fill=tk.X, pady=(6, 14))

        metric_card = ttk.Frame(inspector, style="Inset.TFrame", padding=12)
        metric_card.pack(fill=tk.X, pady=(0, 14))
        ttk.Label(metric_card, textvariable=self.peak_text, style="MetricValue.TLabel").pack(
            anchor=tk.W
        )
        ttk.Label(metric_card, textvariable=self.rms_text, style="MetricSmall.TLabel").pack(
            anchor=tk.W, pady=(5, 0)
        )
        ttk.Label(
            metric_card,
            textvariable=self.source_text,
            style="MetricSmall.TLabel",
            wraplength=240,
        ).pack(anchor=tk.W, pady=(5, 0))

        ttk.Label(inspector, text="Anatomical orientation", style="Muted.TLabel").pack(
            anchor=tk.W
        )
        ttk.Label(
            inspector,
            textvariable=self.view_direction_text,
            style="Body.TLabel",
            wraplength=250,
        ).pack(anchor=tk.W, pady=(3, 7))
        preset_grid = ttk.Frame(inspector, style="Card.TFrame")
        preset_grid.pack(fill=tk.X)
        for index, name in enumerate(VIEW_PRESETS):
            ttk.Button(
                preset_grid,
                text=name,
                style="Secondary.TButton",
                command=lambda preset=name: self.set_view(preset),
            ).grid(
                row=index // 2,
                column=index % 2,
                sticky=tk.EW,
                padx=(0 if index % 2 == 0 else 3, 3 if index % 2 == 0 else 0),
                pady=3,
            )
        preset_grid.columnconfigure(0, weight=1)
        preset_grid.columnconfigure(1, weight=1)

        ttk.Separator(inspector).pack(fill=tk.X, pady=14)
        ttk.Label(
            inspector,
            textvariable=self.disclaimer_text,
            style="Muted.TLabel",
            wraplength=250,
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

    @staticmethod
    def _brain_facecolors(
        fold: np.ndarray, base: tuple[float, float, float]
    ) -> np.ndarray:
        intensity = np.clip(
            (fold - np.min(fold)) / max(float(np.ptp(fold)), 1e-9), 0, 1
        )
        colors = np.empty((*fold.shape, 4), dtype=float)
        for channel, value in enumerate(base):
            colors[..., channel] = np.clip(
                value * (0.82 + 0.24 * intensity), 0, 1
            )
        colors[..., 3] = 0.92
        return colors

    def _draw_cerebrum(self) -> None:
        polar = np.linspace(0.0, np.pi, 40)
        medial = np.linspace(-np.pi / 2, np.pi / 2, 54)
        pp, qq = np.meshgrid(polar, medial, indexing="ij")
        pole_taper = np.sin(pp) ** 0.7
        fold = 1.0 + pole_taper * (
            0.027 * np.sin(13 * pp + 2.0 * np.sin(4 * qq))
            + 0.017 * np.sin(9 * qq - 3 * pp)
            + 0.010 * np.sin(21 * pp + 5 * qq)
        )
        shell = np.sin(pp)
        frontal_shape = 1.0 + 0.045 * np.sin(qq)
        for sign in (-1.0, 1.0):
            x = sign * (0.022 + 0.68 * fold * shell * np.cos(qq))
            y = 0.92 * frontal_shape * fold * shell * np.sin(qq) + 0.025
            z = 0.70 * fold * np.cos(pp)
            z = np.where(z < -0.43, -0.43 + 0.58 * (z + 0.43), z)
            colors = self._brain_facecolors(fold, (0.055, 0.43, 0.53))
            self.axis.plot_surface(
                x,
                y,
                z,
                facecolors=colors,
                linewidth=0,
                antialiased=True,
                shade=True,
            )

    def _draw_cerebellum(self) -> None:
        polar = np.linspace(0.05, np.pi - 0.05, 20)
        azimuth = np.linspace(0, 2 * np.pi, 34)
        pp, aa = np.meshgrid(polar, azimuth, indexing="ij")
        folia = 1.0 + 0.035 * np.sin(14 * pp) + 0.012 * np.cos(8 * aa)
        for sign in (-1.0, 1.0):
            x = sign * 0.19 + 0.27 * folia * np.sin(pp) * np.cos(aa)
            y = -0.61 + 0.28 * folia * np.sin(pp) * np.sin(aa)
            z = -0.49 + 0.18 * folia * np.cos(pp)
            colors = self._brain_facecolors(folia, (0.08, 0.31, 0.39))
            self.axis.plot_surface(
                x,
                y,
                z,
                facecolors=colors,
                linewidth=0,
                antialiased=False,
                shade=True,
            )

    def _draw_brainstem(self) -> None:
        angles = np.linspace(0, 2 * np.pi, 24)
        heights = np.linspace(-0.88, -0.46, 18)
        aa, zz = np.meshgrid(angles, heights)
        radius = 0.12 - 0.025 * (zz + 0.88) / 0.42
        x = radius * np.cos(aa)
        y = -0.05 + 0.78 * radius * np.sin(aa)
        colors = np.empty((*x.shape, 4), dtype=float)
        colors[..., 0] = 0.07
        colors[..., 1] = 0.27
        colors[..., 2] = 0.34
        colors[..., 3] = 0.95
        self.axis.plot_surface(
            x,
            y,
            zz,
            facecolors=colors,
            linewidth=0,
            antialiased=False,
            shade=True,
        )

    def _draw_static_scene(self) -> None:
        self.axis.clear()
        self.axis.set_facecolor("#0C2030")
        self.axis.set_axis_off()
        self._draw_brainstem()
        self._draw_cerebellum()
        self._draw_cerebrum()

        sensor_xyz = self.sensor_positions * np.array((0.98, 1.12, 0.98))
        self.scatter = self.axis.scatter(
            sensor_xyz[:, 0],
            sensor_xyz[:, 1],
            sensor_xyz[:, 2],
            c=np.zeros(sensor_xyz.shape[0]),
            cmap="coolwarm",
            vmin=-120,
            vmax=120,
            s=35,
            depthshade=False,
            edgecolors="#E4F5FB",
            linewidths=0.42,
        )
        self.colorbar = self.figure.colorbar(
            self.scatter, ax=self.axis, shrink=0.68, pad=0.01
        )
        self.colorbar.set_label("Magnetic field (fT)", color="#B8CDD8")
        self.colorbar.ax.tick_params(colors="#8CA6B5")
        self.colorbar.outline.set_edgecolor("#244256")
        self.axis.set_xlim(-1.18, 1.18)
        self.axis.set_ylim(-1.26, 1.26)
        self.axis.set_zlim(-1.05, 1.15)
        self.axis.set_box_aspect((1.0, 1.08, 0.95))
        self.view_badge = self.axis.text2D(
            0.02,
            0.97,
            "",
            transform=self.axis.transAxes,
            color="#B8CDD8",
            fontsize=9,
            va="top",
            bbox={
                "boxstyle": "round,pad=0.35",
                "fc": "#102838",
                "ec": "#244256",
            },
        )
        self.orientation_artists = {
            "top": self.axis.text2D(
                0.48,
                0.93,
                "",
                transform=self.axis.transAxes,
                ha="center",
                color="#8CA6B5",
                fontsize=8,
            ),
            "bottom": self.axis.text2D(
                0.48,
                0.04,
                "",
                transform=self.axis.transAxes,
                ha="center",
                color="#8CA6B5",
                fontsize=8,
            ),
            "left": self.axis.text2D(
                0.04,
                0.50,
                "",
                transform=self.axis.transAxes,
                ha="left",
                va="center",
                color="#8CA6B5",
                fontsize=8,
            ),
            "right": self.axis.text2D(
                0.90,
                0.50,
                "",
                transform=self.axis.transAxes,
                ha="right",
                va="center",
                color="#8CA6B5",
                fontsize=8,
            ),
        }
        self.empty_text = self.axis.text2D(
            0.48,
            0.50,
            "",
            transform=self.axis.transAxes,
            ha="center",
            va="center",
            color="#FBBF24",
            fontsize=11,
            linespacing=1.5,
            bbox={
                "boxstyle": "round,pad=0.7",
                "fc": "#102838",
                "ec": "#FBBF24",
            },
        )
        self.empty_text.set_visible(False)
        self.figure.subplots_adjust(left=0.0, right=0.91, top=0.99, bottom=0.0)
        self.set_view("Oblique")

    def _set_sensor_positions(self, positions: np.ndarray) -> None:
        self.sensor_positions = np.asarray(positions, dtype=float)
        sensor_xyz = self.sensor_positions * np.array((0.98, 1.12, 0.98))
        self.scatter._offsets3d = (
            sensor_xyz[:, 0],
            sensor_xyz[:, 1],
            sensor_xyz[:, 2],
        )
        self.scatter.set_sizes(np.full(sensor_xyz.shape[0], 35.0))

    def _show_empty_real_state(self) -> None:
        self.scatter.set_visible(False)
        self.empty_text.set_text(
            "3D real sensor field unavailable\n\n"
            "Load a coordinate-aware FIF, CTF or KIT recording.\n"
            "Processed view also requires denoising first."
        )
        self.empty_text.set_visible(True)
        self.peak_text.set("Peak field —")
        self.rms_text.set("RMS field —")
        self.source_text.set("No real sensor coordinates available")
        self.canvas.draw_idle()

    def update_field(self) -> None:
        try:
            time_value = self.time_value.get()
            if self.data_mode == "simulation":
                signal_config = self.signal_config_provider()
                noise_config = self.noise_config_provider()
                self._set_sensor_positions(generate_sensor_positions(96))
                field = simulate_sensor_field(
                    time_value,
                    signal_config,
                    noise_config,
                    mode=SIMULATION_MODES[self.field_mode.get()],
                    sensor_positions=self.sensor_positions,
                )
                unit = "fT"
                self.source_text.set("96 virtual sensors · generated field")
            else:
                frame = (
                    None
                    if self.real_field_provider is None
                    else self.real_field_provider(
                        time_value,
                        self.field_mode.get() == "Processed Recording",
                    )
                )
                if frame is None:
                    self._show_empty_real_state()
                    return
                self._set_sensor_positions(frame.sensor_positions)
                field = frame.values
                unit = frame.unit
                self.source_text.set(f"{frame.source_name} · {frame.position_kind}")

            self.scatter.set_visible(True)
            self.empty_text.set_visible(False)
            limit = max(
                np.finfo(float).eps,
                float(np.percentile(np.abs(field), 98)) * 1.15,
            )
            self.scatter.set_array(np.asarray(field, dtype=float))
            self.scatter.set_clim(-limit, limit)
            self.colorbar.set_label(f"Sensor field ({unit})", color="#B8CDD8")
            self.time_text.set(f"Time {time_value:.3f} s")
            self.peak_text.set(f"Peak field {np.max(np.abs(field)):.3g} {unit}")
            self.rms_text.set(
                f"RMS field {np.sqrt(np.mean(np.asarray(field) ** 2)):.3g} {unit}"
            )
            self.canvas.draw_idle()
        except Exception as exc:
            self.status_callback(f"3D view error: {exc}")

    def set_duration(self, duration: float) -> None:
        maximum = max(0.001, float(duration))
        self.time_slider.configure(to=maximum)
        if self.time_value.get() > maximum:
            self.time_value.set(maximum)

    def set_data_mode(self, mode: str, duration: float | None = None) -> None:
        if mode not in {"simulation", "real"}:
            raise ValueError(f"Unknown application data mode: {mode}.")
        self.data_mode = mode
        if mode == "simulation":
            self.context_chip_text.set("SYNTHETIC VALIDATION")
            self.mode_box.configure(values=tuple(SIMULATION_MODES))
            if self.field_mode.get() not in SIMULATION_MODES:
                self.field_mode.set("Denoised Estimate")
            self.disclaimer_text.set(
                "The anatomical surface is an orientation model. The coloured points "
                "are a synthetic sensor-space projection, not brain-source localisation."
            )
        else:
            self.context_chip_text.set("REAL SENSOR SPACE")
            self.mode_box.configure(values=REAL_MODES)
            if self.field_mode.get() not in REAL_MODES:
                self.field_mode.set("Raw Recording")
            self.disclaimer_text.set(
                "Recorded values are shown only when valid sensor coordinates exist. "
                "The brain is illustrative; MRI co-registration and an inverse model "
                "are required for clinical source localisation."
            )
        if duration is not None:
            self.set_duration(duration)
        self.update_field()

    def set_view(self, preset: str) -> None:
        elevation, azimuth, caption, labels = VIEW_PRESETS[preset]
        self.axis.view_init(elev=elevation, azim=azimuth)
        self.view_badge.set_text(caption)
        self.view_direction_text.set(caption.title())
        for key, text in zip(("top", "bottom", "left", "right"), labels):
            self.orientation_artists[key].set_text(text)
        self.canvas.draw_idle()

    def _on_manual_rotation(self, event: object) -> None:
        if getattr(event, "inaxes", None) is not self.axis:
            return
        caption = (
            f"FREE ROTATION · ELEV {self.axis.elev:.0f}° · AZIM {self.axis.azim:.0f}°"
        )
        self.view_badge.set_text(caption)
        self.view_direction_text.set(caption.title())
        for artist in self.orientation_artists.values():
            artist.set_text("")
        self.canvas.draw_idle()
