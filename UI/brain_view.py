"""Interactive Matplotlib 3D virtual-sensor view embedded in Tkinter."""

from __future__ import annotations

from collections.abc import Callable
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from Algorithm.config import NoiseConfig, SignalConfig
from Algorithm.spatial import generate_sensor_positions, simulate_sensor_field


MODE_KEYS = {
    "Clean Reference": "clean",
    "Noisy Field": "noisy",
    "Denoised Estimate": "denoised",
}


class BrainFieldView(ttk.Frame):
    """A rotatable synthetic MEG helmet field visualisation."""

    def __init__(
        self,
        parent: tk.Misc,
        signal_config_provider: Callable[[], SignalConfig],
        noise_config_provider: Callable[[], NoiseConfig],
        status_callback: Callable[[str], None],
    ) -> None:
        super().__init__(parent, style="App.TFrame", padding=12)
        self.signal_config_provider = signal_config_provider
        self.noise_config_provider = noise_config_provider
        self.status_callback = status_callback
        self.sensor_positions = generate_sensor_positions(96)
        self.mode = tk.StringVar(value="Denoised Estimate")
        self.time_value = tk.DoubleVar(value=0.125)
        self.time_text = tk.StringVar(value="Time 0.125 s")
        self.peak_text = tk.StringVar(value="Peak field -- fT")
        self.rms_text = tk.StringVar(value="RMS field -- fT")
        self._build()
        self.set_duration(self.signal_config_provider().duration)

    def _build(self) -> None:
        header = ttk.Frame(self, style="App.TFrame")
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="3D MEG Field Map", style="PageTitle.TLabel").pack(
            side=tk.LEFT
        )
        ttk.Label(
            header,
            text="SYNTHETIC SPATIAL PROJECTION",
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
        inspector = ttk.Frame(content, style="Card.TFrame", width=280, padding=16)
        inspector.pack(side=tk.RIGHT, fill=tk.Y)
        inspector.pack_propagate(False)

        self.figure = Figure(figsize=(8.5, 7.0), dpi=100, facecolor="#0C2030")
        self.axis = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_card)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_card, pack_toolbar=False)
        toolbar.configure(background="#0C2030")
        toolbar.update()
        toolbar.pack(fill=tk.X)
        self._draw_static_scene()

        ttk.Label(inspector, text="Field Controls", style="CardTitle.TLabel").pack(
            anchor=tk.W, pady=(0, 14)
        )
        ttk.Label(inspector, text="Display mode", style="Muted.TLabel").pack(anchor=tk.W)
        mode_box = ttk.Combobox(
            inspector,
            textvariable=self.mode,
            values=tuple(MODE_KEYS),
            state="readonly",
        )
        mode_box.pack(fill=tk.X, pady=(4, 16))
        mode_box.bind("<<ComboboxSelected>>", lambda _event: self.update_field())

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
        self.time_slider.pack(fill=tk.X, pady=(6, 18))

        metric_card = ttk.Frame(inspector, style="Inset.TFrame", padding=12)
        metric_card.pack(fill=tk.X, pady=(0, 16))
        ttk.Label(metric_card, textvariable=self.peak_text, style="MetricValue.TLabel").pack(
            anchor=tk.W
        )
        ttk.Label(metric_card, textvariable=self.rms_text, style="MetricSmall.TLabel").pack(
            anchor=tk.W, pady=(5, 0)
        )
        ttk.Label(metric_card, text="96 virtual sensors", style="MetricSmall.TLabel").pack(
            anchor=tk.W, pady=(5, 0)
        )

        ttk.Label(inspector, text="Preset views", style="Muted.TLabel").pack(anchor=tk.W)
        for label, elevation, azimuth in (
            ("Top", 90, -90),
            ("Front", 10, -90),
            ("Left", 12, 180),
            ("Right", 12, 0),
        ):
            ttk.Button(
                inspector,
                text=label,
                style="Secondary.TButton",
                command=lambda e=elevation, a=azimuth: self.set_view(e, a),
            ).pack(fill=tk.X, pady=3)

        ttk.Separator(inspector).pack(fill=tk.X, pady=16)
        ttk.Label(
            inspector,
            text=(
                "This view illustrates a virtual sensor field generated from the "
                "configured synthetic oscillations. It is not anatomical source "
                "localisation and must not be used for clinical interpretation."
            ),
            style="Muted.TLabel",
            wraplength=240,
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

    def _draw_static_scene(self) -> None:
        self.axis.clear()
        self.axis.set_facecolor("#0C2030")
        u = np.linspace(0, 2 * np.pi, 80)
        v = np.linspace(0, np.pi, 48)
        uu, vv = np.meshgrid(u, v)
        folds = 1.0 + 0.025 * np.sin(7 * uu + 2 * vv) + 0.018 * np.cos(11 * uu - vv)
        x = 0.78 * folds * np.sin(vv) * np.cos(uu)
        y = 0.92 * folds * np.sin(vv) * np.sin(uu)
        z = 0.74 * folds * np.cos(vv)
        self.axis.plot_surface(
            x,
            y,
            z,
            color="#176B87",
            edgecolor="#2B7C93",
            linewidth=0.15,
            alpha=0.22,
            antialiased=True,
            shade=True,
        )
        sensor_xyz = self.sensor_positions * np.array((0.96, 1.10, 0.94))
        self.scatter = self.axis.scatter(
            sensor_xyz[:, 0],
            sensor_xyz[:, 1],
            sensor_xyz[:, 2],
            c=np.zeros(sensor_xyz.shape[0]),
            cmap="coolwarm",
            vmin=-120,
            vmax=120,
            s=34,
            depthshade=False,
            edgecolors="#DDF6FF",
            linewidths=0.35,
        )
        colorbar = self.figure.colorbar(self.scatter, ax=self.axis, shrink=0.68, pad=0.02)
        colorbar.set_label("Synthetic magnetic field (fT)", color="#B8CDD8")
        colorbar.ax.tick_params(colors="#8CA6B5")
        colorbar.outline.set_edgecolor("#244256")
        self.axis.set_xlim(-1.18, 1.18)
        self.axis.set_ylim(-1.26, 1.26)
        self.axis.set_zlim(-1.05, 1.15)
        self.axis.set_box_aspect((1.0, 1.08, 0.95))
        self.axis.set_xlabel("Left ↔ Right", color="#8CA6B5", labelpad=8)
        self.axis.set_ylabel("Posterior ↔ Anterior", color="#8CA6B5", labelpad=8)
        self.axis.set_zlabel("Inferior ↔ Superior", color="#8CA6B5", labelpad=8)
        self.axis.tick_params(colors="#638092", labelsize=8)
        for pane in (self.axis.xaxis.pane, self.axis.yaxis.pane, self.axis.zaxis.pane):
            pane.set_facecolor("#0C2030")
            pane.set_edgecolor("#244256")
        self.axis.grid(False)
        self.axis.view_init(elev=24, azim=-54)
        self.figure.subplots_adjust(left=0.02, right=0.92, top=0.98, bottom=0.02)

    def update_field(self) -> None:
        try:
            signal_config = self.signal_config_provider()
            noise_config = self.noise_config_provider()
            maximum_time = max(0.001, signal_config.duration)
            self.time_value.set(min(self.time_value.get(), maximum_time))
            time_value = self.time_value.get()
            field = simulate_sensor_field(
                time_value,
                signal_config,
                noise_config,
                mode=MODE_KEYS[self.mode.get()],
                sensor_positions=self.sensor_positions,
            )
            limit = max(60.0, float(np.percentile(np.abs(field), 98)) * 1.15)
            self.scatter.set_array(field)
            self.scatter.set_clim(-limit, limit)
            self.time_text.set(f"Time {time_value:.3f} s")
            self.peak_text.set(f"Peak field {np.max(np.abs(field)):.1f} fT")
            self.rms_text.set(f"RMS field {np.sqrt(np.mean(field * field)):.1f} fT")
            self.canvas.draw_idle()
        except Exception as exc:
            self.status_callback(f"3D view error: {exc}")

    def set_duration(self, duration: float) -> None:
        maximum = max(0.001, float(duration))
        self.time_slider.configure(to=maximum)
        if self.time_value.get() > maximum:
            self.time_value.set(maximum)
        self.update_field()

    def set_view(self, elevation: float, azimuth: float) -> None:
        self.axis.view_init(elev=elevation, azim=azimuth)
        self.canvas.draw_idle()
