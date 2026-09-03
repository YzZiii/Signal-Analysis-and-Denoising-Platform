"""Real-recording import and descriptive analysis helpers.

The loader never modifies the source recording.  MNE-backed files are loaded as a
bounded in-memory preview so large clinical recordings do not freeze the desktop UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch

from .config import DenoiseConfig
from .denoising import denoise_signal

FloatArray = NDArray[np.float64]

MAX_PREVIEW_SECONDS = 60.0
MAX_PREVIEW_CHANNELS = 128
MAX_CONVERTED_FILE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class RealRecording:
    """A bounded, analysis-ready excerpt from a real recording."""

    data: FloatArray
    sample_rate: float
    channel_names: tuple[str, ...]
    source_path: str
    source_format: str
    unit: str
    sensor_positions: FloatArray | None = None
    position_kind: str = "Sensor coordinates unavailable"
    preview_limited: bool = False

    def validate(self) -> None:
        values = np.asarray(self.data, dtype=float)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 32:
            raise ValueError(
                "Real data must contain at least one channel and 32 time samples."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("The recording contains NaN or infinite values.")
        if not np.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError("Sampling rate must be positive.")
        if len(self.channel_names) != values.shape[0]:
            raise ValueError("Channel-name count does not match the recording.")
        if any(not name.strip() for name in self.channel_names):
            raise ValueError("Channel names cannot be empty.")
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("Channel names must be unique.")
        if self.sensor_positions is not None:
            positions = np.asarray(self.sensor_positions, dtype=float)
            if positions.shape != (values.shape[0], 3):
                raise ValueError("Sensor positions must have shape (channels, 3).")
            if not np.all(np.isfinite(positions)) or np.any(
                np.linalg.norm(positions, axis=1) <= np.finfo(float).eps
            ):
                raise ValueError("Sensor positions must be finite and non-zero.")

    @property
    def channel_count(self) -> int:
        return int(self.data.shape[0])

    @property
    def sample_count(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration(self) -> float:
        return self.sample_count / self.sample_rate

    @property
    def last_time(self) -> float:
        return (self.sample_count - 1) / self.sample_rate

    @property
    def times(self) -> FloatArray:
        return np.arange(self.sample_count, dtype=float) / self.sample_rate


@dataclass(frozen=True)
class RealChannelMetrics:
    """Reference-free metrics suitable for a real recording."""

    input_rms: float
    output_rms: float | None
    input_peak_to_peak: float
    output_peak_to_peak: float | None
    line_attenuation_db: float | None
    dominant_frequency: float


@dataclass(frozen=True)
class RealSpatialFrame:
    """One real sensor-space field frame for the 3D viewer."""

    values: FloatArray
    sensor_positions: FloatArray
    unit: str
    position_kind: str
    source_name: str


def _normalise_matrix(values: FloatArray, layout: str) -> FloatArray:
    array = np.asanyarray(values)
    if array.ndim == 1:
        return array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError("Data arrays must be one- or two-dimensional.")
    if layout not in {"channels_by_samples", "samples_by_channels"}:
        raise ValueError("NumPy layout must identify the channel axis.")
    if layout == "samples_by_channels":
        array = array.T
    return array


def _read_delimited(path: Path, fallback_sample_rate: float) -> RealRecording:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t; ")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","

    first_line = next((line for line in sample.splitlines() if line.strip()), "")
    tokens = [token.strip().lstrip("#").strip() for token in first_line.split(delimiter)]
    has_header = False
    for token in tokens:
        try:
            float(token)
        except ValueError:
            has_header = True
            break

    values = np.genfromtxt(
        path,
        delimiter=delimiter,
        skip_header=1 if has_header else 0,
        dtype=float,
        ndmin=2,
    )
    values = values[~np.all(np.isnan(values), axis=1)]
    values = values[:, ~np.all(np.isnan(values), axis=0)]
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("The delimited file must contain a complete numeric table.")

    headers = tokens if has_header else []
    time_names = {"time", "time_s", "times", "timestamp", "seconds", "sec"}
    has_time = bool(headers and headers[0].strip().lower() in time_names)
    sample_rate = float(fallback_sample_rate)
    if has_time:
        if values.shape[1] < 2:
            raise ValueError("A time column must be followed by at least one channel.")
        times = values[:, 0]
        differences = np.diff(times)
        if np.any(differences <= 0):
            raise ValueError("The time column must increase strictly.")
        median_step = float(np.median(differences))
        if not np.allclose(differences, median_step, rtol=0.02, atol=1e-9):
            raise ValueError("The current filters require a regularly sampled time column.")
        sample_rate = 1.0 / median_step
        values = values[:, 1:]
        headers = headers[1:]

    # Delimited recordings conventionally store one time sample per row.
    matrix = np.asarray(values.T, dtype=float)
    original_channels, original_samples = matrix.shape
    matrix = matrix[:MAX_PREVIEW_CHANNELS]
    channel_names = tuple(
        headers[index].strip() if index < len(headers) and headers[index].strip()
        else f"Channel {index + 1:03d}"
        for index in range(matrix.shape[0])
    )
    maximum_samples = max(32, int(round(MAX_PREVIEW_SECONDS * sample_rate)))
    limited = (
        original_samples > maximum_samples
        or original_channels > MAX_PREVIEW_CHANNELS
    )
    matrix = matrix[:, :maximum_samples]
    recording = RealRecording(
        data=matrix,
        sample_rate=sample_rate,
        channel_names=channel_names,
        source_path=str(path),
        source_format=path.suffix.lstrip(".").upper() or "TEXT",
        unit="native units",
        preview_limited=limited,
    )
    recording.validate()
    return recording


def _choose_npz_array(archive: np.lib.npyio.NpzFile) -> FloatArray:
    if "data" in archive.files:
        return np.asarray(archive["data"], dtype=float)
    candidates: list[tuple[int, str, FloatArray]] = []
    for key in archive.files:
        value = np.asarray(archive[key])
        if np.issubdtype(value.dtype, np.number) and value.ndim in (1, 2):
            candidates.append((value.size, key, np.asarray(value, dtype=float)))
    if not candidates:
        raise ValueError("NPZ must contain a numeric one- or two-dimensional array.")
    return max(candidates, key=lambda item: item[0])[2]


def _read_numpy(
    path: Path, fallback_sample_rate: float, numpy_layout: str
) -> RealRecording:
    sample_rate = float(fallback_sample_rate)
    names: tuple[str, ...] | None = None
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            values = _choose_npz_array(archive)
            for key in ("sample_rate", "sfreq", "sampling_rate"):
                if key in archive.files:
                    sample_rate = float(np.asarray(archive[key]).squeeze())
                    break
            if "channel_names" in archive.files:
                names = tuple(str(item) for item in np.asarray(archive["channel_names"]))
    else:
        values = np.load(path, allow_pickle=False, mmap_mode="r")
    inferred_layout = numpy_layout
    if names is not None and np.ndim(values) == 2:
        if len(names) == values.shape[0] and len(names) != values.shape[1]:
            inferred_layout = "channels_by_samples"
        elif len(names) == values.shape[1] and len(names) != values.shape[0]:
            inferred_layout = "samples_by_channels"
    matrix = _normalise_matrix(values, inferred_layout)
    original_channels, original_samples = matrix.shape
    maximum_samples = max(32, int(round(MAX_PREVIEW_SECONDS * sample_rate)))
    matrix = np.array(
        matrix[:MAX_PREVIEW_CHANNELS, :maximum_samples], dtype=float, copy=True
    )
    if names is None or len(names) != matrix.shape[0]:
        names = tuple(f"Channel {index + 1:03d}" for index in range(matrix.shape[0]))
    recording = RealRecording(
        data=matrix,
        sample_rate=sample_rate,
        channel_names=names,
        source_path=str(path),
        source_format=path.suffix.lstrip(".").upper(),
        unit="native units",
        preview_limited=(
            original_samples > matrix.shape[1]
            or original_channels > matrix.shape[0]
        ),
    )
    recording.validate()
    return recording


def _sensor_positions(raw: object, picks: NDArray[np.int_]) -> FloatArray | None:
    from mne.io.constants import FIFF
    from mne.transforms import apply_trans

    head_positions: list[np.ndarray] = []
    for pick in picks:
        channel = raw.info["chs"][int(pick)]
        position = np.asarray(channel["loc"][:3], dtype=float)
        coordinate_frame = int(channel["coord_frame"])
        if not np.all(np.isfinite(position)) or np.linalg.norm(position) == 0:
            return None
        if coordinate_frame == int(FIFF.FIFFV_COORD_DEVICE):
            device_to_head = raw.info.get("dev_head_t")
            if device_to_head is None:
                return None
            position = np.asarray(apply_trans(device_to_head, position), dtype=float)
        elif coordinate_frame != int(FIFF.FIFFV_COORD_HEAD):
            return None
        head_positions.append(position)

    positions = np.asarray(head_positions, dtype=float)
    norms = np.linalg.norm(positions, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(float).eps):
        return None
    directions = positions / norms
    if np.unique(np.round(directions, decimals=5), axis=0).shape[0] != directions.shape[0]:
        # Signed planar-gradient pairs can occupy the same physical location; plotting
        # one over the other would hide a channel and imply a false scalar field.
        return None
    return directions


def _read_mne(path: Path) -> RealRecording:
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - exercised without optional dep
        raise RuntimeError(
            "MNE-Python is required for FIF, CTF and KIT recordings. "
            "Install the project requirements first."
        ) from exc

    raw = mne.io.read_raw(path, preload=False, verbose="ERROR")
    try:
        magnetometers = mne.pick_types(
            raw.info, meg="mag", eeg=False, ref_meg=False, exclude="bads"
        )
        gradiometers = mne.pick_types(
            raw.info, meg="grad", eeg=False, ref_meg=False, exclude="bads"
        )
        if magnetometers.size:
            available = magnetometers
            omitted_other_type = len(gradiometers)
            scale = 1e15
            unit = "fT"
            channel_kind = "magnetometers"
        else:
            if not gradiometers.size:
                raise ValueError("The selected recording contains no usable MEG channels.")
            available = gradiometers
            omitted_other_type = 0
            scale = 1e13  # T/m -> fT/cm
            unit = "fT/cm"
            channel_kind = "gradiometers"

        available_count = len(available)
        if available_count > MAX_PREVIEW_CHANNELS:
            selection = np.linspace(
                0, available_count - 1, MAX_PREVIEW_CHANNELS, dtype=int
            )
            picks = available[selection]
        else:
            picks = available
        sample_rate = float(raw.info["sfreq"])
        stop = min(raw.n_times, max(32, int(round(MAX_PREVIEW_SECONDS * sample_rate))))
        data = np.asarray(
            raw.get_data(
                picks=picks,
                start=0,
                stop=stop,
                reject_by_annotation="NaN",
            ),
            dtype=float,
        )
        valid_samples = np.all(np.isfinite(data), axis=0)
        annotation_limited = not bool(np.all(valid_samples))
        if annotation_limited:
            padded = np.concatenate(([False], valid_samples, [False])).astype(int)
            transitions = np.diff(padded)
            starts = np.flatnonzero(transitions == 1)
            stops = np.flatnonzero(transitions == -1)
            if starts.size == 0:
                raise ValueError("The preview contains no samples outside BAD annotations.")
            lengths = stops - starts
            best = int(np.argmax(lengths))
            data = data[:, starts[best] : stops[best]]
            if data.shape[1] < 32:
                raise ValueError(
                    "No continuous good segment in the preview contains 32 samples."
                )
        data *= scale
        positions = _sensor_positions(raw, picks)
        selected_label = (
            f"{len(picks)}/{available_count} "
            if len(picks) < available_count
            else f"{len(picks)} "
        )
        type_note = (
            f" · {omitted_other_type} gradiometers excluded (different unit)"
            if omitted_other_type
            else ""
        )
        recording = RealRecording(
            data=data,
            sample_rate=sample_rate,
            channel_names=tuple(raw.ch_names[int(index)] for index in picks),
            source_path=str(path),
            source_format=(
                "CTF"
                if path.is_dir()
                else "FIF"
                if path.name.lower().endswith((".fif", ".fif.gz"))
                else path.suffix.lstrip(".").upper()
            ),
            unit=unit,
            sensor_positions=positions,
            position_kind=(
                f"{selected_label}recorded {channel_kind} · MNE head coordinates{type_note}"
                if positions is not None
                else f"{selected_label}recorded {channel_kind} · head coordinates unavailable{type_note}"
            ),
            preview_limited=(
                stop < raw.n_times
                or len(picks) < available_count
                or annotation_limited
                or omitted_other_type > 0
            ),
        )
        recording.validate()
        return recording
    finally:
        raw.close()


def load_real_recording(
    source: str | Path,
    fallback_sample_rate: float = 1000.0,
    numpy_layout: str = "channels_by_samples",
) -> RealRecording:
    """Load a non-destructive, bounded preview of a supported recording."""
    path = Path(source).expanduser().resolve()
    if not np.isfinite(fallback_sample_rate) or fallback_sample_rate <= 0:
        raise ValueError("Fallback sampling rate must be positive.")
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        if path.suffix.lower() != ".ds":
            raise ValueError("Only CTF .ds recording folders are supported.")
        return _read_mne(path)
    lower_name = path.name.lower()
    if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        if path.stat().st_size > MAX_CONVERTED_FILE_BYTES:
            raise ValueError("Converted text files must be 256 MiB or smaller.")
        return _read_delimited(path, fallback_sample_rate)
    if path.suffix.lower() in {".npy", ".npz"}:
        if path.suffix.lower() == ".npz" and path.stat().st_size > MAX_CONVERTED_FILE_BYTES:
            raise ValueError("Compressed NPZ files must be 256 MiB or smaller.")
        return _read_numpy(path, fallback_sample_rate, numpy_layout)
    if lower_name.endswith((".fif", ".fif.gz", ".sqd", ".con")):
        return _read_mne(path)
    raise ValueError(
        "Unsupported file. Choose FIF/FIF.GZ, KIT SQD/CON, CTF .ds, "
        "CSV/TSV/TXT, NPY or NPZ."
    )


def denoise_recording(recording: RealRecording, config: DenoiseConfig) -> FloatArray:
    """Apply the selected zero-phase filter independently to every channel."""
    recording.validate()
    config.validate(recording.sample_rate)
    try:
        return np.vstack(
            [
                denoise_signal(channel, recording.sample_rate, config)
                for channel in recording.data
            ]
        )
    except ValueError as exc:
        raise ValueError(
            "The recording is too short or incompatible with the selected filter; "
            "choose a lower order or a longer continuous segment."
        ) from exc


def summarise_real_channel(
    signal: FloatArray,
    processed: FloatArray | None,
    sample_rate: float,
    line_frequency: float,
) -> RealChannelMetrics:
    """Calculate reference-free descriptive values for a selected channel."""
    raw_values = np.asarray(signal, dtype=float)
    output = None if processed is None else np.asarray(processed, dtype=float)
    if raw_values.ndim != 1 or raw_values.size < 32:
        raise ValueError("A real-data channel must contain at least 32 samples.")
    segment = min(4096, raw_values.size)
    frequencies, raw_psd = welch(raw_values, fs=sample_rate, nperseg=segment)
    positive = frequencies > 0
    dominant = float(frequencies[positive][np.argmax(raw_psd[positive])])
    attenuation = None
    if output is not None and 0 < line_frequency < sample_rate / 2.0:
        _, output_psd = welch(output, fs=sample_rate, nperseg=segment)
        line_index = int(np.argmin(np.abs(frequencies - line_frequency)))
        before = max(float(raw_psd[line_index]), np.finfo(float).tiny)
        after = max(float(output_psd[line_index]), np.finfo(float).tiny)
        attenuation = float(10.0 * np.log10(before / after))
    return RealChannelMetrics(
        input_rms=float(np.sqrt(np.mean(np.square(raw_values)))),
        output_rms=(
            None if output is None else float(np.sqrt(np.mean(np.square(output))))
        ),
        input_peak_to_peak=float(np.ptp(raw_values)),
        output_peak_to_peak=None if output is None else float(np.ptp(output)),
        line_attenuation_db=attenuation,
        dominant_frequency=dominant,
    )
