from pathlib import Path

import numpy as np
import pytest

from Algorithm.config import DenoiseConfig
from Algorithm.real_data import (
    RealRecording,
    denoise_recording,
    load_real_recording,
    summarise_real_channel,
)


def test_csv_loader_infers_sampling_rate_and_channel_names(tmp_path: Path) -> None:
    sample_rate = 200.0
    times = np.arange(400) / sample_rate
    table = np.column_stack(
        (
            times,
            np.sin(2 * np.pi * 10 * times),
            np.cos(2 * np.pi * 20 * times),
        )
    )
    path = tmp_path / "recording.csv"
    np.savetxt(path, table, delimiter=",", header="time_s,MEG011,MEG012", comments="")

    recording = load_real_recording(path, fallback_sample_rate=999.0)

    assert recording.data.shape == (2, 400)
    assert recording.sample_rate == pytest.approx(sample_rate)
    assert recording.channel_names == ("MEG011", "MEG012")
    assert recording.duration == pytest.approx(2.0)


def test_numpy_loader_uses_fallback_rate_and_denoises_every_channel(
    tmp_path: Path,
) -> None:
    sample_rate = 200.0
    times = np.arange(800) / sample_rate
    data = np.vstack(
        (
            np.sin(2 * np.pi * 10 * times) + 0.2 * np.sin(2 * np.pi * 50 * times),
            np.sin(2 * np.pi * 15 * times) + 0.2 * np.sin(2 * np.pi * 50 * times),
        )
    )
    path = tmp_path / "recording.npy"
    np.save(path, data)
    recording = load_real_recording(path, fallback_sample_rate=sample_rate)
    processed = denoise_recording(recording, DenoiseConfig())

    assert processed.shape == recording.data.shape
    assert np.all(np.isfinite(processed))
    metrics = summarise_real_channel(
        recording.data[0], processed[0], sample_rate, line_frequency=50.0
    )
    assert metrics.output_rms is not None
    assert metrics.line_attenuation_db is not None
    assert metrics.line_attenuation_db > 0


def test_numpy_layout_is_explicit_for_short_multichannel_arrays(
    tmp_path: Path,
) -> None:
    channels_by_samples = np.arange(306 * 100, dtype=float).reshape(306, 100)
    channel_first_path = tmp_path / "channel_first.npy"
    sample_first_path = tmp_path / "sample_first.npy"
    np.save(channel_first_path, channels_by_samples)
    np.save(sample_first_path, channels_by_samples.T)

    channel_first = load_real_recording(
        channel_first_path, 200.0, numpy_layout="channels_by_samples"
    )
    sample_first = load_real_recording(
        sample_first_path, 200.0, numpy_layout="samples_by_channels"
    )

    assert channel_first.data.shape == (128, 100)
    assert np.array_equal(channel_first.data, sample_first.data)
    assert channel_first.preview_limited


def test_npz_channel_names_can_identify_channel_axis(tmp_path: Path) -> None:
    values = np.arange(400 * 3, dtype=float).reshape(400, 3)
    path = tmp_path / "named_recording.npz"
    np.savez(
        path,
        data=values,
        sfreq=np.array(200.0),
        channel_names=np.array(("MEG001", "MEG002", "MEG003")),
    )
    recording = load_real_recording(path)
    assert recording.data.shape == (3, 400)
    assert recording.channel_names == ("MEG001", "MEG002", "MEG003")


def test_irregular_time_column_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "irregular.csv"
    path.write_text("time,MEG001\n0.0,1\n0.1,2\n0.25,3\n0.3,4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="regularly sampled"):
        load_real_recording(path)


def test_commented_time_header_is_recognised(tmp_path: Path) -> None:
    times = np.arange(100) / 100.0
    path = tmp_path / "commented_header.csv"
    np.savetxt(
        path,
        np.column_stack((times, np.sin(2 * np.pi * 10 * times))),
        delimiter=",",
        header="time_s,MEG001",
    )
    recording = load_real_recording(path, fallback_sample_rate=999.0)
    assert recording.sample_rate == pytest.approx(100.0)
    assert recording.channel_names == ("MEG001",)


def test_sensor_coordinate_count_must_match_channels() -> None:
    recording = RealRecording(
        data=np.ones((2, 64)),
        sample_rate=100.0,
        channel_names=("MEG001", "MEG002"),
        source_path="memory",
        source_format="TEST",
        unit="fT",
        sensor_positions=np.ones((1, 3)),
    )
    with pytest.raises(ValueError, match="Sensor positions"):
        recording.validate()


def test_nonfinite_sampling_rate_and_sensor_positions_are_rejected() -> None:
    with pytest.raises(ValueError, match="Sampling rate"):
        RealRecording(
            data=np.ones((1, 64)),
            sample_rate=float("nan"),
            channel_names=("MEG001",),
            source_path="memory",
            source_format="TEST",
            unit="fT",
        ).validate()
    with pytest.raises(ValueError, match="finite and non-zero"):
        RealRecording(
            data=np.ones((1, 64)),
            sample_rate=100.0,
            channel_names=("MEG001",),
            source_path="memory",
            source_format="TEST",
            unit="fT",
            sensor_positions=np.zeros((1, 3)),
        ).validate()


def test_line_attenuation_is_unavailable_above_nyquist() -> None:
    values = np.sin(2 * np.pi * 10 * np.arange(200) / 100.0)
    metrics = summarise_real_channel(values, values, 100.0, line_frequency=60.0)
    assert metrics.line_attenuation_db is None


def test_native_fif_loader_reads_meg_channels_and_positions(tmp_path: Path) -> None:
    mne = pytest.importorskip("mne")
    from mne.io.constants import FIFF
    from mne.transforms import Transform

    sample_rate = 200.0
    names = tuple(f"MEG{index:03d}" for index in range(1, 17))
    info = mne.create_info(names, sample_rate, ["mag"] * len(names))
    info["dev_head_t"] = Transform("meg", "head", np.eye(4))
    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    for index, angle in enumerate(angles):
        info["chs"][index]["coord_frame"] = FIFF.FIFFV_COORD_HEAD
        info["chs"][index]["loc"][:3] = (
            0.09 * np.cos(angle),
            0.09 * np.sin(angle),
            0.04 + 0.02 * np.cos(2 * angle),
        )
    times = np.arange(800) / sample_rate
    data = np.vstack(
        [
            1e-13 * np.sin(2 * np.pi * (8 + index * 0.2) * times)
            for index in range(len(names))
        ]
    )
    path = tmp_path / "test_meg_raw.fif"
    mne.io.RawArray(data, info, verbose="ERROR").save(
        path, overwrite=True, verbose="ERROR"
    )

    recording = load_real_recording(path)

    assert recording.source_format == "FIF"
    assert recording.data.shape == (16, 800)
    assert recording.unit == "fT"
    assert recording.sensor_positions is not None
    assert recording.sensor_positions.shape == (16, 3)
