# MEG Signal Analysis and Denoising Platform

A Tkinter desktop prototype with two isolated workflows: controlled synthetic-data
validation and read-only analysis of imported real MEG recordings. It is an
engineering research prototype, not a clinical or diagnostic system.

## Core capabilities

- Configurable alpha, beta, and gamma neural oscillations.
- Gaussian sensor noise, 50/60 Hz power-line interference, and low-frequency drift.
- Butterworth low-pass and band-pass filters, notch filtering, and a combined method.
- SNR, RMSE, and correlation evaluation against the known clean reference.
- Time-domain and power-spectrum visualisation in a Tkinter GUI.
- Simulation Mode for repeatable functional verification with known ground truth.
- Real Data Mode for FIF/FIF.GZ, CTF `.ds`, KIT SQD/CON, CSV/TSV/TXT and NumPy data.
- Channel selection, reference-free metrics and read-only real-recording previews.
- Interactive anatomical 3D brain orientation model with rotatable sensor fields.
- Clear Superior/Inferior, Anterior/Posterior and left/right preset orientations.
- CSV/JSON result export and a repeatable command-line experiment runner.
- Unit and integration tests.

## Setup

Python 3.10 or newer is recommended. Tkinter is included with the standard Windows
Python installer; on Linux it may need to be installed as an operating-system package.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r Requirements.txt
python -m pip install -e .
```

The editable install and the project-level `.vscode/settings.json` allow both PyDev
and Pylance to resolve the `Algorithm` and `UI` packages. After creating the virtual
environment, reload VS Code or run **PyDev: Clear caches** once.

## Run the GUI

```powershell
python app.py
```

Use the buttons from left to right: **Generate Signal**, **Add Noise**, then
**Denoise Signal**. The **Run Full Pipeline** button performs all three stages.

The header contains a segmented **Simulation Mode / Real Data Mode** selector. In
Real Data Mode, choose **Load MEG File** for FIF/KIT or converted arrays, or **Load
CTF .ds Folder** for CTF recordings. The source file is never modified. The current
prototype limits its analysis working set to an in-memory preview of at most 60
seconds and 128 same-type MEG channels.

Delimited files may contain a first column named `time`, `time_s`, `times`,
`timestamp`, `seconds`, or `sec`; the sampling rate is then inferred from its regular
spacing. Otherwise the fallback sampling-rate field is used. NumPy inputs may be a
1-D signal or a 2-D array; select **Channels × Samples** or **Samples × Channels**
before loading so short multichannel recordings are never silently guessed. NPZ
archives can also provide `data`, `sample_rate`/`sfreq`, and `channel_names` arrays.
Converted text and compressed NPZ inputs are limited to 256 MiB; pre-crop larger
converted files before import.

## Run tests

```powershell
python -m pytest
```

## Run reproducible experiments

```powershell
python scripts/run_experiments.py
```

The experiment runner writes all individual runs to `Results/experiment_results.csv`
and the method/noise-level aggregates to `Results/experiment_summary.csv`. Every
experiment records the random seed and processing parameters so it can be repeated.

## Scope and limitations

The clean simulation reference is a simplified sum of sinusoidal frequency
components. Its noise models do not reproduce the full spatial, sensor, or biological
complexity of clinical MEG. Because real recordings have no known clean reference,
Real Data Mode deliberately does not report synthetic SNR/RMSE/correlation scores;
it reports descriptive RMS, peak-to-peak amplitude, dominant frequency and line-noise
attenuation instead.

The procedural brain surface provides anatomical orientation only. Simulation Mode
colours virtual sensors. Real Data Mode displays a 3D sensor field only when a native
recording provides valid MEG sensor coordinates; converted arrays show an explicit
unavailable state. Neither view performs MRI co-registration, forward modelling or
inverse source localisation, so it must not be used for diagnosis. Imported recordings
are ignored by Git under `Data/`; real-data exports default outside the project and
cannot be saved inside the project folder. Their JSON metadata withholds the source
filename and channel name. Always review derived files for identifiable information
before sharing them.
