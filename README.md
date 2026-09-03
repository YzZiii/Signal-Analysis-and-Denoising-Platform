# MEG Signal Analysis and Denoising Platform

A Tkinter desktop prototype for controlled, reproducible evaluation of denoising
methods on synthetic MEG-like signals. It is an engineering research prototype,
not a clinical or diagnostic system.

## Core capabilities

- Configurable alpha, beta, and gamma neural oscillations.
- Gaussian sensor noise, 50/60 Hz power-line interference, and low-frequency drift.
- Butterworth low-pass and band-pass filters, notch filtering, and a combined method.
- SNR, RMSE, and correlation evaluation against the known clean reference.
- Time-domain and power-spectrum visualisation in a Tkinter GUI.
- Interactive rotatable 3D virtual-sensor field map with clean/noisy/denoised modes.
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

The clean reference is a simplified sum of sinusoidal frequency components. The
noise models are controlled approximations and do not reproduce the full spatial,
sensor, or biological complexity of clinical MEG recordings. ICA is deliberately not
included in the core single-channel workflow because meaningful ICA requires
multiple mixed observations. Real-data compatibility, multichannel simulation, ICA,
and AI denoising remain optional extensions. The 3D field map is a synthetic virtual
sensor projection for education and interface exploration; it is not anatomical source
localisation or a clinical interpretation tool.
