# Core acceptance criteria

| ID | Requirement | Evidence |
| --- | --- | --- |
| AC-01 | Generate a configurable synthetic alpha/beta/gamma MEG-like signal. | `test_signal_shape_and_components` and GUI Signal tab |
| AC-02 | Add controlled, reproducible noise at selected levels. | Gaussian, power-line, and drift controls; fixed seed tests |
| AC-03 | Provide at least two technically valid denoising methods. | Low-pass, band-pass, notch, and combined implementations |
| AC-04 | Calculate SNR, RMSE, and correlation against the clean reference. | Metric unit tests and GUI evaluation panel |
| AC-05 | Display clean, noisy, and denoised signals. | Embedded time-domain and PSD plots |
| AC-06 | Provide an interactive Tkinter workflow. | Generate, Add Noise, Denoise, Run Full Pipeline, and Export controls |
| AC-07 | Record enough repeatable evidence for comparison. | 60 fixed-seed runs plus 12 aggregate conditions |
| AC-08 | Document the synthetic-data limitation. | README scope and limitations section |
| AC-09 | Reject invalid parameters and inconsistent workflow state. | Configuration validation and stage-state checks |
| AC-10 | Pass backend unit and integration tests. | `python -m pytest` |

## Backend-GUI interface

The GUI constructs immutable `SignalConfig`, `NoiseConfig`, and `DenoiseConfig`
objects. Backend functions return NumPy arrays or immutable result dataclasses. The
end-to-end `run_experiment` function is the stable integration boundary used by both
the GUI and the experiment runner. GUI code does not implement signal-processing
mathematics.

## Scope decision

ICA is not a core method because the current platform is single-channel and therefore
does not provide multiple mixed observations for defensible source separation. It can
be reconsidered together with the optional multichannel simulation extension.
