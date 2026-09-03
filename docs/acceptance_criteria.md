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
| AC-11 | Explore a MEG sensor field in a rotatable anatomical 3D view. | Procedural brain, time slider, mode selector, and seven labelled preset views |
| AC-12 | Keep simulation validation and real-recording analysis separate. | Header mode selector and independent page/state/metrics |
| AC-13 | Import real MEG data without modifying or committing it. | MNE FIF/CTF/KIT reader, array reader, bounded preview, and `Data/*` ignore rule |
| AC-14 | Avoid full-reference quality claims on real recordings. | Real mode uses RMS, peak-to-peak, dominant frequency, and line attenuation only |
| AC-15 | Prevent synthetic fields from appearing as recorded spatial data. | Real 3D field requires native sensor coordinates and otherwise shows an unavailable state |

## Backend-GUI interface

The GUI constructs immutable `SignalConfig`, `NoiseConfig`, and `DenoiseConfig`
objects. Backend functions return NumPy arrays or immutable result dataclasses. The
end-to-end `run_experiment` function is the stable simulation boundary used by both
the GUI and the experiment runner. `load_real_recording` and `denoise_recording` are
the corresponding real-data boundaries. GUI code does not implement filtering
mathematics.

## Scope decision

ICA is not yet a core method. Although Real Data Mode can import multiple channels,
defensible ICA requires artefact-component inspection and rejection controls that are
outside the current validated scope. MRI co-registration and forward/inverse source
models are likewise required before the 3D page can claim cortical source activity.
