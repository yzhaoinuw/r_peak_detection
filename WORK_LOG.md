# Work Log

## 2026-05-07

- Added `AGENTS.md` with project-specific Codex instructions, including the `ecg` environment and Miniconda environment location.
- Added `PROJECT_OVERVIEW.md` describing the R-peak detection pipeline, model flow, data contracts, training utilities, and current implementation notes.
- Fixed Git safe-directory setup outside the repository so Git commands can run despite the Windows ownership mismatch.
- Refactored `detect_r_peaks.py` so it can run both as a CLI and directly in Spyder with editable parameters in the `__main__` block.
- Refactored `visualize.py` so it can run both as a CLI and directly in Spyder with an editable `MAT_FILE` in the `__main__` block.
- Updated `detect_r_peaks.py` so `good_r_peaks` honors the selected threshold instead of always using `0.5`.
- Added wake/dex condition presets for candidate peak finding while keeping the default wake path equivalent to the original raw ECG `find_peaks` behavior.
- Added preprocessing/filtering helpers for future candidate-generation experiments, but left them inactive in the default presets to avoid reducing candidate recall before validation.
- Validated the edited Python files with `C:\Users\yzhao\miniconda3\envs\ecg\python.exe -m py_compile`.
- Compared candidate counts on `data/F26C_07112023_signals.mat`; the wake preset preserved all existing `detected_r_peaks`, while the dex preset was only slightly more selective.
- Converted the runtime code to a `src/r_peak_detection` package with modern `pyproject.toml` metadata.
- Added bundled checkpoint package data and made the large checkpoint the installed default model.
- Added optional visualization dependencies and the `visualize-r-peaks` console command.
- Added automatic visualization port selection with `--port`, `--host`, and `--no-browser` options.
- Moved persistent CLI settings to a user config path, with `R_PEAK_DETECTION_CONFIG` as an override for tests and scripted runs.
- Added basic `unittest` coverage and a GitHub Actions CI workflow.
- Removed clear development sketch and historical evaluation scripts while preserving training and labeling scripts.
- Removed top-level `detect_r_peaks.py` and `visualize.py` wrappers; the canonical scripts now live only under `src/r_peak_detection/` with Spyder-friendly editable `__main__` blocks.
- Added direct-file execution support for `src/r_peak_detection/detect_r_peaks.py` in Spyder while preserving package-relative imports for CLI usage.
- Hardened visualization server startup with TCP port probing, auto-retry on Dash address conflicts, and same-session Dash server cleanup.
- Set the visualization resampler's initial displayed sample count to 4096.
