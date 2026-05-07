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
