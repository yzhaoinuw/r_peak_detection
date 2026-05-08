# R-Peak Detection Developer Overview

## Purpose

This project labels candidate ECG peaks with a trained PyTorch CNN that estimates the probability that each candidate is a true R-peak. The command-line tool currently does both stages:

1. Load an ECG recording from a MATLAB `.mat` file.
2. Generate candidate peaks with `scipy.signal.find_peaks`.
3. Extract a fixed-width ECG neighborhood around each candidate.
4. Standardize each candidate segment independently.
5. Run a trained binary CNN checkpoint to assign R-peak confidence scores.
6. Save all candidate indices, confidence scores, and thresholded good peaks back into the same `.mat` file.

The reusable inference function, `validate_r_peaks`, also accepts candidate peak indices from another upstream peak-finding algorithm, which is the useful integration point when peak candidates are generated outside this package.

## Runtime Entry Point

The installed console command is:

```bash
detect-r-peaks path\to\recording.mat
```

`pyproject.toml` registers this command as:

```text
detect-r-peaks=r_peak_detection.detect_r_peaks:main
```

The CLI expects the input `.mat` file to contain:

- `ECG`: required ECG signal array. It is flattened before processing.

The visualization utility additionally expects:

- `t_ECG`: ECG time vector.

After inference, the CLI writes these fields back to the same `.mat` file:

- `detected_r_peaks`: candidate peak indices that survived edge filtering.
- `r_peak_confidence`: model output probabilities, aligned with `detected_r_peaks`.
- `good_r_peaks`: candidate indices whose confidence passed the good-peak threshold.

## Inference Pipeline

The production path is split between `src/r_peak_detection/detect_r_peaks.py` and `src/r_peak_detection/run_inference.py`.

`r_peak_detection.detect_r_peaks` handles command-line arguments, persistent config, `.mat` I/O, and candidate generation:

```python
r_peak_inds, _ = find_candidate_peaks(ecg, fs, condition)
r_peak_inds, r_peak_conf = validate_r_peaks(
    ecg, r_peak_inds, checkpoint_path, decision_threshold=threshold
)
```

`run_inference.validate_r_peaks` handles model selection, segment extraction, normalization, batching, checkpoint loading, and probability generation.

### Candidate Peak Generation

The current CLI candidate generator uses `scipy.signal.find_peaks` on the raw ECG by default. This preserves the high-recall behavior of the original pipeline, where the CNN is responsible for rejecting non-R-peak candidates. Filtering support exists in `detect_r_peaks.py`, but it is not enabled in the default wake/dex presets until recall is validated against labeled data.

Two mouse ECG presets are available:

- `wake`: raw ECG, 50 ms minimum peak distance, fixed prominence 0.5. This matches the original candidate finder.
- `dex`: raw ECG, 52 ms minimum peak distance, fixed prominence 0.5. This is only slightly more selective while staying high-recall.

The minimum peak distance is computed from `t_ECG` when available, with a 1000 Hz fallback. If another upstream detector is preferred, call `validate_r_peaks(ecg, candidate_indices, checkpoint_path, ...)` directly.

The default active candidate-generation path is therefore still the original high-recall raw-ECG detector when `condition` is `wake`. Detrending, bandpass filtering, and adaptive MAD prominence helpers are present in the code, but inactive in the default presets until they can be validated as additive candidate sources rather than replacements for the raw detector.

### Segment Extraction

Segment length is selected by checkpoint filename:

- If `"large"` appears in `checkpoint_path`, the large CNN is used and each segment has 64 samples.
- Otherwise, the smaller CNN is used and each segment has 32 samples.

For the large model:

- Candidate peaks must be at least 32 samples from each edge.
- Segment indices are built from `[-32, ..., 0, 1, ..., 31]`, for 64 total samples.

For the smaller model:

- Candidate peaks must be at least 15 samples from the left edge and 16 samples from the right edge.
- Segment indices are built from `[-15, ..., 0, 1, ..., 16]`, for 32 total samples.

Each extracted segment is z-scored independently:

```python
mean = np.mean(r_peak_segments, axis=1, keepdims=True)
std = np.std(r_peak_segments, axis=1, keepdims=True)
r_peak_segments = (r_peak_segments - mean) / std
```

This keeps inference consistent with training, where each labeled segment is standardized the same way before being passed to the model.

### Model Inference

`R_Peak_Dataset` wraps segment arrays for PyTorch and adds the channel dimension expected by `Conv1d`, so model inputs have shape:

```text
batch_size x 1 x segment_length
```

Inference uses:

- `cuda` if available, otherwise `cpu`.
- `torch.load(..., map_location=device, weights_only=True)`.
- `model.eval()` and `torch.no_grad()`.
- `tqdm` progress display over the inference `DataLoader`.

The model output is a flattened NumPy array of probabilities in `[0, 1]`.

## Models

Model definitions live in `src/r_peak_detection/models.py`.

### `R_Peak_Classifier`

The smaller legacy model is intended for 32-sample segments:

- 1D convolution.
- ReLU.
- Max pooling.
- Fully connected hidden layer.
- Dropout.
- Sigmoid output.

Default constructor:

```python
R_Peak_Classifier(out_channels=8, hidden_size=16, dropout=0.5)
```

### `R_Peak_Classifier_Large`

The current default checkpoint uses the larger model for 64-sample segments:

- Conv1d: `1 -> out1`.
- MaxPool1d: `64 -> 32`.
- Conv1d: `out1 -> out2`.
- MaxPool1d: `32 -> 16`.
- Fully connected hidden layer.
- Dropout.
- Sigmoid output.

Default constructor:

```python
R_Peak_Classifier_Large(out1=32, out2=64, hidden_size=256, dropout=0.5)
```

The default configured checkpoint is:

```text
checkpoints/r_peak_classifier_large_out1_32_out2_64_bs_32_dt_0.5.pth
```

## Configuration

The CLI stores last-used inference settings in a user config file. Set `R_PEAK_DETECTION_CONFIG` to override the path. Otherwise Windows uses `%APPDATA%\r_peak_detection\config.json`, and other platforms use `~/.config/r_peak_detection/config.json`:

```json
{
    "decision_threshold": 0.5,
    "checkpoint_path": "C:\\path\\to\\r_peak_classifier_large_out1_32_out2_64_bs_32_dt_0.5.pth",
    "condition": "wake"
}
```

The CLI reads this file at startup. If `--threshold`, `--checkpoint_path`, or `--condition` are passed, the file is updated so the new values become the default for future runs.

## Training Data Workflow

Training data is stored as NumPy arrays in `data/`:

- `r_peak_data.npy`: labeled ECG segments.
- `r_peak_labels.npy`: binary labels, where `1` means good R-peak and `0` means bad peak.
- `r_peak_test_data.npy`: held-out or manually prepared test segments.
- `r_peak_test_labels.npy`: labels for test segments.

`make_r_peak_data.py` is an interactive labeling script. It:

1. Loads a source `.mat` ECG file.
2. Finds candidate peaks with the same SciPy `find_peaks` strategy.
3. Extracts 64-sample neighborhoods.
4. Displays each candidate segment and its z-scored version.
5. Prompts the user to label the peak as good (`1`) or bad (`0`).
6. Saves accumulated segments, labels, and used candidate indices to `.npy` files.

This script is research/development tooling, not part of the installed CLI.

## Training and Evaluation

`run_train.py` trains the large CNN from `r_peak_data.npy` and `r_peak_labels.npy`.

Important training behavior:

- Labels are stratified with `train_test_split(..., test_size=0.4)`.
- Each segment is standardized independently before splitting.
- Training uses `R_Peak_Dataset(..., noise_prob=0.5)` for augmentation.
- Loss is binary cross entropy.
- Optimizer is Adam with learning rate `0.001`.
- `ReduceLROnPlateau` lowers learning rate when validation loss stalls.
- Checkpoints are saved when validation accuracy exceeds the current best.

`run_test.py` loads `r_peak_test_data.npy` and `r_peak_test_labels.npy`, runs a selected checkpoint, thresholds probabilities at `decision_threshold`, and computes accuracy.

`find_best_hyperparameters.py` is an older search script for the smaller model. It repeatedly trains/validates over random splits and records best validation accuracy.

Historical ad hoc evaluation scripts were removed from the production repo surface. Recreate focused validation scripts or tests as needed for future labeled datasets.

## Visualization and Inspection

`visualize-r-peaks` opens an interactive Plotly/Plotly Resampler view of the ECG signal and detected candidates. Candidate markers are colored by model confidence from red to green. Visualization dependencies are optional and installed with `pip install -e .[visualization]`.

The intended flow is:

1. Run `detect-r-peaks recording.mat`.
2. Run `visualize-r-peaks recording.mat`.
3. Inspect confidence-colored candidate peaks in the browser.

The visualization server tries `127.0.0.1:8050` first and automatically moves to the next free port when that port is occupied. Users can force a port with `--port`, suppress browser opening with `--no-browser`, or opt out of same-session Dash cleanup with `--keep-existing-server`.

The Plotly Resampler figure initially shows 4096 samples. When the visualization is run repeatedly in the same Spyder or Python session, the tool stops a previously registered Dash server on the selected port before launching the new one. It does not kill unrelated Python processes from old kernels or other applications.

## File Map

- `src/r_peak_detection/detect_r_peaks.py`: installed CLI entry point; reads `.mat`, finds candidates, runs inference, writes outputs.
- `src/r_peak_detection/run_inference.py`: reusable inference function for scoring candidate peak indices.
- `src/r_peak_detection/models.py`: PyTorch CNN model definitions.
- `src/r_peak_detection/dataset.py`: PyTorch `Dataset` wrapper for ECG segments and optional labels/noise augmentation.
- `src/r_peak_detection/checkpoints/`: bundled PyTorch checkpoints used by the installed CLI.
- `run_train.py`: training script for the large CNN.
- `run_test.py`: checkpoint evaluation against saved test arrays.
- `make_r_peak_data.py`: interactive manual segment-labeling tool.
- `src/r_peak_detection/visualize.py`: interactive confidence visualization for labeled `.mat` files.
- `checkpoints/`: saved PyTorch checkpoint and log files.
- `data/`: example/source `.mat` files, labeled segment arrays, annotations, and archived training arrays.

## Dependencies

The package runtime requirements are listed in `pyproject.toml`:

- `numpy`
- `scipy`
- `torch`
- `tqdm`

Optional extras in `pyproject.toml` include:

- `visualization`: `dash`, `plotly`, `plotly-resampler`
- `training`: `scikit-learn`, `matplotlib`, `pandas`

## Current Implementation Notes

- `detect_r_peaks.py` passes the configured threshold into `validate_r_peaks`, but `validate_r_peaks` currently only returns probabilities and does not apply the threshold internally.
- Model selection is based on whether the checkpoint path string contains `"large"`. This works with the current checkpoint names, but it couples architecture selection to filename conventions.
- Segment standardization now guards against zero standard deviation by using `1` for constant segments.
- The CLI overwrites the input `.mat` file in place. This is convenient, but developers should use copies when testing.
- Development and training scripts still contain hard-coded local paths and subject filenames. Keep production behavior concentrated in the `src/r_peak_detection/` package.

## Common Developer Tasks

### Score externally generated candidate peaks

```python
from scipy.io import loadmat
from r_peak_detection.run_inference import validate_r_peaks

mat = loadmat("recording.mat")
ecg = mat["ECG"].flatten()

candidate_indices = ...  # produced by an upstream detector
peak_indices, probabilities = validate_r_peaks(
    ecg,
    candidate_indices,
    checkpoint_path="checkpoints/r_peak_classifier_large_out1_32_out2_64_bs_32_dt_0.5.pth",
)
```

`peak_indices` may be shorter than `candidate_indices` because candidates too close to the signal edges are removed before segment extraction.

### Change the default CLI checkpoint, threshold, or condition

Run the CLI once with explicit arguments:

```bash
detect-r-peaks recording.mat --threshold 0.6 --condition dex --checkpoint_path path\to\checkpoint.pth
```

Those values are saved to the user config file.

### Train a new large-model checkpoint

1. Build or update `data/r_peak_data.npy` and `data/r_peak_labels.npy`.
2. Adjust hyperparameters in `run_train.py` if needed.
3. Run `python run_train.py`.
4. Confirm the checkpoint architecture matches the constructor used by `validate_r_peaks`.
5. Point the user config file or `--checkpoint_path` to the new `.pth` file.

## Open Questions for Future Cleanup

- Should the CLI only validate upstream candidate peaks, or should SciPy `find_peaks` remain part of this package's production responsibility?
- Should thresholding live in `validate_r_peaks`, in the CLI, or in a small shared helper so the threshold behavior is consistent everywhere?
- Should the wake/dex detection presets be tuned against a manually corrected validation set and saved with explicit version labels?
- Should checkpoint metadata explicitly encode model architecture and segment length instead of relying on checkpoint filename strings?
- Should CI add heavier end-to-end inference tests against a tiny synthetic or fixture `.mat` file?
