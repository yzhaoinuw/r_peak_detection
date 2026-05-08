# R-Peak Detection

Command-line tools for detecting ECG R-peaks in MATLAB `.mat` files and, optionally, visualizing the detected peaks in a browser.

## Installation

Create and activate a conda environment:

```bash
conda create -n r_peak_detection python=3.11
conda activate r_peak_detection
```

Install the base CLI:

```bash
pip install -e .
```

Install with the optional browser visualization tool:

```bash
pip install -e .[visualization]
```

For model training and labeling utilities, install the training extras too:

```bash
pip install -e .[visualization,training]
```

## Detect R-Peaks

Run the detector on an ECG `.mat` file:

```bash
detect-r-peaks C:\path\to\recording_signals.mat
```

The input file must contain an `ECG` array. If `t_ECG` is present, the CLI uses it to estimate the sampling rate; otherwise it falls back to 1000 Hz.

The command writes these arrays back into the same `.mat` file:

- `detected_r_peaks`: candidate peak indices scored by the model
- `r_peak_confidence`: model confidence for each candidate
- `good_r_peaks`: candidates above the selected confidence threshold

Optional arguments:

```bash
detect-r-peaks C:\path\to\recording_signals.mat --threshold 0.6
detect-r-peaks C:\path\to\recording_signals.mat --condition dex
detect-r-peaks C:\path\to\recording_signals.mat --checkpoint_path C:\path\to\checkpoint.pth
```

The default checkpoint is bundled with the package. If you pass a custom threshold, condition, or checkpoint path, the choice is remembered in a user config file for future runs.

## Visualize R-Peaks

Install the visualization extra first:

```bash
pip install -e .[visualization]
```

Then run:

```bash
visualize-r-peaks C:\path\to\recording_signals.mat
```

The visualization CLI starts at port `8050` and automatically moves to the next free port if needed. To choose a port yourself:

```bash
visualize-r-peaks C:\path\to\recording_signals.mat --port 8060
```

Useful options:

```bash
visualize-r-peaks C:\path\to\recording_signals.mat --host 127.0.0.1
visualize-r-peaks C:\path\to\recording_signals.mat --no-browser
visualize-r-peaks C:\path\to\recording_signals.mat --keep-existing-server
```

The viewer uses Plotly Resampler and initially shows 4096 samples for responsive inspection of long ECG recordings. When run repeatedly from the same Spyder or Python session, the tool stops an existing Dash server on the selected port before starting a new one.

## Tests

Run the basic test suite:

```bash
python -m unittest discover -s tests
```

The GitHub Actions workflow installs the package and runs the same tests.
