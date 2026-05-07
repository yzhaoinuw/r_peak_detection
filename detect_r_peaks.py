# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 20:18:01 2024

@author: yzhao
"""

import os
import json
import argparse
import sys

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.io import loadmat, savemat

from run_inference import validate_r_peaks


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(PROJECT_DIR, "config.json")
DEFAULT_FS = 1000
DETECTION_PRESETS = {
    "wake": {
        "use_filter": False,
        "highpass_hz": 1.0,
        "lowpass_hz": 150.0,
        "distance_ms": 50,
        "prominence": 0.5,
        "prominence_mad": None,
    },
    "dex": {
        "use_filter": False,
        "highpass_hz": 0.5,
        "lowpass_hz": 100.0,
        "distance_ms": 52,
        "prominence": 0.5,
        "prominence_mad": None,
    },
}


def parse_args(checkpoint_path):
    parser = argparse.ArgumentParser(description="Extract R-peaks from ECG data.")
    parser.add_argument(
        "mat_file", type=str, help="Path to the .mat file containing the ECG data."
    )
    parser.add_argument(
        "-threshold",
        "--threshold",
        type=float,
        help="The confidence threshold for good R-peaks.",
    )
    parser.add_argument(
        "-checkpoint_path",
        "--checkpoint_path",
        type=str,
        help=f"Path to the model checkpoint. You are using {checkpoint_path}.",
    )
    parser.add_argument(
        "-condition",
        "--condition",
        choices=DETECTION_PRESETS.keys(),
        help="Peak-finding preset for mouse ECG condition.",
    )
    return parser.parse_args()


def get_sampling_rate(mat):
    time = mat.get("t_ECG")
    if time is None:
        return DEFAULT_FS

    time = time.flatten()
    duration = time[-1] - time[0]
    if duration <= 0:
        return DEFAULT_FS
    return round((time.size - 1) / duration)


def filter_for_peak_detection(ecg, fs, highpass_hz, lowpass_hz):
    lowpass_hz = min(lowpass_hz, 0.45 * fs)
    sos = butter(
        3,
        [highpass_hz, lowpass_hz],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return sosfiltfilt(sos, ecg)


def find_candidate_peaks(ecg, fs, condition):
    preset = DETECTION_PRESETS[condition]
    peak_signal = ecg
    if preset["use_filter"]:
        peak_signal = filter_for_peak_detection(
            ecg,
            fs,
            preset["highpass_hz"],
            preset["lowpass_hz"],
        )
    distance = max(1, round(preset["distance_ms"] * fs / 1000))
    prominence = preset["prominence"]
    if prominence is None:
        median = np.median(peak_signal)
        noise = 1.4826 * np.median(np.abs(peak_signal - median))
        prominence = preset["prominence_mad"] * noise
    return find_peaks(
        peak_signal,
        height=0,
        distance=distance,
        prominence=(prominence, None),
    )


def main(mat_file=None, threshold=None, checkpoint_path=None, condition=None):
    config = {}
    params_updated = False
    if os.path.isfile(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    default_checkpoint_path = config.get(
        "checkpoint_path",
        os.path.join(PROJECT_DIR, "checkpoints", "r_peak_classifier_out_8_dt_0.5.pth"),
    )
    default_threshold = config.get("decision_threshold", 0.5)
    default_condition = config.get("condition", "wake")

    if mat_file is None:
        args = parse_args(default_checkpoint_path)
        mat_file = args.mat_file
        checkpoint_path = args.checkpoint_path
        threshold = args.threshold
        condition = args.condition

    if checkpoint_path is None:
        checkpoint_path = default_checkpoint_path
    else:
        config["checkpoint_path"] = checkpoint_path
        params_updated = True

    if threshold is None:
        threshold = default_threshold
    else:
        config["decision_threshold"] = threshold
        params_updated = True

    if condition is None:
        condition = default_condition
    else:
        config["condition"] = condition
        params_updated = True
    if condition not in DETECTION_PRESETS:
        raise ValueError(
            f"Unknown condition '{condition}'. Choose from {list(DETECTION_PRESETS)}."
        )

    save_path = mat_file
    if params_updated:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)

    # Load .mat file
    mat = loadmat(mat_file)
    ecg = mat["ECG"].flatten()
    fs = get_sampling_rate(mat)

    # Process ECG data
    print("Finding peaks...")
    r_peak_inds, _ = find_candidate_peaks(ecg, fs, condition)
    r_peak_inds, r_peak_conf = validate_r_peaks(
        ecg, r_peak_inds, checkpoint_path, decision_threshold=threshold
    )
    good_peak_inds = r_peak_inds[r_peak_conf > threshold]

    # Save outputs
    mat["detected_r_peaks"] = r_peak_inds
    mat["r_peak_confidence"] = r_peak_conf
    mat["good_r_peaks"] = good_peak_inds

    savemat(save_path, mat)
    print(f"R-peak indices saved to {save_path}.")


if __name__ == "__main__":
    # Edit these values when running this file directly in Spyder.
    MAT_FILE = os.path.join(PROJECT_DIR, "data", "F26C_07112023_signals.mat")
    THRESHOLD = None
    CHECKPOINT_PATH = None
    CONDITION = None

    if len(sys.argv) > 1:
        main()
    else:
        main(
            mat_file=MAT_FILE,
            threshold=THRESHOLD,
            checkpoint_path=CHECKPOINT_PATH,
            condition=CONDITION,
        )
