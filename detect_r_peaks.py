# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 20:18:01 2024

@author: yzhao
"""

import os
import json
import argparse

import numpy as np
from scipy.signal import find_peaks
from scipy.io import loadmat, savemat

from run_inference import validate_r_peaks


CONFIG_FILE = "config.json"

def main():
    config = {}
    params_updated = False
    if os.path.isfile(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    checkpoint_path = config.get("checkpoint_path", "./checkpoints/r_peak_classifier_out_8_dt_0.5.pth")
    threshold = config.get("decision_threshold", 0.5)
        
    parser = argparse.ArgumentParser(description="Extract R-peaks from ECG data.")
    parser.add_argument(
        "mat_file", type=str, help="Path to the .mat file containing the ECG data."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to the model checkpoint.",
    )
    parser.add_argument("-threshold", "--threshold", type=float, help="The confidence threshold for good R-peaks.")

    args = parser.parse_args()
    mat_file = args.mat_file
    save_path = mat_file
    if args.checkpoint_path is not None:
        checkpoint_path = args.checkpoint_path
        config["checkpoint_path"] = checkpoint_path
        params_updated = True
    if args.threshold is not None:
        threshold = args.threshold
        config["decision_threshold"] = threshold
        params_updated = True
    if params_updated:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
    

    # Load .mat file
    mat = loadmat(mat_file)
    ecg = mat["ECG"].flatten()

    # Process ECG data
    end = len(ecg)
    print("Finding peaks...")
    r_peak_inds, _ = find_peaks(ecg, height=0, distance=50, prominence=(0.5, None))

    print("Validating peaks...")
    r_peak_inds = r_peak_inds[(15 <= r_peak_inds) & (r_peak_inds <= end - 16)]
    r_peak_inds = np.expand_dims(r_peak_inds, axis=1)
    left_neighborhood_array = np.arange(-15, 1)
    right_neighborhood_array = np.arange(1, 17)

    # Use broadcasting to add the range_array to each start index
    left_segment_indices = r_peak_inds + left_neighborhood_array
    right_segment_indices = r_peak_inds + right_neighborhood_array
    r_peak_segment_indices = np.concatenate(
        (left_segment_indices, right_segment_indices), axis=1
    )
    r_peak_segments = ecg[r_peak_segment_indices]

    # Validate R-peaks
    r_peak_inds = r_peak_inds.flatten()
    r_peak_conf = validate_r_peaks(r_peak_segments, checkpoint_path, decision_threshold=threshold)
    good_peak_inds = r_peak_inds[r_peak_conf > 0.5]

    # Save outputs
    mat["detected_r_peaks"] = r_peak_inds
    mat["r_peak_confidence"] = r_peak_conf
    mat["good_r_peaks"] = good_peak_inds

    savemat(save_path, mat)
    print(f"R-peak indices saved to {save_path}.")


if __name__ == "__main__":
    main()
