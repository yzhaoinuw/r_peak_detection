# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 19:05:54 2024

@author: yzhao
"""

import os

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from dataset import R_Peak_Dataset
from models import R_Peak_Classifier, R_Peak_Classifier_Large


def validate_r_peaks(
    ecg: np.array,
    r_peak_inds: np.array,
    checkpoint_path,
    batch_size=64,
    decision_threshold=0.5,
):
    end = len(ecg)
    print("Validating peaks...")
    if "large" in checkpoint_path:
        model = R_Peak_Classifier_Large()
        N = 64
        r_peak_inds = r_peak_inds[
            (N // 2 <= r_peak_inds) & (r_peak_inds <= end - N // 2)
        ]
        r_peak_inds = np.expand_dims(r_peak_inds, axis=1)
        left_neighborhood_array = np.arange(-N // 2, 1)
        right_neighborhood_array = np.arange(1, N // 2)

    else:
        model = R_Peak_Classifier()
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

    # standardization
    mean = np.mean(r_peak_segments, axis=1, keepdims=True)
    std = np.std(r_peak_segments, axis=1, keepdims=True)
    r_peak_segments = (r_peak_segments - mean) / std

    # Create Dataset objects
    test_dataset = R_Peak_Dataset(r_peak_segments)

    # Create DataLoader objects
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()

    r_peak_conf = []
    with torch.no_grad():
        for _, segments in enumerate(tqdm(test_loader)):
            segments = segments.to(device)
            outputs = model(segments)
            r_peak_conf.extend(outputs.tolist())

    r_peak_conf = np.array(r_peak_conf)
    return r_peak_inds, r_peak_conf


# %%
if __name__ == "__main__":
    from scipy.signal import find_peaks
    from scipy.io import loadmat

    CHECKPOINT_PATH = "./checkpoints/"
    DATA_PATH = "./data/"
    checkpoint_path = os.path.join(
        CHECKPOINT_PATH, "r_peak_classifier_out_8_dt_0.5.pth"
    )
    checkpoint_path = os.path.join(
        CHECKPOINT_PATH, "r_peak_classifier_large_out1_32_out2_64_bs_32_dt_0.5.pth"
    )

    mat_file = "F32CSS3h_20122023_signals.mat"  # good data
    # mat_file = "F26C_07112023_signals.mat"  # medium data
    # mat_file = "M38A_23112023_signals.mat" # challenging data

    mat_file = os.path.join(DATA_PATH, mat_file)
    mat = loadmat(mat_file)
    ecg = mat["ECG"].flatten()

    # Process ECG data
    print("Finding peaks...")
    r_peak_inds, _ = find_peaks(ecg, height=0, distance=50, prominence=(0.5, None))
    r_peak_inds, r_peak_conf = validate_r_peaks(ecg, r_peak_inds, checkpoint_path)
