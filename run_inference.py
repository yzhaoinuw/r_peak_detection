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
from models import R_Peak_Classifier


def validate_r_peaks(data: np.array, model_path, batch_size=64, decision_threshold=0.5):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    data = (data - mean) / std

    # Create Dataset objects
    test_dataset = R_Peak_Dataset(data)

    # Create DataLoader objects
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = R_Peak_Classifier()
    model.to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    r_peak_conf = []
    with torch.no_grad():
        for _, segments in enumerate(tqdm(test_loader)):
            segments = segments.to(device)
            outputs = model(segments)
            r_peak_conf.extend(outputs.tolist())

    r_peak_conf = np.array(r_peak_conf)
    # predictions = (r_peak_conf > decision_threshold)
    return r_peak_conf


# %%
if __name__ == "__main__":
    CHECKPOINT_PATH = "./checkpoints/"
    DATA_PATH = "./data/"
    model_path = os.path.join(CHECKPOINT_PATH, "r_peak_classifier_out_8_dt_0.5.pth")
    data_file = os.path.join(DATA_PATH, "r_peak_test_data.npy")
    data = np.load(data_file, allow_pickle=True)
    r_peak_conf = validate_r_peaks(data, model_path)
