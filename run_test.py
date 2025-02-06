# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:22:51 2024

@author: yzhao
"""

import os

import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset import R_Peak_Dataset
from models import R_Peak_Classifier, R_Peak_Classifier_Large


CHECKPOINT_PATH = "./checkpoints/"
DATA_PATH = "./data/"
model_path = os.path.join(
    CHECKPOINT_PATH, "r_peak_classifier_large_out1_32_out2_64_bs_32_dt_0.5.pth"
)
# %%
data_file = os.path.join(DATA_PATH, "r_peak_test_data.npy")
label_file = os.path.join(DATA_PATH, "r_peak_test_labels.npy")
data = np.load(data_file, allow_pickle=True)
labels = np.load(label_file, allow_pickle=True)
labels = labels.astype(int)

mean = np.mean(data, axis=1, keepdims=True)
std = np.std(data, axis=1, keepdims=True)
data = (data - mean) / std

# Create Dataset objects
test_dataset = R_Peak_Dataset(data)

# Hyperparameters
batch_size = 64
decision_threshold = 0.5

# Create DataLoader objects
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = R_Peak_Classifier_Large(out1=32, out2=64, hidden_size=256)
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

r_peak_conf = []
with torch.no_grad():
    for segments in test_loader:
        segments = segments.to(device)
        outputs = model(segments)
        r_peak_conf.extend(outputs.tolist())

r_peak_conf = np.array(r_peak_conf)
predictions = r_peak_conf > decision_threshold
accuracy = np.sum(predictions == labels) / labels.size
