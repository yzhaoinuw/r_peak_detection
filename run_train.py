# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 00:22:14 2024

@author: yzhao
"""

import os
import logging

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import R_Peak_Dataset
from models import R_Peak_Classifier

DATA_PATH = "C:/Users/yzhao/matlab_projects/ECG_data/"
SAVE_PATH = "./data/"
CHECKPOINT_PATH = "./checkpoints/"

data_file = os.path.join(SAVE_PATH, "r_peak_data.npy")
label_file = os.path.join(SAVE_PATH, "r_peak_labels.npy")
data = np.load(data_file, allow_pickle=True)
labels = np.load(label_file, allow_pickle=True)
labels = labels.astype(int)

mean = np.mean(data, axis=1, keepdims=True)
std = np.std(data, axis=1, keepdims=True)
data = (data - mean) / std

# %%
train_data, val_data, train_labels, val_labels = train_test_split(
    data, labels, stratify=labels, test_size=0.4
)
# Create Dataset objects
train_dataset = R_Peak_Dataset(train_data, train_labels, noise_prob=0.5)
val_dataset = R_Peak_Dataset(val_data, val_labels)

# Hyperparameters
batch_size = 16

# Create DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

learning_rate = 0.001
num_epochs = 30
decision_threshold = 0.5
out_channels = 8

model_name = f"r_peak_classifier_out_{out_channels}_dt_{decision_threshold}"
logger = logging.getLogger("logger")
logging.getLogger().setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s: %(message)s")
file_handler = logging.FileHandler(
    filename=os.path.join(CHECKPOINT_PATH, model_name + ".log"), mode="w"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.cuda.empty_cache()
model = R_Peak_Classifier(out_channels=out_channels, hidden_size=16, dropout=0.5)
model.to(device)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_accuracy = 0.95
for epoch in range(1, num_epochs + 1):
    # Validation
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    with torch.no_grad():
        for segments, labels in val_loader:
            segments, labels = segments.to(device), labels.to(device)
            outputs = model(segments)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * segments.size(0)
            predictions = (outputs >= decision_threshold).float()
            correct_predictions += (predictions == labels).sum().item()
    val_loss /= len(val_dataset)
    val_accuracy = correct_predictions / len(val_dataset)

    model.train()
    train_loss = 0.0
    for segments, labels in train_loader:
        optimizer.zero_grad()
        segments, labels = segments.to(device), labels.to(device)
        outputs = model(segments)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * segments.size(0)
    train_loss /= len(train_dataset)

    logging.getLogger("logger").debug(f"Epoch {epoch}/{num_epochs}")
    logging.getLogger("logger").debug(f"Train Loss: {train_loss:.4f}")
    logging.getLogger("logger").debug(f"Val Loss: {val_loss:.4f}")
    logging.getLogger("logger").debug(f"Val Accuracy: {val_accuracy:.4f}")

    # Check if validation accuracy exceeds 95% and increases
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        checkpoint_path = os.path.join(CHECKPOINT_PATH, model_name + ".pth")
        torch.save(model.state_dict(), checkpoint_path)
        logging.getLogger("logger").debug(f"Model saved to {checkpoint_path}")

handlers = logger.handlers
for handler in handlers:
    logger.removeHandler(handler)
    handler.close()
