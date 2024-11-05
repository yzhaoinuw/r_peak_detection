# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 00:12:59 2024

@author: yzhao
"""

import torch
from torch.utils.data import Dataset


class R_Peak_Dataset(Dataset):
    def __init__(self, segments, labels=None, noise_prob=0.0, noise_factor=0.1):
        """
        Args:
            segments (numpy.ndarray or list): Array of ECG segments.
            labels (numpy.ndarray or list): Array of labels (1 for true R peak, 0 for false).
        """
        self.segments = torch.tensor(segments, dtype=torch.float32)
        self.labels = labels
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32)

        self.noise_prob = noise_prob
        self.noise_factor = noise_factor

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        # Add a channel dimension (needed for Conv1D)
        segment = self.segments[idx].unsqueeze(0)
        # Randomly add noise
        if torch.rand(1).item() < self.noise_prob:
            noise = torch.randn_like(segment) * self.noise_factor
            segment = segment + noise

        if self.labels is not None:
            label = self.labels[idx]
            return segment, label
        return segment
