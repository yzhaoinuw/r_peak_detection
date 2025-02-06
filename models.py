# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 00:15:55 2024

@author: yzhao
"""

import torch.nn as nn
import torch.nn.functional as F


class R_Peak_Classifier(nn.Module):
    def __init__(self, out_channels=8, hidden_size=16, dropout=0.5):
        super(R_Peak_Classifier, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=out_channels, kernel_size=5, padding=2
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.fc1 = nn.Linear(out_channels * 16, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))  # Output shape: [batch_size, 8, 20]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 160]
        x = self.relu2(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x.flatten()


class R_Peak_Classifier_Large(nn.Module):
    def __init__(self, out1=32, out2=64, hidden_size=256, dropout=0.5):
        super(R_Peak_Classifier_Large, self).__init__()

        # First convolution: using a kernel size of 5 to capture the expected R-peak shape.
        # Padding of 2 keeps the output length the same as the input (i.e. "same" padding).
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=out1, kernel_size=5, stride=1, padding=2
        )

        # Max pooling to reduce the sequence length by half (from 64 to 32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Second convolution: further feature extraction with a smaller kernel.
        # Padding of 1 ensures the output length remains the same as the input.
        self.conv2 = nn.Conv1d(
            in_channels=out1, out_channels=out2, kernel_size=3, stride=1, padding=1
        )

        # Additional max pooling: further reduces the sequence length from 32 to 16.
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # After the two pooling layers, the sequence length is reduced from 64 -> 32 -> 16.
        # The output of conv2 has 32 channels, so the flattened size is 32 * 16 = 512.
        self.fc1 = nn.Linear(out2 * 16, hidden_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Final output layer for binary classification (single output neuron)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x should have shape: (batch_size, 1, 64)
        x = F.relu(self.conv1(x))  # Shape: (batch_size, 16, 64)
        x = self.pool1(x)  # Shape: (batch_size, 16, 32)
        x = F.relu(self.conv2(x))  # Shape: (batch_size, 32, 32)
        x = self.pool2(x)  # Shape: (batch_size, 32, 16)

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 32*16)
        x = F.relu(self.fc1(x))  # Shape: (batch_size, 64)
        x = self.dropout(x)
        x = F.sigmoid(self.fc2(x))  # Shape: (batch_size, 1)
        return x.flatten()
