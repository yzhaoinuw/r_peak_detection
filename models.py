# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 00:15:55 2024

@author: yzhao
"""

import torch.nn as nn


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
        return x.squeeze()
