# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:02:38 2024

@author: yzhao
"""

import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks

import matplotlib.pyplot as plt


DATA_PATH = "C:/Users/yzhao/matlab_projects/ECG_data/"
SAVE_PATH = "./data/"

# mat_file = "F32CSS3h_20122023_signals.mat" # good data
mat_file = "F26C_07112023_signals.mat"  # medium data
# mat_file = "M38A_23112023_signals.mat" # challenging data

mat_name = Path(mat_file).stem
data_file = os.path.join(SAVE_PATH, "r_peak_test_data.npy")
label_file = os.path.join(SAVE_PATH, "r_peak_test_labels.npy")
used_index_file = os.path.join(SAVE_PATH, mat_name + "_used_indices.npy")

mat = loadmat(os.path.join(DATA_PATH, mat_file))
ecg = mat["ECG"].flatten()
# time = mat["t_ECG"].flatten()
# fs = round(time.size / time[-1])

# %%
end = len(ecg)
r_peak_inds, _ = find_peaks(ecg, height=0, distance=50, prominence=(0.5, None))
r_peak_inds = r_peak_inds[(15 <= r_peak_inds) & (r_peak_inds <= end - 16)]

r_peak_inds = r_peak_inds[:, np.newaxis]
left_neighborhood_array = np.arange(-15, 1)
right_neighborhood_array = np.arange(1, 17)
# Use broadcasting to add the range_array to each start index
left_segment_indices = r_peak_inds + left_neighborhood_array
right_segment_indices = r_peak_inds + right_neighborhood_array

r_peak_segment_indices = np.concatenate(
    (left_segment_indices, right_segment_indices), axis=1
)
r_peak_segments = ecg[r_peak_segment_indices]
n_seg = len(r_peak_segments)
# %%
if Path(data_file).is_file():
    segments = np.load(data_file, allow_pickle=True)
    segments = segments.tolist()
else:
    segments = []

if Path(label_file).is_file():
    labels = np.load(label_file, allow_pickle=True)
    labels = labels.astype(int)
    labels = labels.tolist()
else:
    labels = []

if Path(used_index_file).is_file():
    used_indices_list = np.load(used_index_file, allow_pickle=True)
    used_indices_list = used_indices_list.tolist()
else:
    used_indices_list = []

ind_array = np.arange(n_seg)
if len(used_indices_list) != 0:
    ind_array = np.delete(ind_array, used_indices_list)
rand_indices = np.random.permutation(ind_array)

for ind in rand_indices:
    r_peak_segment = r_peak_segments[ind]
    plt.plot(r_peak_segment, ".-")
    plt.show(block=False)
    label = input("Good Peak (1) / Bad Peak (0)\n")
    if label == "q":
        break
    if label == "save":
        np.save(data_file, np.array(segments), allow_pickle=True)
        np.save(label_file, np.array(labels, dtype=np.int8), allow_pickle=True)
        np.save(used_index_file, np.array(used_indices_list), allow_pickle=True)
        break
    while not label.isnumeric() or int(label) not in [1, 0]:
        label = input("Good Peak (1) / Bad Peak (0)\n")
    used_indices_list.append(ind)
    segments.append(r_peak_segment)
    labels.append(int(label))
