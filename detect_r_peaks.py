# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 20:18:01 2024

@author: yzhao
"""

import os

# import wfdb
# import neurokit2
# import wfdb.processing
# import sleepecg

import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import find_peaks


DATA_PATH = "C:/Users/yzhao/matlab_projects/ECG_data/"
SAVE_PATH = "./data/"

#mat_file = "F32CSS3h_20122023_signals.mat" # good data
#mat_file = "F26C_07112023_signals.mat"  # medium data
mat_file = "M38A_23112023_signals.mat" # challenging data
save_file = os.path.join(SAVE_PATH, mat_file)

mat = loadmat(os.path.join(DATA_PATH, mat_file))
ecg = mat["ECG"].flatten()
time = mat["t_ECG"].flatten()
fs = round(time.size / time[-1])

# %%
# t_start = 110
# t_end = 115
# ecg_seg = ecg[t_start * fs : t_end * fs]

# r_peak_inds = sleepecg.detect_heartbeats(ecg, fs=fs)
# _, results = neurokit2.ecg_peaks(ecg_seg, sampling_rate=fs, method="pantompkins1985")
# r_peaks = results["ECG_R_Peaks"]
# r_peaks= wfdb.processing.find_local_peaks(ecg, radius=80)
r_peak_inds, _ = find_peaks(ecg, height=0, distance=50, prominence=(0.5, None))

mat["detected_r_peaks"] = r_peak_inds

savemat(save_file, mat)
