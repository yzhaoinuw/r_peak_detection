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
from scipy.io import loadmat

# from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt, sosfiltfilt

import plotly.io as io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB

io.renderers.default = "browser"
data_path = "C:/Users/yzhao/matlab_projects/ECG_data/"

mat_file = "F32CSS3h_20122023_signals.mat"  # good data
# mat_file = "F26C_07112023_signals.mat" # medium data
# mat_file = "M38A_23112023_signals.mat" # challenging data


# https://stackoverflow.com/a/48677312/9075227
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype="band", output="sos")
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


# https://stackoverflow.com/a/25192640/9075227
def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype="low", analog=False)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


mat = loadmat(os.path.join(data_path, mat_file))
ecg = mat["ECG"].flatten()
time = mat["t_ECG"].flatten()

# Sample rate and desired cutoff frequencies (in Hz).
fs = round(time.size / time[-1])

lowcut = 2
highcut = 150

# ecg_filtered = butter_bandpass_filter(ecg, lowcut, highcut, fs, order=6)
ecg_filtered = butter_lowpass_filter(data=ecg, cutoff=highcut, fs=fs, order=6)

ecg_filtered = ecg_filtered.copy()
# %%

# t_start = 110
# t_end = 115
# ecg_seg = ecg[t_start * fs : t_end * fs]

# r_peak_inds = sleepecg.detect_heartbeats(ecg, fs=fs)
# _, results = neurokit2.ecg_peaks(ecg_seg, sampling_rate=fs, method="pantompkins1985")
# r_peaks = results["ECG_R_Peaks"]
# r_peaks= wfdb.processing.find_local_peaks(ecg, radius=80)
# r_peak_inds, _ = find_peaks(ecg, height=0, distance=50, prominence=(0.5, None))

fig = FigureResampler(
    make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "Raw ECG",
            "Processed ECG",
        ),
        # row_heights=[0.5, 0.5],
    ),
    default_n_shown_samples=8000,
    default_downsampler=MinMaxLTTB(parallel=True),
)

# %% add ECG trace
fig.add_trace(
    go.Scattergl(
        name=" ",
        line=dict(width=1),
        marker=dict(size=2, color="blue"),
        showlegend=False,
        mode="lines+markers",
    ),
    hf_x=time,
    hf_y=ecg,
    row=1,
    col=1,
)


fig.add_trace(
    go.Scattergl(
        name=" ",
        line=dict(width=1),
        marker=dict(size=2, color="blue"),
        showlegend=False,
        mode="lines+markers",
    ),
    hf_x=time,
    hf_y=ecg_filtered,
    row=2,
    col=1,
)

fig.update_layout(
    autosize=True,
    margin=dict(t=0, l=10, r=10, b=0),
    # height=800,
    hovermode="x unified",  # gives crosshair in one subplot
    hoverlabel=dict(bgcolor="rgba(255, 255, 255, 0.6)"),
    title=dict(
        text=mat_file,
        font=dict(size=16),
        xanchor="left",
        x=0,
        yanchor="top",
        yref="container",
    ),
    xaxis2=dict(tickformat="digits"),
    modebar_remove=["lasso2d", "zoom", "autoScale"],
    dragmode="pan",
    clickmode="event",
)

fig.update_traces(xaxis="x2")  # gives crosshair across all subplots
fig.update_xaxes(
    range=[0, np.ceil(time[-1])],
    title_text="<b>Time (s)</b>",
    minor=dict(
        tick0=0,
        dtick=3600,
        tickcolor="black",
        ticks="outside",
        ticklen=5,
        tickwidth=2,
    ),
    row=2,
    col=1,
)

fig.show_dash(config={"scrollZoom": True})
