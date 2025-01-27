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

import plotly.io as io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB

from run_inference import validate_r_peaks


io.renderers.default = "browser"
DATA_PATH = "./data/"
CHECKPOINT_PATH = "./checkpoints/"

mat_file = "M22BC_signals.mat"  # good data
# mat_file = "F26C_07112023_signals.mat"  # medium data
# mat_file = "M38A_23112023_signals.mat" # challenging data

mat = loadmat(os.path.join(DATA_PATH, mat_file))
ecg = mat["ECG"].flatten()
time = mat["t_ECG"].flatten()

# %%
r_peak_inds = mat.get("detected_r_peaks")
if r_peak_inds is not None:
    model_path = os.path.join(CHECKPOINT_PATH, "r_peak_classifier_out_8_dt_0.5.pth")
    r_peak_inds = np.transpose(r_peak_inds)
    left_neighborhood_array = np.arange(-15, 1)
    right_neighborhood_array = np.arange(1, 17)
    # Use broadcasting to add the range_array to each start index
    left_segment_indices = r_peak_inds + left_neighborhood_array
    right_segment_indices = r_peak_inds + right_neighborhood_array

    r_peak_segment_indices = np.concatenate(
        (left_segment_indices, right_segment_indices), axis=1
    )
    r_peak_segments = ecg[r_peak_segment_indices]

    r_peak_conf = validate_r_peaks(r_peak_segments, model_path)

r_peak_inds = r_peak_inds.flatten()

fig = FigureResampler(
    make_subplots(
        rows=1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "Raw ECG",
            # "Processed ECG",
        ),
        # row_heights=[0.5, 0.5],
    ),
    default_n_shown_samples=8000,
    default_downsampler=MinMaxLTTB(parallel=True),
)

# %% add ECG trace
fig.add_trace(
    go.Scattergl(
        name="ECG",
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

# %% add r peaks trace
fig.add_trace(
    go.Scattergl(
        name="Peak Confidence",
        marker=dict(
            symbol="circle-open",
            size=5,
            color=r_peak_conf,
            colorscale=[[0.0, "red"], [1.0, "green"]],
            cauto=False,
            showscale=False,
            line=dict(
                width=2,
            ),
        ),
        showlegend=False,
        mode="markers",
        customdata=r_peak_conf,
        hovertemplate="<b>y</b>: %{y}"
        + "<br><b>Confidence</b>: %{customdata:.2f}<extra></extra>",
    ),
    hf_x=time[r_peak_inds],
    hf_y=ecg[r_peak_inds],
    row=1,
    col=1,
)
"""
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
    row=2,
    col=1,
)
"""

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
    xaxis1=dict(tickformat="digits"),
    modebar_remove=["lasso2d", "zoom", "autoScale"],
    dragmode="pan",
    clickmode="event",
)

fig.update_traces(xaxis="x1")  # gives crosshair across all subplots
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
    row=1,
    col=1,
)

fig.show_dash(config={"scrollZoom": True})
