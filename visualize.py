# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 20:18:01 2024

@author: yzhao
"""

import argparse
import os
import sys
import webbrowser

import numpy as np
from scipy.io import loadmat

import plotly.io as io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_figure(mat_file, time, ecg, r_peak_inds, r_peak_conf):
    fig = FigureResampler(
        make_subplots(
            rows=1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Raw ECG",),
        ),
        default_n_shown_samples=8000,
        default_downsampler=MinMaxLTTB(parallel=True),
    )

    # add ECG trace
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

    # add r peaks trace
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

    fig.update_layout(
        autosize=True,
        margin=dict(t=20, l=10, r=10, b=20),
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
    return fig


def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize R-peaks after labeling R-peaks in ECG data."
    )
    parser.add_argument(
        "mat_file",
        type=str,
        help="Path to the .mat file containing the ECG data and the R-peak labels.",
    )
    return parser.parse_args()


def main(mat_file=None):
    if mat_file is None:
        args = parse_args()
        mat_file = args.mat_file

    mat = loadmat(mat_file)
    time, ecg, r_peak_inds, r_peak_conf = (
        mat["t_ECG"],
        mat.get("ECG"),
        mat.get("detected_r_peaks"),
        mat.get("r_peak_confidence"),
    )
    if ecg is None or r_peak_inds is None or r_peak_conf is None:
        print(
            "The mat file does not yet contain R peak labels. Have you run detect_r_peaks on this file yet?"
        )
        return

    time = time.flatten()
    ecg = ecg.flatten()
    r_peak_inds = r_peak_inds.flatten()
    r_peak_conf = r_peak_conf.flatten()
    fig = make_figure(mat_file, time, ecg, r_peak_inds, r_peak_conf)
    fig.show_dash(mode="external", config={"scrollZoom": True}) # need to specify mode per update in v0.11.0
    # see https://github.com/predict-idlab/plotly-resampler/pull/360
    open_browser()


if __name__ == "__main__":
    # Edit this value when running this file directly in Spyder.
    #MAT_FILE = os.path.join(PROJECT_DIR, "data", "F153-Dex40-06022025_signals.mat")
    MAT_FILE = os.path.join(PROJECT_DIR, "data", "M166-dex40-04032025_signals.mat")
    io.renderers.default = "browser"
    if len(sys.argv) > 1:
        main()
    else:
        main(mat_file=MAT_FILE)
