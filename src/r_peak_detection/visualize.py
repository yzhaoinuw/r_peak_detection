# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 20:18:01 2024

@author: yzhao
"""

import argparse
from pathlib import Path
import sys
import socket
import threading
import time
import webbrowser

import numpy as np
from scipy.io import loadmat

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8050
MAX_PORT_ATTEMPTS = 100


def make_figure(mat_file, time, ecg, r_peak_inds, r_peak_conf):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from plotly_resampler import FigureResampler
        from plotly_resampler.aggregation import MinMaxLTTB
    except ImportError as exc:
        raise RuntimeError(
            "Visualization dependencies are not installed. Install them with "
            "`pip install r_peak_detection[visualization]`."
        ) from exc

    fig = FigureResampler(
        make_subplots(
            rows=1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Raw ECG",),
        ),
        default_n_shown_samples=4096,
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


def is_port_available(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) != 0


def stop_dash_server_in_current_session(host, port):
    try:
        import dash
    except ImportError:
        return False

    server_key = (host, str(port))
    old_server = dash.jupyter_dash._servers.get(server_key)
    if old_server is None:
        return False

    old_server.shutdown()
    del dash.jupyter_dash._servers[server_key]
    return True


def find_available_port(host=DEFAULT_HOST, preferred_port=DEFAULT_PORT):
    final_port = min(65535, preferred_port + MAX_PORT_ATTEMPTS - 1)
    for port in range(preferred_port, final_port + 1):
        if is_port_available(host, port):
            return port
    raise OSError(
        f"No available ports found from {preferred_port} "
        f"to {final_port}."
    )


def resolve_port(host, port):
    if port is not None:
        if port < 1 or port > 65535:
            raise ValueError("port must be between 1 and 65535.")
        if not is_port_available(host, port):
            raise OSError(
                f"Address 'http://{host}:{port}' already in use. "
                "Pass a different --port or omit --port to choose automatically."
            )
        return port
    return find_available_port(host)


def open_browser(url):
    webbrowser.open_new(url)


def open_browser_when_ready(url, host, port, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_port_available(host, port):
            open_browser(url)
            return
        time.sleep(0.2)


def is_address_in_use_error(exc):
    return (
        isinstance(exc, OSError)
        and "already in use" in str(exc)
        and "Try passing a different port" in str(exc)
    )


def run_dash(fig, host, port, url, open_browser_on_start):
    print(f"Starting R-peak visualization at {url}")
    if open_browser_on_start:
        threading.Thread(
            target=open_browser_when_ready,
            args=(url, host, port),
            daemon=True,
        ).start()
    fig.show_dash(
        mode="external",
        config={"scrollZoom": True},
        host=host,
        port=port,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize R-peaks after labeling R-peaks in ECG data."
    )
    parser.add_argument(
        "mat_file",
        type=str,
        help="Path to the .mat file containing the ECG data and the R-peak labels.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Host for the local visualization server. Default: {DEFAULT_HOST}.",
    )
    parser.add_argument(
        "--port",
        type=int,
        help=(
            "Port for the local visualization server. "
            f"If omitted, starts at {DEFAULT_PORT} and chooses the first free port."
        ),
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the server without opening a browser tab.",
    )
    parser.add_argument(
        "--keep-existing-server",
        action="store_true",
        help=(
            "Do not stop an existing Dash server on the selected port from the "
            "current Python session before starting a new one."
        ),
    )
    return parser.parse_args()


def main(
    mat_file=None,
    host=DEFAULT_HOST,
    port=None,
    open_browser_on_start=True,
    stop_existing_server=True,
):
    if mat_file is None:
        args = parse_args()
        mat_file = args.mat_file
        host = args.host
        port = args.port
        open_browser_on_start = not args.no_browser
        stop_existing_server = not args.keep_existing_server

    mat = loadmat(mat_file)
    time, ecg, r_peak_inds, r_peak_conf = (
        mat.get("t_ECG"),
        mat.get("ECG"),
        mat.get("detected_r_peaks"),
        mat.get("r_peak_confidence"),
    )
    if time is None or ecg is None or r_peak_inds is None or r_peak_conf is None:
        print(
            "The mat file needs t_ECG, ECG, detected_r_peaks, and "
            "r_peak_confidence. Have you run detect-r-peaks on this file yet?"
        )
        return

    time = time.flatten()
    ecg = ecg.flatten()
    r_peak_inds = r_peak_inds.flatten()
    r_peak_conf = r_peak_conf.flatten()
    explicit_port = port is not None
    if stop_existing_server:
        server_port = port if explicit_port else DEFAULT_PORT
        if stop_dash_server_in_current_session(host, server_port):
            print(f"Stopped existing Dash server at http://{host}:{server_port}/")

    port = resolve_port(host, port)
    final_port = min(65535, port + MAX_PORT_ATTEMPTS - 1)
    while True:
        if stop_existing_server and stop_dash_server_in_current_session(host, port):
            print(f"Stopped existing Dash server at http://{host}:{port}/")
        fig = make_figure(mat_file, time, ecg, r_peak_inds, r_peak_conf)
        url = f"http://{host}:{port}/"
        try:
            run_dash(fig, host, port, url, open_browser_on_start)
            return
        except OSError as exc:
            if explicit_port or not is_address_in_use_error(exc) or port >= final_port:
                raise
            print(f"Port {port} became unavailable; trying {port + 1}.")
            port = find_available_port(host, port + 1)


if __name__ == "__main__":
    # Edit these values when running this file directly in Spyder.
    PROJECT_DIR = Path(__file__).resolve().parents[2]
    MAT_FILE = PROJECT_DIR / "data" / "M166-dex40-04032025_signals.mat"
    HOST = DEFAULT_HOST
    PORT = None
    OPEN_BROWSER_ON_START = True
    STOP_EXISTING_SERVER = True

    try:
        if len(sys.argv) > 1:
            main()
        else:
            main(
                mat_file=str(MAT_FILE),
                host=HOST,
                port=PORT,
                open_browser_on_start=OPEN_BROWSER_ON_START,
                stop_existing_server=STOP_EXISTING_SERVER,
            )
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
