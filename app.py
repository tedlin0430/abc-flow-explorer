from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update
from scipy.integrate import solve_ivp
from scipy.io import loadmat

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DEFAULT_MAP_NAME = "frontspeed_N400_T100_eps_1.mat"
MAP_PATH = Path(os.environ.get("FRONT_SPEED_FILE", DATA_DIR / DEFAULT_MAP_NAME))


def _scalar(value, default=np.nan):
    try:
        arr = np.asarray(value).squeeze()
        return float(arr.item() if arr.size == 1 else arr.flat[0])
    except Exception:
        return float(default)


def load_frontspeed(path: Path):
    """Load MATLAB v7/v7.2 or v7.3 HDF5 front-speed data."""
    if not path.exists():
        raise FileNotFoundError(
            f"Front-speed file not found: {path}\n"
            f"Put {DEFAULT_MAP_NAME} inside data/, or set FRONT_SPEED_FILE."
        )

    try:
        data = loadmat(path)
        speed = np.asarray(data["speed_abc_dir"], dtype=float)
        y = np.asarray(data["y"], dtype=float).squeeze()
        z = np.asarray(data["z"], dtype=float).squeeze()
        epsi = _scalar(data.get("epsi"))
        T = _scalar(data.get("T"))
        N = int(round(_scalar(data.get("N"), speed.shape[0])))
        x0 = _scalar(data.get("x0"), 0.0)
        source_format = "MATLAB MAT"
    except NotImplementedError:
        import h5py
        with h5py.File(path, "r") as f:
            missing = [k for k in ("speed_abc_dir", "y", "z") if k not in f]
            if missing:
                raise KeyError(f"Missing variable(s): {', '.join(missing)}")
            # MATLAB v7.3/HDF5 exposes MATLAB matrix dimensions reversed
            # relative to scipy.loadmat, so transpose the 2-D map here.
            speed = np.asarray(f["speed_abc_dir"], dtype=float).T
            y = np.asarray(f["y"], dtype=float).squeeze()
            z = np.asarray(f["z"], dtype=float).squeeze()

            def h5scalar(name, default=np.nan):
                if name not in f:
                    return float(default)
                return _scalar(np.asarray(f[name]), default)

            epsi = h5scalar("epsi")
            T = h5scalar("T")
            N = int(round(h5scalar("N", speed.shape[0])))
            x0 = h5scalar("x0", 0.0)
            source_format = "MATLAB v7.3 / HDF5"

    if speed.shape != (z.size, y.size):
        raise ValueError(
            f"Map shape mismatch: speed={speed.shape}, len(y)={y.size}, len(z)={z.size}"
        )

    return {
        "speed": speed,
        "y": y,
        "z": z,
        "epsi": epsi,
        "T": T,
        "N": N,
        "x0": x0,
        "source_format": source_format,
    }


def abc_rhs(_t, xyz, epsilon):
    # Same convention as the MATLAB GUI:
    # dx/dt = epsilon*sin(z) + cos(y)
    # dy/dt = sin(x) + epsilon*cos(z)
    # dz/dt = sin(y) + cos(x)
    x, y, z = xyz
    return (
        epsilon * np.sin(z) + np.cos(y),
        np.sin(x) + epsilon * np.cos(z),
        np.sin(y) + np.cos(x),
    )


def compute_trajectory(y0, z0, epsilon, T, x0=0.0):
    n_plot = int(np.clip(max(700, 40 * T), 700, 8000))
    t_eval = np.linspace(0.0, T, n_plot)
    sol = solve_ivp(
        lambda t, X: abc_rhs(t, X, epsilon),
        (0.0, T),
        (x0, y0, z0),
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
        max_step=0.5,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y.T


def make_empty_trajectory_figure():
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        title="3D trajectory",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
        margin=dict(l=0, r=0, t=48, b=0),
        annotations=[dict(
            text="Click the front-speed map, enter ε and T, then press Update.",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
            font=dict(size=15),
        )],
    )
    return fig


MAP_ERROR = None
MAP = None
try:
    MAP = load_frontspeed(MAP_PATH)
except Exception as exc:
    MAP_ERROR = str(exc)


def make_frontspeed_figure():
    if MAP is None:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            title="Front-speed map unavailable",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            annotations=[dict(
                text=MAP_ERROR.replace("\n", "<br>"),
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, align="left",
            )],
        )
        return fig

    fig = go.Figure(go.Heatmap(
        x=MAP["y"], y=MAP["z"], z=MAP["speed"],
        colorscale="Turbo",
        colorbar=dict(title="Front speed"),
        hovertemplate=(
            "y₀=%{x:.6f}<br>z₀=%{y:.6f}<br>front speed=%{z:.6f}<extra></extra>"
        ),
    ))

    title = "Front-speed map"
    if np.isfinite(MAP["epsi"]) and np.isfinite(MAP["T"]):
        title += f" — ε={MAP['epsi']:g}, T={MAP['T']:g}"

    fig.update_layout(
        template="plotly_white",
        title=title,
        clickmode="event+select",
        xaxis=dict(
            title="Initial y₀",
            range=[float(np.min(MAP["y"])), float(np.max(MAP["y"]))],
            autorange=False,
        ),
        yaxis=dict(
            title="Initial z₀",
            range=[float(np.min(MAP["z"])), float(np.max(MAP["z"]))],
            autorange=False,
        ),
        margin=dict(l=58, r=24, t=55, b=55),
    )
    return fig


app = Dash(__name__, title="ABC Flow Orbit Explorer")
server = app.server

app.layout = html.Div(className="page-shell", children=[
    html.Div(className="header", children=[
        html.H1("ABC Flow Orbit Explorer"),
        html.P("Click the precomputed front-speed map to select (y₀,z₀), then integrate a 3D trajectory."),
    ]),
    html.Div(className="main-grid", children=[
        html.Div(className="card map-card", children=[
            dcc.Graph(id="frontspeed-map", figure=make_frontspeed_figure(),
                      config={"displaylogo": False}, style={"height": "650px"})
        ]),
        html.Div(className="card trajectory-card", children=[
            dcc.Graph(id="trajectory-graph", figure=make_empty_trajectory_figure(),
                      config={"displaylogo": False}, style={"height": "650px"})
        ]),
        html.Div(className="card controls-card", children=[
            html.H3("Parameters"),
            html.Label("epsilon (ε)"),
            dcc.Input(id="epsilon-input", type="text", value="", placeholder="e.g. 1", debounce=True),
            html.Label("y₀"),
            dcc.Input(id="y0-input", type="text", value="", placeholder="click map or type value", debounce=True),
            html.Label("z₀"),
            dcc.Input(id="z0-input", type="text", value="", placeholder="click map or type value", debounce=True),
            html.Label("T"),
            dcc.Input(id="T-input", type="text", value="", placeholder="e.g. 100", debounce=True),
            html.Button("Update", id="update-button", n_clicks=0),
            html.Div(id="status-box", className="status-box"),
            html.Hr(),
            html.Div(className="meta-box", children=[
                html.Strong("Background map"),
                html.Div(
                    "File not loaded" if MAP is None else (
                        f"N={MAP['N']}, ε={MAP['epsi']:g}, T={MAP['T']:g}"
                        if np.isfinite(MAP["epsi"]) and np.isfinite(MAP["T"])
                        else f"N={MAP['N']}"
                    )
                ),
                html.Div("" if MAP is None else f"Format: {MAP['source_format']}"),
            ]),
        ]),
    ]),
    html.Div(className="footer-note", children=[
        "This web app is for interactive numerical exploration. Use the MATLAB/INTLAB workflow for rigorous validation."
    ]),
])


@app.callback(
    Output("y0-input", "value"),
    Output("z0-input", "value"),
    Input("frontspeed-map", "clickData"),
    prevent_initial_call=True,
)
def pick_initial_point(click_data):
    if not click_data or not click_data.get("points"):
        return no_update, no_update
    point = click_data["points"][0]
    return f"{float(point['x']):.8f}", f"{float(point['y']):.8f}"


@app.callback(
    Output("frontspeed-map", "figure"),
    Input("y0-input", "value"),
    Input("z0-input", "value"),
)
def show_selected_point(y0, z0):
    fig = make_frontspeed_figure()
    if MAP is None or y0 is None or z0 is None:
        return fig
    try:
        y0 = float(str(y0).strip())
        z0 = float(str(z0).strip())
    except Exception:
        return fig
    fig.add_trace(go.Scatter(
        x=[y0], y=[z0], mode="markers",
        marker=dict(size=13, color="white", line=dict(width=3, color="black")),
        name="selected point",
        hovertemplate="selected<br>y₀=%{x:.6f}<br>z₀=%{y:.6f}<extra></extra>",
    ))
    return fig


@app.callback(
    Output("trajectory-graph", "figure"),
    Output("status-box", "children"),
    Input("update-button", "n_clicks"),
    State("epsilon-input", "value"),
    State("y0-input", "value"),
    State("z0-input", "value"),
    State("T-input", "value"),
    prevent_initial_call=True,
)
def update_trajectory(_n_clicks, epsilon, y0, z0, T):
    if MAP is None:
        return make_empty_trajectory_figure(), html.Span(
            "Front-speed MAT file is not loaded.", className="status-error"
        )

    if any(v is None or str(v).strip() == "" for v in (epsilon, y0, z0, T)):
        return no_update, html.Span("Please enter ε, y₀, z₀, and T.", className="status-error")

    try:
        epsilon, y0, z0, T = map(float, (epsilon, y0, z0, T))
    except Exception:
        return no_update, html.Span("All four parameters must be numerical.", className="status-error")

    if not np.isfinite([epsilon, y0, z0, T]).all():
        return no_update, html.Span("All four parameters must be finite.", className="status-error")
    if T <= 0:
        return no_update, html.Span("T must be positive.", className="status-error")

    ymin, ymax = float(np.min(MAP["y"])), float(np.max(MAP["y"]))
    zmin, zmax = float(np.min(MAP["z"])), float(np.max(MAP["z"]))
    if not (ymin <= y0 <= ymax and zmin <= z0 <= zmax):
        return no_update, html.Span(
            f"For this map, y₀ must be in [{ymin:.6g},{ymax:.6g}] and z₀ in [{zmin:.6g},{zmax:.6g}].",
            className="status-error",
        )

    try:
        _, xyz = compute_trajectory(y0, z0, epsilon, T, x0=MAP["x0"])
    except Exception as exc:
        return no_update, html.Span(f"Integration failed: {exc}", className="status-error")

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines",
        line=dict(width=4), name="trajectory",
        hovertemplate="x=%{x:.6f}<br>y=%{y:.6f}<br>z=%{z:.6f}<extra>trajectory</extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=[xyz[0, 0]], y=[xyz[0, 1]], z=[xyz[0, 2]], mode="markers",
        marker=dict(size=6), name="initial point",
    ))
    fig.add_trace(go.Scatter3d(
        x=[xyz[-1, 0]], y=[xyz[-1, 1]], z=[xyz[-1, 2]], mode="markers",
        marker=dict(size=5), name="final point",
    ))
    fig.update_layout(
        template="plotly_white",
        title=f"ε={epsilon:.6g}, X(0)=({MAP['x0']:.6g}, {y0:.6f}, {z0:.6f}), T={T:.6g}",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=0, r=0, t=70, b=0),
    )

    status = [html.Div(f"Selected point: y₀={y0:.6f}, z₀={z0:.6f}", className="status-ok")]
    mismatch = []
    if np.isfinite(MAP["epsi"]) and not np.isclose(epsilon, MAP["epsi"]):
        mismatch.append(f"ε={MAP['epsi']:g}")
    if np.isfinite(MAP["T"]) and not np.isclose(T, MAP["T"]):
        mismatch.append(f"T={MAP['T']:g}")
    if mismatch:
        status.append(html.Div(
            "Warning: the background front-speed map was computed at " + ", ".join(mismatch)
            + "; the current trajectory uses different parameters.",
            className="status-warning",
        ))

    return fig, status


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8050")), debug=True)
