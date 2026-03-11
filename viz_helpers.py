"""
viz_helpers.py
==============
Reusable Plotly figure builders for the Minkowski Lab.
"""

from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from typing import Optional, Tuple
from scipy.spatial import ConvexHull
from geometry_core import hull_mesh_arrays, hyperplane_quad


# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "P":       "#4d9ef7",   # blue  — polytope P
    "Q":       "#f77b4d",   # orange — polytope Q
    "sum":     "#7df7a0",   # green — Minkowski sum
    "contact": "#f7e94d",   # yellow — contact point
    "plane":   "#cc88ff",   # purple — hyperplane
    "hull":    "#4dfff7",   # cyan — neuro hull
    "cloud":   "#ff6688",   # pink — raw point cloud
    "u_vec":   "#ffffff",   # white — direction vector
    "bg":      "#0d0f14",
    "grid":    "#1e2d48",
    "text":    "#c8d0e0",
}

_LAYOUT_BASE = dict(
    paper_bgcolor=PALETTE["bg"],
    plot_bgcolor=PALETTE["bg"],
    font=dict(family="JetBrains Mono, monospace", color=PALETTE["text"], size=11),
    scene=dict(
        xaxis=dict(backgroundcolor=PALETTE["bg"], gridcolor=PALETTE["grid"],
                   showbackground=True, color=PALETTE["text"]),
        yaxis=dict(backgroundcolor=PALETTE["bg"], gridcolor=PALETTE["grid"],
                   showbackground=True, color=PALETTE["text"]),
        zaxis=dict(backgroundcolor=PALETTE["bg"], gridcolor=PALETTE["grid"],
                   showbackground=True, color=PALETTE["text"]),
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
        aspectmode="cube",
    ),
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=PALETTE["text"])),
    height=550,
)


def _layout(**overrides):
    import copy
    lay = copy.deepcopy(_LAYOUT_BASE)
    lay.update(overrides)
    return lay


# ─────────────────────────────────────────────────────────────────────────────
# Mesh trace for a convex hull
# ─────────────────────────────────────────────────────────────────────────────

def hull_mesh_trace(
    points: np.ndarray,
    hull: ConvexHull,
    color: str,
    name: str,
    opacity: float = 0.35,
    show_wireframe: bool = True,
) -> list:
    """Returns list of go traces: [Mesh3d, optionally Scatter3d wireframe]."""
    x, y, z, i, j, k = hull_mesh_arrays(hull, points)
    traces = []

    # Filled mesh
    traces.append(go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color=color,
        opacity=opacity,
        name=name,
        flatshading=True,
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3),
        hoverinfo="name",
    ))

    # Wireframe edges
    if show_wireframe:
        edge_x, edge_y, edge_z = [], [], []
        for s in hull.simplices:
            for a, b in [(s[0], s[1]), (s[1], s[2]), (s[0], s[2])]:
                pa, pb = points[a], points[b]
                edge_x += [pa[0], pb[0], None]
                edge_y += [pa[1], pb[1], None]
                edge_z += [pa[2], pb[2], None]
        traces.append(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode="lines",
            line=dict(color=color, width=1.5),
            name=f"{name} edges",
            hoverinfo="skip",
            showlegend=False,
        ))

    return traces


# ─────────────────────────────────────────────────────────────────────────────
# Full Minkowski Sum figure
# ─────────────────────────────────────────────────────────────────────────────

def figure_minkowski_sum(
    P_pts: np.ndarray, P_hull: ConvexHull,
    Q_pts: np.ndarray, Q_hull: ConvexHull,
    S_pts: np.ndarray, S_hull: ConvexHull,
    show_P: bool = True,
    show_Q: bool = True,
    show_sum: bool = True,
) -> go.Figure:
    """3-panel combined figure: P, Q, and P⊕Q."""
    traces = []
    if show_P:
        traces += hull_mesh_trace(P_pts, P_hull, PALETTE["P"], "P", opacity=0.4)
    if show_Q:
        traces += hull_mesh_trace(Q_pts, Q_hull, PALETTE["Q"], "Q", opacity=0.4)
    if show_sum:
        traces += hull_mesh_trace(S_pts, S_hull, PALETTE["sum"], "P⊕Q", opacity=0.30)

    fig = go.Figure(data=traces)
    fig.update_layout(**_layout(title="Minkowski Sum  P ⊕ Q"))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Support Explorer figure
# ─────────────────────────────────────────────────────────────────────────────

def figure_support_explorer(
    P_pts: np.ndarray, P_hull: ConvexHull,
    Q_pts: np.ndarray, Q_hull: ConvexHull,
    S_pts: np.ndarray, S_hull: ConvexHull,
    u: np.ndarray,
    cp_P: np.ndarray,
    cp_Q: np.ndarray,
    cp_S: np.ndarray,
    h_P: float,
    h_Q: float,
    h_S: float,
) -> go.Figure:
    """
    Shows all three polytopes + the direction vector u, contact points,
    and supporting hyperplanes for P, Q, and P⊕Q.
    """
    traces = []
    traces += hull_mesh_trace(P_pts, P_hull, PALETTE["P"],   "P",   opacity=0.25)
    traces += hull_mesh_trace(Q_pts, Q_hull, PALETTE["Q"],   "Q",   opacity=0.25)
    traces += hull_mesh_trace(S_pts, S_hull, PALETTE["sum"], "P⊕Q", opacity=0.18)

    u_norm = np.linalg.norm(u)
    if u_norm > 1e-12:
        u_hat = u / u_norm

        # Direction arrow  u
        scale = max(
            np.max(np.abs(P_pts)) if len(P_pts) else 1,
            np.max(np.abs(Q_pts)) if len(Q_pts) else 1,
        ) * 0.9
        arrow_end = u_hat * scale
        traces.append(go.Scatter3d(
            x=[0, arrow_end[0]], y=[0, arrow_end[1]], z=[0, arrow_end[2]],
            mode="lines+markers",
            line=dict(color=PALETTE["u_vec"], width=5),
            marker=dict(symbol=["circle", "diamond"], size=[3, 8], color=PALETTE["u_vec"]),
            name="u⃗  direction",
        ))

        # Hyperplanes for P, Q, and sum
        for (cp, h, color, lbl) in [
            (cp_P, h_P, PALETTE["P"],    "H(P)"),
            (cp_Q, h_Q, PALETTE["Q"],    "H(Q)"),
            (cp_S, h_S, PALETTE["sum"],  "H(P⊕Q)"),
        ]:
            X, Y, Z = hyperplane_quad(u_hat, h, size=2.8)
            traces.append(go.Surface(
                x=X, y=Y, z=Z,
                colorscale=[[0, color], [1, color]],
                opacity=0.15,
                showscale=False,
                name=lbl,
                hoverinfo="name",
            ))
            traces.append(go.Scatter3d(
                x=[cp[0]], y=[cp[1]], z=[cp[2]],
                mode="markers",
                marker=dict(size=9, color=PALETTE["contact"], symbol="diamond"),
                name=f"contact {lbl}",
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(**_layout(title="Support Function Explorer"))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Neuro Point Cloud figure
# ─────────────────────────────────────────────────────────────────────────────

def figure_neuro_cloud(
    cloud: np.ndarray,
    hull_pts: np.ndarray,
    hull: ConvexHull,
fig = go.Figure()
    for hs, color, name in [
        (hP, PALETTE["P"],   "h_P"),
        (hQ, PALETTE["Q"],   "h_Q"),
        (hS, PALETTE["sum"], "h_{P⊕Q}"),
    ]:
        # MUST BE INDENTED INSIDE THE FOR LOOP
        fig.add_trace(go.Scatterpolar(
            r=hs, theta=np.degrees(thetas),
            mode="lines",
            line=dict(color=color, width=2),
            name=name,
            fill="toself",
            # This logic ensures each trace has its own transparent color
            fillcolor=f"rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.15)" if color.startswith('#') else "rgba(125,125,125,0.1)"
        ))

    # Hull mesh
    traces += hull_mesh_trace(hull_pts, hull, PALETTE["hull"], "Convex Hull", opacity=0.25)

    fig = go.Figure(data=traces)
    fig.update_layout(**_layout(title="Neuro Point Cloud — Convex Hull"))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Support-function polar plot (2-D slice)
# ─────────────────────────────────────────────────────────────────────────────

def figure_support_polar(
    P_pts: np.ndarray,
    Q_pts: np.ndarray,
    S_pts: np.ndarray,
    plane: str = "xy",
    n_theta: int = 360,
) -> go.Figure:
    """
    Polar plot of h_K(u(θ)) for P, Q, and P⊕Q, evaluated in a 2-D slice.
    Confirms h_{P⊕Q} = h_P + h_Q.
    """
    from geometry_core import support_function
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    axis_map = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}
    a1, a2, _ = axis_map.get(plane, (0, 1, 2))

    def h_values(pts):
        hs = []
        for t in thetas:
            u = np.zeros(3)
            u[a1] = np.cos(t)
            u[a2] = np.sin(t)
            hs.append(support_function(pts, u))
        return np.array(hs)

    hP = h_values(P_pts)
    hQ = h_values(Q_pts)
    hS = h_values(S_pts)

    fig = go.Figure()
    for hs, color, name in [
        (hP, PALETTE["P"],   "h_P"),
        (hQ, PALETTE["Q"],   "h_Q"),
        (hS, PALETTE["sum"], "h_{P⊕Q}"),
    ]:
    fig.add_trace(go.Scatterpolar(
        r=hs, theta=np.degrees(thetas),
        mode="lines",
        line=dict(color=color, width=2),
        name=name,
        fill="toself",
        # Fix: Use a Plotly-safe color string for transparency
        fillcolor="rgba(125, 125, 125, 0.1)" 
    ))

    fig.update_layout(
        polar=dict(
            bgcolor=PALETTE["bg"],
            radialaxis=dict(gridcolor=PALETTE["grid"], color=PALETTE["text"]),
            angularaxis=dict(gridcolor=PALETTE["grid"], color=PALETTE["text"]),
        ),
        paper_bgcolor=PALETTE["bg"],
        font=dict(family="JetBrains Mono, monospace", color=PALETTE["text"]),
        title=f"Support Functions  (plane: {plane.upper()})",
        showlegend=True,
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig
