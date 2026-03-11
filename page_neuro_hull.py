"""
pages_modules/page_neuro_hull.py
=================================
Page 3 — Neuro Point Cloud & Convex Hull
"""

import streamlit as st
import numpy as np
import pandas as pd
import io
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from geometry_core import convex_hull_from_cloud
from viz_helpers import figure_neuro_cloud, PALETTE


# ── Demo cloud generators ─────────────────────────────────────────────────────

def _demo_neural_attractor(n: int = 600, seed: int = 0) -> np.ndarray:
    """Lorenz-like neural attractor point cloud."""
    rng = np.random.default_rng(seed)
    pts = []
    x, y, z = 0.1, 0.0, 0.0
    dt = 0.01
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    for _ in range(n * 20):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt + rng.normal(0, 0.05)
        y += dy * dt + rng.normal(0, 0.05)
        z += dz * dt + rng.normal(0, 0.05)
        pts.append([x, y, z])
    pts = np.array(pts)
    # Subsample
    idx = rng.choice(len(pts), size=n, replace=False)
    return pts[idx]


def _demo_sphere_cloud(n: int = 400) -> np.ndarray:
    rng = np.random.default_rng(7)
    pts = rng.standard_normal((n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts += rng.normal(0, 0.12, pts.shape)
    return pts


def _demo_torus_cloud(n: int = 500) -> np.ndarray:
    rng = np.random.default_rng(3)
    R, r = 2.0, 0.7
    theta = rng.uniform(0, 2 * np.pi, n)
    phi   = rng.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return np.column_stack([x, y, z]) + rng.normal(0, 0.1, (n, 3))


DEMO_CLOUDS = {
    "Lorenz neural attractor": _demo_neural_attractor,
    "Noisy sphere":            _demo_sphere_cloud,
    "Torus state-space":       _demo_torus_cloud,
}


def render():
    st.markdown("# Neuro Point Cloud — Convex Hull")
    st.markdown(
        "Import a 3-D point cloud representing **neural state-space data** "
        "and compute its convex hull using the **Quickhull algorithm** (O(n log n))."
    )

    with st.expander("📐  Quickhull Algorithm", expanded=False):
        st.latex(
            r"\text{ConvexHull}(P) = \text{smallest convex set } C "
            r"\text{ s.t. } P \subseteq C"
        )
        st.markdown("""
        **Quickhull** (O(n log n) average, O(n²) worst):
        1. Find extreme points along each axis → initial simplex
        2. For each face, find the farthest outside point
        3. Recurse: replace face with triangles connecting to that point
        4. Points inside the current hull are discarded at each step
        
        *Implemented via `scipy.spatial.ConvexHull` (Qhull C library).*
        """)

    st.markdown("---")

    # ── Data source ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Data Source</div>', unsafe_allow_html=True)
    source = st.radio(
        "Input",
        ["Demo cloud", "Upload CSV/NPY"],
        horizontal=True,
        label_visibility="collapsed",
    )

    cloud = None

    if source == "Demo cloud":
        demo_name = st.selectbox("Choose demo", list(DEMO_CLOUDS.keys()))
        n_pts = st.slider("Number of points", 100, 2000, 500, 50)
        cloud = DEMO_CLOUDS[demo_name](n_pts)

    else:
        uploaded = st.file_uploader(
            "Upload .csv (3 columns: x,y,z) or .npy (shape N×3)",
            type=["csv", "npy"],
        )
        if uploaded is not None:
            try:
                if uploaded.name.endswith(".npy"):
                    cloud = np.load(io.BytesIO(uploaded.read()))
                else:
                    df = pd.read_csv(uploaded)
                    cloud = df.iloc[:, :3].values.astype(float)
                st.success(f"Loaded {len(cloud)} points.")
            except Exception as e:
                st.error(f"Failed to load file: {e}")
        else:
            st.info("Upload a file or switch to Demo cloud.")

    if cloud is None:
        return

    cloud = np.asarray(cloud, dtype=float)
    if cloud.shape[1] != 3:
        st.error("Point cloud must have exactly 3 columns (x, y, z).")
        return
    if len(cloud) < 4:
        st.error("Need at least 4 non-coplanar points.")
        return

    # ── Compute hull ──────────────────────────────────────────────────────────
    try:
        hull_pts, hull = convex_hull_from_cloud(cloud)
    except Exception as e:
        st.error(f"Convex hull computation failed: {e}")
        return

    # ── Metrics ───────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"""<div class="metric-card">
        <div class="metric-label">Total Points</div>
        <div class="metric-value">{len(cloud)}</div>
    </div>""", unsafe_allow_html=True)
    m2.markdown(f"""<div class="metric-card">
        <div class="metric-label">Hull Vertices</div>
        <div class="metric-value">{len(hull.vertices)}</div>
    </div>""", unsafe_allow_html=True)
    m3.markdown(f"""<div class="metric-card">
        <div class="metric-label">Hull Volume</div>
        <div class="metric-value">{hull.volume:.3f}</div>
    </div>""", unsafe_allow_html=True)
    m4.markdown(f"""<div class="metric-card">
        <div class="metric-label">Hull Surface Area</div>
        <div class="metric-value">{hull.area:.3f}</div>
    </div>""", unsafe_allow_html=True)

    # ── 3-D figure ────────────────────────────────────────────────────────────
    fig = figure_neuro_cloud(cloud, hull_pts, hull)
    st.plotly_chart(fig, use_container_width=True)

    # ── Statistics ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Point Cloud Statistics")

    stat_data = {
        "Dimension": ["X", "Y", "Z"],
        "Min":    [f"{cloud[:,i].min():.4f}" for i in range(3)],
        "Max":    [f"{cloud[:,i].max():.4f}" for i in range(3)],
        "Mean":   [f"{cloud[:,i].mean():.4f}" for i in range(3)],
        "Std":    [f"{cloud[:,i].std():.4f}"  for i in range(3)],
    }
    st.dataframe(stat_data, use_container_width=True)

    # ── Principal axes ────────────────────────────────────────────────────────
    st.markdown("### Principal Component Analysis  (state-space axes)")
    centered = cloud - cloud.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    pca_data = {
        "Component": ["PC1", "PC2", "PC3"],
        "Variance":  [f"{v:.4f}" for v in eigvals],
        "% Variance":[f"{100*v/eigvals.sum():.2f}%" for v in eigvals],
        "Direction": [
            f"({eigvecs[0,i]:.3f}, {eigvecs[1,i]:.3f}, {eigvecs[2,i]:.3f})"
            for i in range(3)
        ],
    }
    st.dataframe(pca_data, use_container_width=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Export Hull Vertices")
    hull_df = pd.DataFrame(hull_pts, columns=["x", "y", "z"])
    csv = hull_df.to_csv(index=False).encode()
    st.download_button(
        "⬇  Download hull vertices (CSV)",
        data=csv,
        file_name="hull_vertices.csv",
        mime="text/csv",
    )
