"""
pages_modules/page_minksum.py
==============================
Page 1 — Minkowski Sum Visualizer
"""
import streamlit as st
import numpy as np
from scipy.spatial import ConvexHull
import sys, os

# Force the root directory into the path
sys.path.append(os.getcwd())
from geometry_core import make_polytope, minkowski_sum_full, _hull_vertices
from viz_helpers import figure_minkowski_sum, figure_support_polar, PALETTE


POLYTOPES = ["cube", "octahedron", "tetrahedron", "icosahedron", "dodecahedron", "random_convex"]


def render():
    st.markdown("# Minkowski Sum  —  P ⊕ Q")
    st.markdown(
        "Computes the Minkowski Sum of two 3-D polytopes using the **support function** "
        "decomposition. The sum is taken as the convex hull of all pairwise vertex sums."
    )

    # ── Theory box ────────────────────────────────────────────────────────────
    with st.expander("📐  Mathematical Foundation", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r"P \oplus Q \;=\; \{ p + q \;\mid\; p \in P,\; q \in Q \}")
            st.latex(r"h_{P \oplus Q}(u) \;=\; h_P(u) \;+\; h_Q(u)")
        with col2:
            st.latex(r"h_K(u) \;=\; \sup\,\{ u \cdot x \;\mid\; x \in K \}")
            st.latex(r"\text{Key property: additivity under Minkowski sum}")

    st.markdown("---")

    # ── Controls ──────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Polytope P</div>', unsafe_allow_html=True)
        p_shape  = st.selectbox("Shape P", POLYTOPES, index=0, key="p_shape")
        p_scale  = st.slider("Scale P", 0.3, 3.0, 1.0, 0.1, key="p_scale")
        p_cx = st.slider("Center P — x", -2.0, 2.0, 0.0, 0.1, key="pcx")
        p_cy = st.slider("Center P — y", -2.0, 2.0, 0.0, 0.1, key="pcy")
        p_cz = st.slider("Center P — z", -2.0, 2.0, 0.0, 0.1, key="pcz")

    with col_b:
        st.markdown('<div class="section-header">Polytope Q</div>', unsafe_allow_html=True)
        q_shape  = st.selectbox("Shape Q", POLYTOPES, index=1, key="q_shape")
        q_scale  = st.slider("Scale Q", 0.3, 3.0, 1.0, 0.1, key="q_scale")
        q_cx = st.slider("Center Q — x", -2.0, 2.0, 0.0, 0.1, key="qcx")
        q_cy = st.slider("Center Q — y", -2.0, 2.0, 0.0, 0.1, key="qcy")
        q_cz = st.slider("Center Q — z", -2.0, 2.0, 0.0, 0.1, key="qcz")

    # ── Visibility toggles ────────────────────────────────────────────────────
    st.markdown("---")
    v_col1, v_col2, v_col3 = st.columns(3)
    show_P   = v_col1.checkbox("Show P",   value=True)
    show_Q   = v_col2.checkbox("Show Q",   value=True)
    show_sum = v_col3.checkbox("Show P⊕Q", value=True)

    # ── Compute ───────────────────────────────────────────────────────────────
    P_pts = make_polytope(p_shape, p_scale, center=[p_cx, p_cy, p_cz])
    Q_pts = make_polytope(q_shape, q_scale, center=[q_cx, q_cy, q_cz])

    P_hull = ConvexHull(P_pts)
    Q_hull = ConvexHull(Q_pts)
    S_pts, S_hull = minkowski_sum_full(P_pts, Q_pts)

    # ── 3-D figure ────────────────────────────────────────────────────────────
    fig3d = figure_minkowski_sum(
        P_pts, P_hull, Q_pts, Q_hull, S_pts, S_hull,
        show_P=show_P, show_Q=show_Q, show_sum=show_sum,
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"""<div class="metric-card">
        <div class="metric-label">P Vertices</div>
        <div class="metric-value">{len(P_pts)}</div>
    </div>""", unsafe_allow_html=True)
    m2.markdown(f"""<div class="metric-card">
        <div class="metric-label">Q Vertices</div>
        <div class="metric-value">{len(Q_pts)}</div>
    </div>""", unsafe_allow_html=True)
    m3.markdown(f"""<div class="metric-card">
        <div class="metric-label">P⊕Q Vertices</div>
        <div class="metric-value">{len(S_pts)}</div>
    </div>""", unsafe_allow_html=True)
    vol_P = P_hull.volume
    vol_Q = Q_hull.volume
    vol_S = S_hull.volume
    m4.markdown(f"""<div class="metric-card">
        <div class="metric-label">Sum Volume</div>
        <div class="metric-value">{vol_S:.3f}</div>
    </div>""", unsafe_allow_html=True)

    # ── Support function polar plot ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Support Function  h_K(u)  — 2-D Slice")
    plane = st.selectbox("Projection plane", ["xy", "xz", "yz"], index=0)
    fig_polar = figure_support_polar(P_pts, Q_pts, S_pts, plane=plane)
    st.plotly_chart(fig_polar, use_container_width=True)

    st.info(
        "**Verification**: The outer trace h_{P⊕Q} should equal h_P + h_Q at every angle — "
        "confirming the additivity of support functions."
    )

    # ── Volume table ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Geometric Properties")
    st.dataframe({
        "Polytope":      ["P", "Q", "P ⊕ Q"],
        "Vertices":      [len(P_pts), len(Q_pts), len(S_pts)],
        "Faces (tri)":   [len(P_hull.simplices), len(Q_hull.simplices), len(S_hull.simplices)],
        "Volume":        [f"{vol_P:.4f}", f"{vol_Q:.4f}", f"{vol_S:.4f}"],
        "Surface Area":  [
            f"{P_hull.area:.4f}",
            f"{Q_hull.area:.4f}",
            f"{S_hull.area:.4f}",
        ],
    }, use_container_width=True)
