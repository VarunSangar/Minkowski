"""
pages_modules/page_support_explorer.py
=======================================
Page 2 — Support Function Explorer
"""

import streamlit as st
import numpy as np
from scipy.spatial import ConvexHull
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from geometry_core import (
    make_polytope, minkowski_sum_full,
    support_function, support_contact_point, supporting_hyperplane,
)
from viz_helpers import figure_support_explorer, PALETTE

POLYTOPES = ["cube", "octahedron", "tetrahedron", "icosahedron", "dodecahedron", "random_convex"]


def render():
    st.markdown("# Support Function Explorer")
    st.markdown(
        "Move the direction vector **ū** to interactively see the supporting "
        "hyperplane and contact point on P, Q, and P⊕Q."
    )

    with st.expander("📐  Support Function Definition"):
        st.latex(r"h_K(\vec{u}) \;=\; \sup\, \bigl\{\, \vec{u} \cdot x \;\mid\; x \in K \bigr\}")
        st.latex(
            r"\text{Contact point: } x^*(\vec{u}) = \arg\max_{x \in K}\; \vec{u} \cdot x"
        )
        st.latex(
            r"\text{Supporting hyperplane: } H(\vec{u}) = \bigl\{\, x : \vec{u} \cdot x = h_K(\vec{u}) \,\bigr\}"
        )

    st.markdown("---")

    # ── Shape controls ────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Polytope P</div>', unsafe_allow_html=True)
        p_shape = st.selectbox("Shape P", POLYTOPES, index=0, key="exp_p")
        p_scale = st.slider("Scale P", 0.3, 3.0, 1.0, 0.1, key="exp_ps")
    with c2:
        st.markdown('<div class="section-header">Polytope Q</div>', unsafe_allow_html=True)
        q_shape = st.selectbox("Shape Q", POLYTOPES, index=2, key="exp_q")
        q_scale = st.slider("Scale Q", 0.3, 3.0, 0.8, 0.1, key="exp_qs")

    # ── Direction vector ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">Direction Vector  ū</div>', unsafe_allow_html=True)
    dc1, dc2, dc3 = st.columns(3)
    ux = dc1.slider("u_x", -1.0, 1.0, 0.577, 0.01, key="ux")
    uy = dc2.slider("u_y", -1.0, 1.0, 0.577, 0.01, key="uy")
    uz = dc3.slider("u_z", -1.0, 1.0, 0.577, 0.01, key="uz")

    u = np.array([ux, uy, uz], dtype=float)
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-12:
        st.warning("Direction vector is zero — please move the sliders.")
        return
    u_hat = u / u_norm

    # ── Compute ───────────────────────────────────────────────────────────────
    P_pts = make_polytope(p_shape, p_scale)
    Q_pts = make_polytope(q_shape, q_scale, center=[2.0, 0, 0])
    P_hull = ConvexHull(P_pts)
    Q_hull = ConvexHull(Q_pts)
    S_pts, S_hull = minkowski_sum_full(P_pts, Q_pts)

    h_P  = support_function(P_pts, u_hat)
    h_Q  = support_function(Q_pts, u_hat)
    h_S  = support_function(S_pts, u_hat)

    cp_P = support_contact_point(P_pts, u_hat)
    cp_Q = support_contact_point(Q_pts, u_hat)
    cp_S = support_contact_point(S_pts, u_hat)

    # ── Metrics display ───────────────────────────────────────────────────────
    r1, r2, r3, r4 = st.columns(4)
    r1.markdown(f"""<div class="metric-card">
        <div class="metric-label">h_P(ū)</div>
        <div class="metric-value">{h_P:.4f}</div>
    </div>""", unsafe_allow_html=True)
    r2.markdown(f"""<div class="metric-card">
        <div class="metric-label">h_Q(ū)</div>
        <div class="metric-value">{h_Q:.4f}</div>
    </div>""", unsafe_allow_html=True)
    r3.markdown(f"""<div class="metric-card">
        <div class="metric-label">h_P + h_Q</div>
        <div class="metric-value">{h_P + h_Q:.4f}</div>
    </div>""", unsafe_allow_html=True)
    r4.markdown(f"""<div class="metric-card">
        <div class="metric-label">h_{{P⊕Q}}(ū)</div>
        <div class="metric-value" style="color:{'#7df7a0' if abs(h_S-(h_P+h_Q))<1e-6 else '#f77b4d'}">{h_S:.4f}</div>
    </div>""", unsafe_allow_html=True)

    # ── Additivity verification ───────────────────────────────────────────────
    err = abs(h_S - (h_P + h_Q))
    if err < 1e-5:
        st.success(f"✓ Additivity verified:  h_{{P⊕Q}}(ū) = h_P(ū) + h_Q(ū)  (error = {err:.2e})")
    else:
        st.warning(f"Additivity error = {err:.2e}  (may be numerical for this configuration)")

    # ── LaTeX readout ─────────────────────────────────────────────────────────
    st.latex(
        r"\vec{u} = \begin{pmatrix}"
        + f"{u_hat[0]:.3f} \\\\ {u_hat[1]:.3f} \\\\ {u_hat[2]:.3f}"
        + r"\end{pmatrix},\quad"
        + r"h_P = " + f"{h_P:.4f}" + r",\quad"
        + r"h_Q = " + f"{h_Q:.4f}" + r",\quad"
        + r"h_{{P \oplus Q}} = " + f"{h_S:.4f}"
    )

    # ── Contact point readouts ────────────────────────────────────────────────
    cp_cols = st.columns(3)
    for col, cp, label, color in zip(
        cp_cols,
        [cp_P, cp_Q, cp_S],
        ["Contact P", "Contact Q", "Contact P⊕Q"],
        [PALETTE["P"], PALETTE["Q"], PALETTE["sum"]],
    ):
        col.markdown(
            f'<div class="metric-card" style="border-color:{color}55">'
            f'<div class="metric-label" style="color:{color}">{label}</div>'
            f'<div class="metric-value" style="font-size:14px">'
            f"({cp[0]:.3f}, {cp[1]:.3f}, {cp[2]:.3f})"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # ── 3-D figure ────────────────────────────────────────────────────────────
    fig = figure_support_explorer(
        P_pts, P_hull, Q_pts, Q_hull, S_pts, S_hull,
        u_hat, cp_P, cp_Q, cp_S, h_P, h_Q, h_S,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Info ──────────────────────────────────────────────────────────────────
    st.info(
        "The **yellow diamonds** are the contact points where the supporting hyperplane "
        "touches each polytope. The **translucent planes** are the supporting hyperplanes "
        "at the current direction ū."
    )
