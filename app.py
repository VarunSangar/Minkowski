"""
Minkowski Geometry Visualizer — Streamlit Application
======================================================
Entry point. Run with: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Minkowski Geometry Lab",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (laboratory dark theme) ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0f14;
    color: #c8d0e0;
}
.main { background-color: #0d0f14; }
.stApp { background-color: #0d0f14; }

h1, h2, h3 { color: #7eb8f7; font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; }

.metric-card {
    background: #161b27;
    border: 1px solid #2a3550;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 8px 0;
    font-family: 'JetBrains Mono', monospace;
}
.metric-label { color: #7eb8f7; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; }
.metric-value { color: #e2e8f0; font-size: 22px; font-weight: 600; margin-top: 4px; }

.section-header {
    border-left: 3px solid #7eb8f7;
    padding-left: 12px;
    margin: 20px 0 12px 0;
    color: #7eb8f7;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-family: 'JetBrains Mono', monospace;
}

.stSelectbox label, .stSlider label, .stFileUploader label {
    color: #7eb8f7 !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #161b27;
    color: #8899bb;
    border-radius: 6px 6px 0 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}
.stTabs [aria-selected="true"] {
    background-color: #1e2d48 !important;
    color: #7eb8f7 !important;
}

.equation-box {
    background: #0a0d14;
    border: 1px solid #2a3550;
    border-radius: 6px;
    padding: 20px;
    text-align: center;
    margin: 12px 0;
}

code {
    background: #1a2030 !important;
    color: #7dd8a8 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stButton>button {
    background: #1e3a5f;
    color: #7eb8f7;
    border: 1px solid #2a5a9f;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-radius: 4px;
    transition: all 0.2s;
}
.stButton>button:hover {
    background: #2a4f80;
    border-color: #7eb8f7;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# ── Import pages ─────────────────────────────────────────────────────────────
from pages_modules import (
    import pages_modules.page_minksum as page_minksum.py
    import pages_modules.page_support_explorer as page_support_explorer
    import pages_modules.page_neuro_hull as page_neuro_hull
    import pages_modules.page_about as page_about
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⬡ Minkowski Lab")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "🔷  Minkowski Sum",
            "🔎  Support Explorer",
            "🧠  Neuro Point Cloud",
            "📖  Theory & About",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<span style='font-size:11px;color:#445566;font-family:JetBrains Mono,monospace;'>"
        "v1.0 · Computational Geometry<br>© 2024 Minkowski Lab</span>",
        unsafe_allow_html=True,
    )

# ── Route ─────────────────────────────────────────────────────────────────────
if page.startswith("🔷"):
    page_minksum.render()
elif page.startswith("🔎"):
    page_support_explorer.render()
elif page.startswith("🧠"):
    page_neuro_hull.render()
else:
    page_about.render()
