"""
pages_modules/page_about.py
============================
Page 4 — Theory & About
"""

import streamlit as st


def render():
    st.markdown("# Theory & About")
    st.markdown("---")

    st.markdown("## Minkowski Geometry — Core Concepts")

    with st.expander("1.  Minkowski Sum", expanded=True):
        st.latex(r"P \oplus Q = \{ p + q \mid p \in P,\; q \in Q \}")
        st.markdown("""
        The Minkowski Sum of two sets is obtained by **vector addition** of every pair of points.
        For convex polytopes:
        - The result is always convex
        - The number of faces is at most |faces(P)| + |faces(Q)|
        - It can be computed in **O(n log n)** via support functions
        """)

    with st.expander("2.  Support Function"):
        st.latex(r"h_K(\vec{u}) = \sup\,\{ \vec{u} \cdot x \mid x \in K \}")
        st.markdown("""
        The support function encodes the **width** of a convex body in every direction.
        
        Key properties:
        """)
        st.latex(r"h_{P \oplus Q}(\vec{u}) = h_P(\vec{u}) + h_Q(\vec{u}) \quad \text{(additivity)}")
        st.latex(r"h_{\lambda K}(\vec{u}) = \lambda\, h_K(\vec{u}) \quad \text{(positive homogeneity)}")
        st.latex(r"h_{K+t}(\vec{u}) = h_K(\vec{u}) + \vec{u}\cdot t \quad \text{(translation)}")

    with st.expander("3.  Supporting Hyperplane"):
        st.latex(
            r"H_{K}(\vec{u}) = \bigl\{\, x \in \mathbb{R}^3 \;\mid\; \vec{u} \cdot x = h_K(\vec{u}) \,\bigr\}"
        )
        st.markdown("""
        The supporting hyperplane **touches** the polytope K from the direction **u** without 
        penetrating it. Every convex body is the intersection of all its supporting half-spaces:
        """)
        st.latex(r"K = \bigcap_{\vec{u} \in \mathbb{S}^2} \bigl\{\, x : \vec{u} \cdot x \leq h_K(\vec{u}) \,\bigr\}")

    with st.expander("4.  Quickhull Algorithm"):
        st.markdown("""
        **Quickhull** computes the convex hull of n points in **O(n log n)** average time.
        
        Steps:
        1. Find the point farthest in ±x, ±y, ±z to form initial simplex
        2. For each face, find the point farthest outside (if any)
        3. Add it as a new vertex, create new faces, delete hidden faces
        4. Recurse until no outside points remain
        
        Worst case: O(n²) for adversarial inputs; average O(n log n) for random inputs.
        """)
        st.latex(r"\text{Output: } \partial\,\text{conv}(P) = \text{boundary of convex hull}")

    st.markdown("---")
    st.markdown("## Applications in Theoretical Neuroscience")

    with st.expander("Neural State-Space Geometry"):
        st.markdown("""
        In computational neuroscience, neural population activity can be modeled as 
        **points in a high-dimensional state space**. The geometry of these trajectories 
        encodes information processing:
        
        - **Minkowski Sum** → models the "reachable set" of a neural dynamical system
        - **Convex Hull** → the minimal convex region containing all accessible states
        - **Support Function** → quantifies the extremal response of a neural population 
          to linear readouts
        
        Key applications:
        - **Neural manifolds**: low-dimensional attractors embedded in high-D activity space
        - **Capacity analysis**: Minkowski geometry bounds the linear separability of classes
        - **Reachability**: support functions define the boundary of computationally reachable states
        """)
        st.latex(
            r"\text{State space: } \mathbf{r}(t) \in \mathbb{R}^N, \quad "
            r"N = \text{number of neurons}"
        )
        st.latex(
            r"\text{Linear readout: } y = \vec{w}^\top \mathbf{r}(t), \quad "
            r"y^* = h_{\mathcal{M}}(\vec{w})"
        )

    st.markdown("---")
    st.markdown("## Architecture")
    st.code("""
minkowski_app/
├── app.py                    # Entry point, routing, CSS
├── geometry_core.py          # Mathematical engine
│   ├── support_function()    # h_K(u) = sup{ u·x | x ∈ K }
│   ├── support_contact_point()
│   ├── supporting_hyperplane()
│   ├── minkowski_sum_full()  # O(nm) + O(k log k) hull
│   ├── convex_hull_from_cloud() # Quickhull via scipy
│   └── make_polytope()       # Canonical polytope generators
├── viz_helpers.py            # Plotly figure builders
│   ├── figure_minkowski_sum()
│   ├── figure_support_explorer()
│   ├── figure_neuro_cloud()
│   └── figure_support_polar()
├── pages_modules/
│   ├── page_minksum.py       # Minkowski Sum page
│   ├── page_support_explorer.py
│   ├── page_neuro_hull.py
│   └── page_about.py
└── requirements.txt
    """, language="")

    st.markdown("## References")
    st.markdown("""
    1. **Grünbaum, B.** (1967). *Convex Polytopes*. Interscience Publishers.
    2. **Schneider, R.** (1993). *Convex Bodies: The Brunn-Minkowski Theory*. Cambridge University Press.
    3. **Barber, C.B., Dobkin, D., Huhdanpaa, H.** (1996). *The Quickhull Algorithm for Convex Hulls*. ACM TOMS.
    4. **Cunningham, J.P., Yu, B.M.** (2014). *Dimensionality reduction for large-scale neural recordings*. Nature Neuroscience.
    5. **Jazayeri, M., Ostojic, S.** (2021). *Interpreting neural computations by examining intrinsic and embedding dimensionality*. Current Opinion in Neurobiology.
    """)
