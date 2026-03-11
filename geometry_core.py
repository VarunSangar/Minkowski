"""
geometry_core.py
================
Mathematical core for Minkowski Geometry computations.

Algorithms:
  - Support function  h_K(u) = sup{ u·x | x ∈ K }
  - Minkowski Sum via support functions + convex hull
  - O(n log n) convex hull (Quickhull via scipy)
  - Supporting hyperplane & contact-point extraction
  - Convex Hull from point cloud (Quickhull / scipy.spatial)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from typing import Tuple, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Support Function
# ─────────────────────────────────────────────────────────────────────────────

def support_function(points: np.ndarray, u: np.ndarray) -> float:
    """
    h_K(u) = sup { u · x  |  x ∈ K }

    For a finite point set (vertices of a polytope), the supremum is achieved
    at one of the vertices, so we just take the max inner-product.

    Parameters
    ----------
    points : (N, 3)  vertex set of polytope K
    u      : (3,)    direction vector

    Returns
    -------
    float : support value
    """
    u = np.asarray(u, dtype=float)
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-12:
        return 0.0
    u_hat = u / u_norm
    dots = points @ u_hat
    return float(np.max(dots))


def support_contact_point(points: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Returns the vertex (or centroid of face) that achieves h_K(u).
    If multiple vertices tie within tolerance, returns their centroid.
    """
    u = np.asarray(u, dtype=float)
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-12:
        return points[0]
    u_hat = u / u_norm
    dots = points @ u_hat
    max_val = np.max(dots)
    mask = dots >= max_val - 1e-9
    return points[mask].mean(axis=0)


def supporting_hyperplane(
    points: np.ndarray, u: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Returns (normal, offset) of the supporting hyperplane in direction u:
      H = { x : n·x = h_K(u) }
    where n = u / ||u||.
    """
    u = np.asarray(u, dtype=float)
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-12:
        return np.array([0.0, 0.0, 1.0]), 0.0
    n = u / u_norm
    h = support_function(points, u)
    return n, h


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Minkowski Sum
# ─────────────────────────────────────────────────────────────────────────────

def minkowski_sum_pointsets(
    P: np.ndarray, Q: np.ndarray
) -> np.ndarray:
    """
    Compute the Minkowski Sum  P ⊕ Q  as a point cloud.

    Strategy (O(n·m) generation + O(k log k) hull):
      1. Generate all pairwise sums  { p + q | p ∈ P, q ∈ Q }
         (|P|·|Q| candidates — for polytopes only vertex-vertex sums
          on the convex hull matter, so we pre-reduce to hull vertices.)
      2. Take the convex hull of the candidates → vertices of P ⊕ Q.

    Returns
    -------
    hull_vertices : (V, 3)  vertices of the Minkowski Sum polytope
    """
    # Pre-reduce to convex hull vertices (only hull vertices contribute)
    P_hull = _hull_vertices(P)
    Q_hull = _hull_vertices(Q)

    # Pairwise sums  (|P_hull| * |Q_hull|, 3)
    sums = (P_hull[:, None, :] + Q_hull[None, :, :]).reshape(-1, 3)

    return _hull_vertices(sums)


def minkowski_sum_full(
    P: np.ndarray, Q: np.ndarray
) -> Tuple[np.ndarray, ConvexHull]:
    """
    Returns (hull_vertices, ConvexHull object) of P ⊕ Q.
    The ConvexHull object carries `.simplices` for mesh rendering.
    """
    verts = minkowski_sum_pointsets(P, Q)
    if len(verts) < 4:
        # Degenerate — pad slightly for hull computation
        verts = verts + np.random.randn(*verts.shape) * 1e-8
    hull = ConvexHull(verts)
    return verts, hull


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Convex Hull (Quickhull via scipy)
# ─────────────────────────────────────────────────────────────────────────────

def convex_hull_from_cloud(
    points: np.ndarray,
) -> Tuple[np.ndarray, ConvexHull]:
    """
    Compute the 3-D convex hull of an arbitrary point cloud using
    scipy's Quickhull implementation (O(n log n) average).

    Parameters
    ----------
    points : (N, 3)

    Returns
    -------
    hull_vertices : (V, 3)
    hull          : scipy ConvexHull
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[1] != 3:
        raise ValueError("Point cloud must have shape (N, 3).")
    if len(pts) < 4:
        raise ValueError("Need at least 4 non-coplanar points.")

    hull = ConvexHull(pts)
    return pts[hull.vertices], hull


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hull_vertices(points: np.ndarray) -> np.ndarray:
    """Return just the convex-hull vertices of a point set."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 4:
        return pts
    try:
        hull = ConvexHull(pts)
        return pts[hull.vertices]
    except Exception:
        return pts


def make_polytope(name: str, scale: float = 1.0, center: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Generate canonical polytope vertices.

    Available: cube, octahedron, tetrahedron, icosahedron, dodecahedron,
               random_convex
    """
    c = np.zeros(3) if center is None else np.asarray(center, dtype=float)

    if name == "cube":
        v = np.array([
            [s0, s1, s2]
            for s0 in [-1, 1]
            for s1 in [-1, 1]
            for s2 in [-1, 1]
        ], dtype=float)

    elif name == "octahedron":
        v = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
        ], dtype=float)

    elif name == "tetrahedron":
        v = np.array([
            [1, 1, 1], [1, -1, -1],
            [-1, 1, -1], [-1, -1, 1],
        ], dtype=float)

    elif name == "icosahedron":
        phi = (1 + 5**0.5) / 2
        v = np.array([
            [-1,  phi, 0], [1,  phi, 0], [-1, -phi, 0], [1, -phi,  0],
            [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
            [phi, 0, -1], [phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
        ], dtype=float)

    elif name == "dodecahedron":
        phi = (1 + 5**0.5) / 2
        inv_phi = 1.0 / phi
        v = []
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                for s3 in [-1, 1]:
                    v.append([s1, s2, s3])
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                v.append([0, s1 * inv_phi, s2 * phi])
                v.append([s1 * inv_phi, s2 * phi, 0])
                v.append([s1 * phi, 0, s2 * inv_phi])
        v = np.array(v, dtype=float)

    elif name == "random_convex":
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((60, 3))
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        hull = ConvexHull(raw)
        v = raw[hull.vertices]

    else:
        raise ValueError(f"Unknown polytope: {name}")

    # Normalise to unit circumsphere, then scale
    r = np.max(np.linalg.norm(v, axis=1))
    v = v / r * scale + c
    return v


def hull_mesh_arrays(hull: ConvexHull, points: np.ndarray):
    """
    Convert scipy ConvexHull to flat arrays suitable for Plotly Mesh3d.

    Returns
    -------
    x, y, z : 1-D vertex coordinate arrays
    i, j, k : triangle index arrays
    """
    verts = points[hull.vertices] if hasattr(hull, 'vertices') else points
    # Re-index simplices to local vertex numbering
    global_to_local = {g: l for l, g in enumerate(hull.vertices)}
    tri = np.array([[global_to_local[idx] for idx in s] for s in hull.simplices])
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    i, j, k = tri[:, 0], tri[:, 1], tri[:, 2]
    return x, y, z, i, j, k


def hyperplane_quad(
    normal: np.ndarray, offset: float, size: float = 2.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a quad mesh for a hyperplane  n·x = offset  for visualisation.
    Returns (X, Y, Z) 2-D arrays for Plotly surface.
    """
    n = normal / np.linalg.norm(normal)
    # Find two tangent vectors
    if abs(n[0]) < 0.9:
        t1 = np.cross(n, [1, 0, 0])
    else:
        t1 = np.cross(n, [0, 1, 0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)

    base = n * offset
    us = np.linspace(-size, size, 3)
    vs = np.linspace(-size, size, 3)
    U, V = np.meshgrid(us, vs)
    X = base[0] + U * t1[0] + V * t2[0]
    Y = base[1] + U * t1[1] + V * t2[1]
    Z = base[2] + U * t1[2] + V * t2[2]
    return X, Y, Z
