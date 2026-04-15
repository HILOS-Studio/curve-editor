"""Piecewise cubic Bezier evaluation."""

import numpy as np


def eval_cubic(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float
) -> np.ndarray:
    """Evaluate a single cubic Bezier segment at parameter t in [0, 1]."""
    u = 1 - t
    return u**3 * p0 + 3 * u**2 * t * p1 + 3 * u * t**2 * p2 + t**3 * p3


def eval_piecewise_bezier(control_points: np.ndarray, n_eval: int = 100) -> np.ndarray:
    """Evaluate a piecewise cubic Bezier curve at n_eval evenly-spaced parameter values.

    Args:
        control_points: (M, 2) array where M = 3*n_segments + 1.
        n_eval: Number of output points.

    Returns:
        (n_eval, 2) array of evaluated points.
    """
    cp = np.asarray(control_points, dtype=np.float64)
    if len(cp) < 4:
        return cp.copy()
    n_seg = (len(cp) - 1) // 3
    pts = []
    for i in range(n_eval):
        t = i / (n_eval - 1)
        seg_t = t * n_seg
        seg_idx = min(int(seg_t), n_seg - 1)
        local_t = seg_t - seg_idx
        si = seg_idx * 3
        pts.append(eval_cubic(cp[si], cp[si + 1], cp[si + 2], cp[si + 3], local_t))
    return np.array(pts, dtype=np.float64)
