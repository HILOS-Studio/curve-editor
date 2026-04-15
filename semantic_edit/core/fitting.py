"""Least-squares piecewise cubic Bezier fitting."""

import numpy as np


def fit_bezier_lsq(points: np.ndarray, n_segments: int) -> np.ndarray:
    """Fit a piecewise cubic Bezier to a sequence of 2D points.

    Parameterizes by chord length, builds a Bernstein basis matrix, and solves
    via least squares. First and last control points are pinned to the first
    and last data points.

    Args:
        points: (N, 2) array of target points.
        n_segments: Number of cubic segments. Output has 3*n_segments + 1 control points.

    Returns:
        (3*n_segments + 1, 2) control points.
    """
    points = np.asarray(points, dtype=np.float64)
    n_pts = len(points)
    n_cp = 3 * n_segments + 1

    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(diffs)])
    total = cum[-1]
    if total < 1e-10:
        return np.tile(points[0], (n_cp, 1))
    t_vals = cum / total

    A = np.zeros((n_pts, n_cp))
    for i, t in enumerate(t_vals):
        seg_t = t * n_segments
        s = min(int(seg_t), n_segments - 1)
        u = seg_t - s
        u1 = 1 - u
        base = 3 * s
        A[i, base] = u1 * u1 * u1
        A[i, base + 1] = 3 * u1 * u1 * u
        A[i, base + 2] = 3 * u1 * u * u
        A[i, base + 3] = u * u * u

    cp_first = points[0]
    cp_last = points[-1]

    rhs = points.copy()
    rhs -= np.outer(A[:, 0], cp_first)
    rhs -= np.outer(A[:, -1], cp_last)

    A_free = A[:, 1:-1]

    if A_free.shape[1] == 0:
        cp = np.zeros((n_cp, 2))
        cp[0] = cp_first
        cp[-1] = cp_last
        return cp

    result, _, _, _ = np.linalg.lstsq(A_free, rhs, rcond=None)

    cp = np.zeros((n_cp, 2))
    cp[0] = cp_first
    cp[-1] = cp_last
    cp[1:-1] = result

    return cp
