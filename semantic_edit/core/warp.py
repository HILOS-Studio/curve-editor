"""Grid-based image warping driven by curve correspondences."""

import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from ..constants import CURVE_NAMES, SCALE
from .bezier import eval_piecewise_bezier

MOVE_THRESHOLD = 0.5  # pixels at image scale


def build_moved_and_anchor_points(
    src_curves: dict,
    dst_curves: dict,
    sigma: float,
    n_eval_per_curve: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify curve points into moved drivers and fixed anchors.

    1. Evaluate all curves densely at image scale.
    2. Points with displacement > MOVE_THRESHOLD are "moved" (warp drivers).
    3. Unmoved points outside the sigma radius of any moved point become "anchors"
       (zero displacement, actively holding the image in place).

    Returns:
        (moved_src, moved_dst, anchor_pts) — all at image resolution (1024).
    """
    all_src, all_dst = [], []
    for name in CURVE_NAMES:
        src_entry = src_curves.get(name)
        dst_entry = dst_curves.get(name)
        if not src_entry or not dst_entry:
            continue
        src_cp = src_entry["control_points"]
        dst_cp = dst_entry["control_points"]
        if len(src_cp) < 4 or len(dst_cp) < 4:
            continue
        sp = eval_piecewise_bezier(src_cp, n_eval_per_curve) * SCALE
        dp = eval_piecewise_bezier(dst_cp, n_eval_per_curve) * SCALE
        n = min(len(sp), len(dp))
        all_src.append(sp[:n])
        all_dst.append(dp[:n])

    if not all_src:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2))

    src = np.concatenate(all_src)
    dst = np.concatenate(all_dst)

    disp = np.linalg.norm(dst - src, axis=1)
    moved_mask = disp > MOVE_THRESHOLD

    moved_src = src[moved_mask]
    moved_dst = dst[moved_mask]
    unmoved_src = src[~moved_mask]

    if len(moved_src) == 0:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2))

    tree = cKDTree(moved_src)
    dists, _ = tree.query(unmoved_src)
    outside_mask = dists > sigma
    anchor_pts = unmoved_src[outside_mask]

    if len(anchor_pts) > 500:
        step = max(1, len(anchor_pts) // 500)
        anchor_pts = anchor_pts[::step]

    return moved_src, moved_dst, anchor_pts


def compute_grid_warp(
    photo: np.ndarray,
    moved_src: np.ndarray,
    moved_dst: np.ndarray,
    anchor_pts: np.ndarray,
    sigma: float,
    grid_n: int = 30,
) -> tuple[np.ndarray, dict]:
    """Warp an image using Gaussian-weighted grid displacement.

    Moved points drive displacement; anchor points and image corners hold
    the displacement at zero, keeping unedited regions stable.

    Args:
        photo: (H, W, 3) RGB image.
        moved_src: (N, 2) source positions of moved points.
        moved_dst: (N, 2) destination positions of moved points.
        anchor_pts: (M, 2) fixed curve points outside the sigma zone.
        sigma: Gaussian kernel standard deviation (pixels).
        grid_n: Grid resolution (NxN).

    Returns:
        (warped_image, grid_viz_dict)
    """
    h, w = photo.shape[:2]
    disp_moved = moved_dst - moved_src

    corners = np.array(
        [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float64
    )

    parts_src = [moved_src]
    parts_disp = [disp_moved]
    if len(anchor_pts) > 0:
        parts_src.append(anchor_pts)
        parts_disp.append(np.zeros((len(anchor_pts), 2)))
    parts_src.append(corners)
    parts_disp.append(np.zeros((4, 2)))

    src_all = np.concatenate(parts_src)
    disp_all = np.concatenate(parts_disp)

    grid_xs = np.linspace(0, w - 1, grid_n)
    grid_ys = np.linspace(0, h - 1, grid_n)

    grid_dx = np.zeros((grid_n, grid_n))
    grid_dy = np.zeros((grid_n, grid_n))

    for iy in range(grid_n):
        for ix in range(grid_n):
            gx, gy = grid_xs[ix], grid_ys[iy]
            dists = np.sqrt((src_all[:, 0] - gx) ** 2 + (src_all[:, 1] - gy) ** 2)
            weights = np.exp(-(dists**2) / (2 * sigma**2))
            w_sum = weights.sum()
            if w_sum > 1e-10:
                grid_dx[iy, ix] = (weights * disp_all[:, 0]).sum() / w_sum
                grid_dy[iy, ix] = (weights * disp_all[:, 1]).sum() / w_sum

    interp_dx = RegularGridInterpolator(
        (grid_ys, grid_xs), grid_dx, method="linear", bounds_error=False, fill_value=0
    )
    interp_dy = RegularGridInterpolator(
        (grid_ys, grid_xs), grid_dy, method="linear", bounds_error=False, fill_value=0
    )

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = np.column_stack([yy.ravel(), xx.ravel()])
    dx_field = interp_dx(coords).reshape(h, w).astype(np.float32)
    dy_field = interp_dy(coords).reshape(h, w).astype(np.float32)

    map_x = (xx - dx_field).astype(np.float32)
    map_y = (yy - dy_field).astype(np.float32)

    warped = cv2.remap(
        photo, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    gx_grid, gy_grid = np.meshgrid(grid_xs, grid_ys)
    grid_viz = {
        "grid_xs": grid_xs.tolist(),
        "grid_ys": grid_ys.tolist(),
        "warped_gx": (gx_grid + grid_dx).tolist(),
        "warped_gy": (gy_grid + grid_dy).tolist(),
        "grid_n": grid_n,
    }

    return warped, grid_viz
