"""Contour extraction from last mask and feather edge images."""

from __future__ import annotations

import cv2
import numpy as np


def extract_last_contour(image_rgb: np.ndarray) -> list[list[int]]:
    """Extract the largest interior contour from a last mask image.

    Handles both conventions:
    - White shape on black background (e.g., solid last masks)
    - Dark shape on white background

    Tries both threshold polarities and picks the contour that represents
    the actual shape (not the background).

    Args:
        image_rgb: (H, W, 3) RGB image.

    Returns:
        List of [x, y] points forming the contour, or empty list.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    total_area = h * w

    best_contour = None
    best_area = 0

    for thresh_type in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]:
        _, mask = cv2.threshold(gray, 127, 255, thresh_type)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            area = cv2.contourArea(c)
            # Skip contours that are nearly the full image (background)
            if area > total_area * 0.9:
                continue
            # Skip tiny noise contours
            if area < total_area * 0.01:
                continue
            if area > best_area:
                best_area = area
                best_contour = c

    if best_contour is None:
        return []
    return best_contour.reshape(-1, 2).tolist()


def extract_feather_edge(
    image_rgb: np.ndarray, max_points: int = 200
) -> list[list[int]] | None:
    """Extract feather edge points from a red-channel annotation image.

    Red pixels (R > 128, G < 100, B < 100) are treated as the feather edge.

    Args:
        image_rgb: (H, W, 3) RGB image.
        max_points: Subsample to at most this many points.

    Returns:
        List of [x, y] points sorted by x, or None if no red pixels found.
    """
    red = (
        (image_rgb[:, :, 0] > 128)
        & (image_rgb[:, :, 1] < 100)
        & (image_rgb[:, :, 2] < 100)
    )
    if red.sum() == 0:
        return None
    ys, xs = np.where(red)
    order = np.argsort(xs)
    pts = np.column_stack([xs[order], ys[order]])
    if len(pts) > max_points:
        step = max(1, len(pts) // max_points)
        pts = pts[::step]
    return pts.tolist()
