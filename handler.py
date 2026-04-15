"""RunPod serverless handler for semantic editing operations.

Provides curve-driven image warping, curve resampling, and contour extraction.
Each request is stateless - client sends full image data with each request.
"""

import base64
import traceback

import cv2
import numpy as np
import runpod

from semantic_edit.constants import SCALE
from semantic_edit.core.bezier import eval_piecewise_bezier
from semantic_edit.core.contour import extract_feather_edge, extract_last_contour
from semantic_edit.core.fitting import fit_bezier_lsq
from semantic_edit.core.warp import build_moved_and_anchor_points, compute_grid_warp


def decode_image(image_b64: str) -> np.ndarray:
    """Decode a base64 image string to a numpy array (BGR)."""
    image_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image")
    return image


def encode_image(image: np.ndarray, format: str = ".png") -> str:
    """Encode a numpy array (BGR) to a base64 string."""
    _, buffer = cv2.imencode(format, image)
    return base64.b64encode(buffer).decode("utf-8")


def handle_warp(input_data: dict) -> dict:
    """Perform curve-driven image warp.

    Input:
        image_base64: Base64-encoded image (PNG/JPEG)
        src_curves: Dict of curve_name -> {control_points: [[x,y], ...]}
        dst_curves: Dict of curve_name -> {control_points: [[x,y], ...]}
        grid_n: Grid resolution (default 55, range 5-200)
        sigma: Gaussian kernel sigma in pixels (default 30, range 5-500)

    Output:
        warped_image_base64: Base64-encoded warped image
        grid_visualization: Grid warp visualization data
        moved_points: List of [x, y] points that were moved
        anchor_points: List of [x, y] anchor points
    """
    image_b64 = input_data["image_base64"]
    src_curves = input_data["src_curves"]
    dst_curves = input_data["dst_curves"]
    grid_n = max(5, min(200, input_data.get("grid_n", 55)))
    sigma = max(5, min(500, input_data.get("sigma", 30)))

    image = decode_image(image_b64)

    # Scale sigma from workspace (512) to image scale (1024)
    sigma_scaled = sigma * SCALE

    # Build moved and anchor points
    moved_src, moved_dst, anchor_pts = build_moved_and_anchor_points(
        src_curves, dst_curves, sigma_scaled
    )

    if len(moved_src) == 0:
        # No movement detected, return original image
        return {
            "warped_image_base64": image_b64,
            "grid_visualization": None,
            "moved_points": [],
            "anchor_points": [],
        }

    # Perform the warp
    warped, grid_viz = compute_grid_warp(
        image, moved_src, moved_dst, anchor_pts, sigma_scaled, grid_n
    )

    return {
        "warped_image_base64": encode_image(warped),
        "grid_visualization": grid_viz,
        "moved_points": moved_src.tolist(),
        "anchor_points": anchor_pts.tolist(),
    }


def handle_resample(input_data: dict) -> dict:
    """Resample a Bezier curve with a new segment count.

    Input:
        control_points: [[x, y], ...] control points
        new_segments: Number of segments (1-30)

    Output:
        control_points: [[x, y], ...] new control points
    """
    cp = np.array(input_data["control_points"], dtype=np.float64)
    new_segments = max(1, min(30, input_data.get("new_segments", 3)))

    if len(cp) < 4:
        return {"control_points": cp.tolist()}

    # Evaluate the curve densely, then refit
    dense = eval_piecewise_bezier(cp, n_eval=500)
    new_cp = fit_bezier_lsq(dense, new_segments)

    return {"control_points": new_cp.tolist()}


def handle_contour(input_data: dict) -> dict:
    """Extract contour from a last mask image.

    Input:
        last_image_base64: Base64-encoded last mask image
        fe_image_base64: Optional base64-encoded feather edge image

    Output:
        contour: [[x, y], ...] contour points
        feather_edge: [[x, y], ...] feather edge points (if fe_image provided)
    """
    last_b64 = input_data["last_image_base64"]
    fe_b64 = input_data.get("fe_image_base64")

    last_image = decode_image(last_b64)
    last_rgb = cv2.cvtColor(last_image, cv2.COLOR_BGR2RGB)

    contour = extract_last_contour(last_rgb)

    result = {"contour": contour}

    if fe_b64:
        fe_image = decode_image(fe_b64)
        fe_rgb = cv2.cvtColor(fe_image, cv2.COLOR_BGR2RGB)
        feather_edge = extract_feather_edge(fe_rgb)
        if feather_edge:
            result["feather_edge"] = feather_edge

    return result


def handler(job: dict) -> dict:
    """RunPod serverless handler entry point.

    Dispatches to operation-specific handlers based on the 'operation' field.

    Supported operations:
        - warp: Curve-driven image warping
        - resample: Bezier curve resampling
        - contour: Contour extraction from mask images
    """
    try:
        job_input = job.get("input", {})
        operation = job_input.get("operation", "warp")

        if operation == "warp":
            return handle_warp(job_input)
        elif operation == "resample":
            return handle_resample(job_input)
        elif operation == "contour":
            return handle_contour(job_input)
        else:
            return {"error": f"Unknown operation: {operation}"}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
