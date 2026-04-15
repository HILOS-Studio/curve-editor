"""Shared constants for the semantic edit module."""

WORK_SIZE = 512
CANVAS_SIZE = 1024
SCALE = CANVAS_SIZE / WORK_SIZE  # 2.0

CURVE_NAMES = [
    "collar",
    "upper_heel",
    "upper_tongue",
    "bite",
    "ground",
    "heel",
    "toe",
]

# Junction definitions: each junction is a list of (curve_name, "first"|"last").
# "first" = index 0 of control_points, "last" = last index.
# Curves sharing a junction have their endpoints snapped to a common position.
JUNCTIONS = [
    # J1: upper_heel top → collar left
    [("upper_heel", "last"), ("collar", "first")],
    # J2: collar right → upper_tongue top
    [("collar", "last"), ("upper_tongue", "first")],
    # J3: upper_tongue bottom + bite right + toe top
    [("upper_tongue", "last"), ("bite", "last"), ("toe", "first")],
    # J4: toe bottom → ground right
    [("toe", "last"), ("ground", "last")],
    # J5: ground left → heel bottom
    [("ground", "first"), ("heel", "last")],
    # J6: bite left + upper_heel bottom + heel top
    [("bite", "first"), ("upper_heel", "first"), ("heel", "first")],
]
