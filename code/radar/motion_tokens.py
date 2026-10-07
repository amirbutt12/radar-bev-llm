"""Explicit per-object motion tokens: one token per nearby car/pedestrian, carrying its position and its RADAR-MEASURED
velocity (mean vx/vy of the radar points that fall inside its box), instead of asking the model to read speed off the
blurry, pooled dense BEV map. Uses ground-truth boxes (an "oracle" object list) so this first pass isolates whether
explicit structure helps at all, before adding detector noise on top.

Feature per slot (8 numbers): [is_car, is_pedestrian, x/50, y/50, vx/20, vy/20, speed/20, has_radar_points]
Objects are sorted by distance to the car (closest first) and padded/truncated to MAX_OBJECTS; a padding slot is all
zeros (is_car == is_pedestrian == 0 marks "no object here").
"""
import numpy as np

from bev_targets import get_boxes

MAX_OBJECTS = 8
DIM = 8


def get_motion_tokens(nusc, sample_token, grid, points, max_objects=MAX_OBJECTS):
    """points: (N, 8) cached radar points [x, y, z, vx, vy, v_r, rcs, dt] for this keyframe -> (max_objects, 8) float32."""
    boxes = get_boxes(nusc, sample_token, grid)                       # (M, 6) [cls, x, y, w, l, yaw]
    if len(boxes) == 0:
        return np.zeros((max_objects, DIM), np.float32)

    order = np.argsort(np.hypot(boxes[:, 1], boxes[:, 2]))            # nearest object first
    boxes = boxes[order][:max_objects]

    out = np.zeros((max_objects, DIM), np.float32)
    for i, (cls, x, y, w, l, _yaw) in enumerate(boxes):
        radius = max(w, l) / 2 + 1.0                                  # a little slack around the box for sparse radar
        inside = np.hypot(points[:, 0] - x, points[:, 1] - y) < radius
        has_pts = bool(inside.any())
        vx, vy = (points[inside, 3].mean(), points[inside, 4].mean()) if has_pts else (0.0, 0.0)
        out[i] = [cls == 0, cls == 1, x / 50, y / 50, vx / 20, vy / 20, np.hypot(vx, vy) / 20, float(has_pts)]
    return out.astype(np.float32)
