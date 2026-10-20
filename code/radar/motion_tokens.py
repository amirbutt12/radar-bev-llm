"""Explicit per-object motion tokens: one token per nearby car/pedestrian, carrying its position and its RADAR-MEASURED
velocity (mean vx/vy of the radar points that fall inside its box), instead of asking the model to read speed off the
blurry, pooled dense BEV map.

`boxes` is passed in rather than fetched internally, so the same function works for either an "oracle" object list
(ground-truth boxes, from bev_targets.get_boxes -- isolates whether explicit structure helps at all) or a realistic
one (the project's own trained detector's predictions, from make_detections_cache.py -- the real end-to-end test).

Feature per slot (8 numbers): [is_car, is_pedestrian, x/50, y/50, vx/20, vy/20, speed/20, has_radar_points]
Slots are split evenly between classes (nearest MAX_OBJECTS//2 cars + nearest MAX_OBJECTS//2 pedestrians), not just
"nearest MAX_OBJECTS overall" -- with detected (not ground-truth) boxes, cars typically outnumber pedestrian
detections several-to-one, so a shared nearest-N pool lets cars crowd pedestrians out of their slots entirely even
when a pedestrian WAS correctly detected. A padding slot is all zeros (is_car == is_pedestrian == 0 marks "no object
here").
"""
import numpy as np

MAX_OBJECTS = 8
DIM = 8


def get_motion_tokens(boxes, points, max_objects=MAX_OBJECTS):
    """boxes: (M, 6) [cls, x, y, w, l, yaw], either ground-truth or detected.
    points: (N, 8) cached radar points [x, y, z, vx, vy, v_r, rcs, dt] for this keyframe -> (max_objects, 8) float32."""
    if len(boxes) == 0:
        return np.zeros((max_objects, DIM), np.float32)

    half = max_objects // 2
    picked = []
    for cls in (0, 1):                                                 # car slots first, then pedestrian slots
        cls_boxes = boxes[boxes[:, 0] == cls]
        if len(cls_boxes):
            order = np.argsort(np.hypot(cls_boxes[:, 1], cls_boxes[:, 2]))
            picked.append(cls_boxes[order][:half])
    boxes = np.concatenate(picked, axis=0) if picked else boxes[:0]

    out = np.zeros((max_objects, DIM), np.float32)
    for i, (cls, x, y, w, l, _yaw) in enumerate(boxes):
        radius = max(w, l) / 2 + 1.0                                  # a little slack around the box for sparse radar
        inside = np.hypot(points[:, 0] - x, points[:, 1] - y) < radius
        has_pts = bool(inside.any())
        vx, vy = (points[inside, 3].mean(), points[inside, 4].mean()) if has_pts else (0.0, 0.0)
        out[i] = [cls == 0, cls == 1, x / 50, y / 50, vx / 20, vy / 20, np.hypot(vx, vy) / 20, float(has_pts)]
    return out.astype(np.float32)
