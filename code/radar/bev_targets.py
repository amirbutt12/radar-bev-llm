"""Ground-truth BEV targets for the radar detection proxy (CenterNet-style, on the same grid as the radar BEV).

Per keyframe:
    boxes    (M, 6)              [class, x, y, w, l, yaw] in the keyframe ego frame (x forward, y left, metres/radians)
    heatmap  (2, nx, ny)         per class: 1.0 at each object's centre cell, decaying like a Gaussian around it
    reg      (6, nx, ny)         at each centre cell: [dx, dy (centre offset inside the cell, in cells), log w, log l, sin yaw, cos yaw]
    reg_mask (nx, ny)            1 at centre cells (where `reg` is valid), 0 elsewhere
"""
import numpy as np
from pyquaternion import Quaternion

CLASSES = ["car", "pedestrian"]
PEDESTRIAN_KINDS = {"adult", "child", "construction_worker", "police_officer"}


def category_to_class(name):
    """nuScenes category name -> index in CLASSES, or None if we do not use it."""
    if name == "vehicle.car":
        return 0
    parts = name.split(".")
    if parts[:2] == ["human", "pedestrian"] and len(parts) > 2 and parts[2] in PEDESTRIAN_KINDS:
        return 1
    return None


def get_boxes(nusc, sample_token, grid):
    """(M, 6) [class, x, y, w, l, yaw] of the cars / pedestrians whose centre lies inside the grid, in the ego frame."""
    sample = nusc.get("sample", sample_token)
    ego = nusc.get("ego_pose", nusc.get("sample_data", sample["data"]["LIDAR_TOP"])["ego_pose_token"])
    q_ego = Quaternion(ego["rotation"])
    t_ego = np.array(ego["translation"])
    x_max = grid["x0"] + grid["nx"] * grid["cell"]
    y_max = grid["y0"] + grid["ny"] * grid["cell"]

    rows = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        cls = category_to_class(ann["category_name"])
        if cls is None:
            continue
        x, y, _ = q_ego.inverse.rotate(np.array(ann["translation"]) - t_ego)
        if not (grid["x0"] <= x < x_max and grid["y0"] <= y < y_max):
            continue
        yaw = (q_ego.inverse * Quaternion(ann["rotation"])).yaw_pitch_roll[0]
        w, l, _h = ann["size"]                         # nuScenes size = [width, length, height]
        rows.append([cls, x, y, w, l, yaw])
    return np.array(rows, dtype=np.float32).reshape(-1, 6)


def make_targets(boxes, grid):
    """boxes (M, 6) -> heatmap (2, nx, ny), reg (6, nx, ny), reg_mask (nx, ny). Grid layout as in pillarize()."""
    nx, ny, cell = grid["nx"], grid["ny"], grid["cell"]
    heatmap = np.zeros((len(CLASSES), nx, ny), np.float32)
    reg = np.zeros((6, nx, ny), np.float32)
    reg_mask = np.zeros((nx, ny), np.float32)

    for cls, x, y, w, l, yaw in boxes:
        fx, fy = (x - grid["x0"]) / cell, (y - grid["y0"]) / cell
        ix, iy = int(np.floor(fx)), int(np.floor(fy))
        if not (0 <= ix < nx and 0 <= iy < ny):
            continue
        sigma = max(1.0, (w + l) / 2 / cell / 4)       # bigger objects get a wider blob; never thinner than 1 cell
        r = int(np.ceil(3 * sigma))
        xs = np.arange(max(ix - r, 0), min(ix + r + 1, nx))
        ys = np.arange(max(iy - r, 0), min(iy + r + 1, ny))
        blob = np.exp(-((xs[:, None] - ix) ** 2 + (ys[None, :] - iy) ** 2) / (2 * sigma ** 2))
        window = heatmap[int(cls), xs[0]:xs[-1] + 1, ys[0]:ys[-1] + 1]
        np.maximum(window, blob, out=window)
        reg[:, ix, iy] = [fx - (ix + 0.5), fy - (iy + 0.5), np.log(w), np.log(l), np.sin(yaw), np.cos(yaw)]
        reg_mask[ix, iy] = 1.0
    return heatmap, reg, reg_mask
