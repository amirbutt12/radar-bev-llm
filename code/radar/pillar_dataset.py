"""nuScenes radar -> multi-sweep points -> BEV pillars, as a torch Dataset with an on-disk cache.

Per keyframe: the last `n_sweeps` sweeps of all 5 radars are moved into the keyframe's ego frame
(x forward / y left) and stored as an (N, 8) float32 array [x, y, z, vx, vy, v_r, rcs, dt].
Raw points are cached (the learned pillar encoder needs them); the hand-crafted BEV map is cheap
and is recomputed on the fly.

    ds = RadarPillarDataset(nusc, split="mini_train")
    ds.prepare()                                   # optional: fill the cache up front
    loader = DataLoader(ds, batch_size=8, collate_fn=collate)
"""
import os
import time
from pathlib import Path

import numpy as np
import torch
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
from torch.utils.data import Dataset

from bev_targets import get_boxes, make_targets

ROOT = Path(r"D:\radar-bev-llm")
RADARS = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]
COLS = ["x", "y", "z", "vx", "vy", "v_r", "rcs", "dt"]
GRID = dict(x0=-51.2, y0=-51.2, cell=0.8, nx=128, ny=128)
CHANNELS = ["count", "mean_vx", "mean_vy", "mean_v_r", "mean_rcs", "max_rcs", "mean_dt"]


def pose(rec):
    """(rotation matrix, translation) of an ego_pose / calibrated_sensor record."""
    return Quaternion(rec["rotation"]).rotation_matrix, np.array(rec["translation"])


def load_radar_sweeps(nusc, sample, sensor, n_sweeps=5):
    """(N, 8) points [x, y, z, vx, vy, v_r, rcs, dt] of one radar, in the keyframe's ego frame."""
    dataroot = Path(nusc.dataroot)
    key_sd = nusc.get("sample_data", sample["data"][sensor])
    R_ke, t_ke = pose(nusc.get("ego_pose", key_sd["ego_pose_token"]))
    sd, out = key_sd, []
    for _ in range(n_sweeps):
        p = RadarPointCloud.from_file(str(dataroot / sd["filename"])).points   # (18, N), sensor frame
        R_cs, t_cs = pose(nusc.get("calibrated_sensor", sd["calibrated_sensor_token"]))
        R_ge, t_ge = pose(nusc.get("ego_pose", sd["ego_pose_token"]))

        xyz = R_cs @ p[:3] + t_cs[:, None]              # sensor -> ego(t_i)
        xyz = R_ge @ xyz + t_ge[:, None]                # -> global
        xyz = R_ke.T @ (xyz - t_ke[:, None])            # -> ego(keyframe)

        R_tot = R_ke.T @ R_ge @ R_cs                    # rotation only, for vectors
        v = R_tot @ np.stack([p[8], p[9], np.zeros(p.shape[1])])   # vx_comp, vy_comp

        los = p[:2] / np.maximum(np.linalg.norm(p[:2], axis=0), 1e-6)
        v_r = (p[[8, 9]] * los).sum(0)                  # radial Doppler, invariant to the frame
        dt = np.full(p.shape[1], (key_sd["timestamp"] - sd["timestamp"]) * 1e-6)

        out.append(np.stack([xyz[0], xyz[1], xyz[2], v[0], v[1], v_r, p[5], dt], axis=1))
        if not sd["prev"]:
            break
        sd = nusc.get("sample_data", sd["prev"])
    return np.concatenate(out, axis=0).astype(np.float32)


def load_keyframe_radar(nusc, sample, n_sweeps=5):
    """All 5 radars merged -> (N, 8)."""
    return np.concatenate([load_radar_sweeps(nusc, sample, s, n_sweeps) for s in RADARS], axis=0)


def pillarize(pts, g=GRID):
    """(N, 8) ego-frame points -> (len(CHANNELS), nx, ny) BEV map. Dim 1 = x (forward), dim 2 = y (left)."""
    ix = np.floor((pts[:, 0] - g["x0"]) / g["cell"]).astype(int)
    iy = np.floor((pts[:, 1] - g["y0"]) / g["cell"]).astype(int)
    keep = (ix >= 0) & (ix < g["nx"]) & (iy >= 0) & (iy < g["ny"])
    pts, ix, iy = pts[keep], ix[keep], iy[keep]
    flat = ix * g["ny"] + iy
    size = g["nx"] * g["ny"]

    count = np.bincount(flat, minlength=size).astype(np.float32)
    denom = np.maximum(count, 1.0)
    mean = lambda col: np.bincount(flat, weights=pts[:, col], minlength=size) / denom
    max_rcs = np.full(size, -np.inf, dtype=np.float32)
    np.maximum.at(max_rcs, flat, pts[:, 6])
    max_rcs[~np.isfinite(max_rcs)] = 0.0

    bev = np.stack([count, mean(3), mean(4), mean(5), mean(6), max_rcs, mean(7)])
    return bev.reshape(len(CHANNELS), g["nx"], g["ny"]).astype(np.float32)


class RadarPillarDataset(Dataset):
    """One item per keyframe of the scenes in `split` (a nuScenes split name, e.g. 'mini_train').

    drop_first: skip the first keyframe of every scene (it has no earlier sweeps, so ~5x fewer points).
    with_targets: also return the ground-truth BEV targets (see bev_targets.py).
    """

    def __init__(self, nusc, split="mini_train", n_sweeps=5, cache_dir=ROOT / "cache" / "radar_points", grid=GRID,
                 drop_first=False, with_targets=False):
        self.nusc, self.n_sweeps, self.grid, self.with_targets = nusc, n_sweeps, grid, with_targets
        scene_names = set(create_splits_scenes()[split])
        self.tokens = []
        for scene in nusc.scene:
            if scene["name"] not in scene_names:
                continue
            token = scene["first_sample_token"]
            if drop_first:
                token = nusc.get("sample", token)["next"]
            while token:
                self.tokens.append(token)
                token = nusc.get("sample", token)["next"]
        self.cache_dir = Path(cache_dir) / f"{nusc.version}_sweeps{n_sweeps}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.tokens)

    def points(self, token):
        """(N, 8) points of a keyframe; computed once, then read from the cache."""
        path = self.cache_dir / f"{token}.npy"
        if path.exists():
            return np.load(path)
        pts = load_keyframe_radar(self.nusc, self.nusc.get("sample", token), self.n_sweeps)
        tmp = path.with_name(f"{token}.{os.getpid()}.tmp.npy")   # write-then-rename: safe with DataLoader workers
        np.save(tmp, pts)
        os.replace(tmp, path)
        return pts

    def prepare(self):
        """Fill the cache for every keyframe. Returns seconds spent."""
        t0 = time.time()
        for token in self.tokens:
            self.points(token)
        return time.time() - t0

    def __getitem__(self, i):
        token = self.tokens[i]
        pts = self.points(token)
        item = dict(token=token, points=torch.from_numpy(pts), bev=torch.from_numpy(pillarize(pts, self.grid)))
        if self.with_targets:
            boxes = get_boxes(self.nusc, token, self.grid)
            heatmap, reg, reg_mask = make_targets(boxes, self.grid)
            item.update(boxes=torch.from_numpy(boxes), heatmap=torch.from_numpy(heatmap),
                        reg=torch.from_numpy(reg), reg_mask=torch.from_numpy(reg_mask))
        return item


def collate(batch):
    """Point counts differ per keyframe: pad to the largest and return a validity mask."""
    n_max = max(len(b["points"]) for b in batch)
    points = torch.zeros(len(batch), n_max, batch[0]["points"].shape[1])
    mask = torch.zeros(len(batch), n_max, dtype=torch.bool)
    for k, b in enumerate(batch):
        n = len(b["points"])
        points[k, :n] = b["points"]
        mask[k, :n] = True
    out = dict(token=[b["token"] for b in batch], points=points, mask=mask, bev=torch.stack([b["bev"] for b in batch]))
    if "heatmap" in batch[0]:
        for key in ("heatmap", "reg", "reg_mask"):
            out[key] = torch.stack([b[key] for b in batch])
        out["boxes"] = [b["boxes"] for b in batch]          # (M, 6) per keyframe, M differs -> keep as a list
    return out
