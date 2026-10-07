"""Cache explicit per-object motion tokens (see motion_tokens.py) for every nuScenes-mini keyframe:
cache/motion_tokens/<tag>/<sample_token>.npy, each (8, 8) float32.

    python make_motion_cache.py --tag mini_obj
"""
import argparse

import numpy as np
from nuscenes.nuscenes import NuScenes

from motion_tokens import get_motion_tokens
from pillar_dataset import GRID, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    args = ap.parse_args()

    nusc = NuScenes(args.version, str(ROOT / "data" / "nuscenes"), verbose=False)
    points_dir = ROOT / "cache" / "radar_points" / f"{args.version}_sweeps5"
    out_dir = ROOT / "cache" / "motion_tokens" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens = []
    for s in nusc.scene:
        t = s["first_sample_token"]
        while t:
            tokens.append(t)
            t = nusc.get("sample", t)["next"]

    n_with_obj = 0
    for tok in tokens:
        points = np.load(points_dir / f"{tok}.npy")
        feats = get_motion_tokens(nusc, tok, GRID, points)
        np.save(out_dir / f"{tok}.npy", feats)
        n_with_obj += int((feats[:, 0] + feats[:, 1] > 0).any())
    print(f"cached {len(tokens)} keyframes to {out_dir} | {n_with_obj} keyframes have at least one object slot filled")


if __name__ == "__main__":
    main()
