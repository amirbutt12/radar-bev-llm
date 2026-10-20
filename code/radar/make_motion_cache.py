"""Cache explicit per-object motion tokens (see motion_tokens.py) for every keyframe of a nuScenes version:
cache/motion_tokens/<tag>/<sample_token>.npy, each (8, 9) float32.

    python make_motion_cache.py --tag mini_obj                                            # oracle: ground-truth boxes
    python make_motion_cache.py --version v1.0-trainval --tag trainval_obj_det \
        --box-source detected --detections-tag trainval_det                               # realistic: detector's own boxes
"""
import argparse

import numpy as np
from nuscenes.nuscenes import NuScenes

from bev_targets import get_boxes
from motion_tokens import get_motion_tokens
from pillar_dataset import GRID, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--box-source", choices=["gt", "detected"], default="gt")
    ap.add_argument("--detections-tag", default="", help="required if --box-source detected; cache/detected_boxes/<tag>/<token>.npy")
    args = ap.parse_args()
    if args.box_source == "detected" and not args.detections_tag:
        ap.error("--detections-tag is required with --box-source detected")

    nusc = NuScenes(args.version, str(ROOT / "data" / "nuscenes"), verbose=False)
    points_dir = ROOT / "cache" / "radar_points" / f"{args.version}_sweeps5"
    det_dir = ROOT / "cache" / "detected_boxes" / args.detections_tag if args.box_source == "detected" else None
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
        boxes = get_boxes(nusc, tok, GRID) if args.box_source == "gt" else np.load(det_dir / f"{tok}.npy")
        feats = get_motion_tokens(boxes, points)
        np.save(out_dir / f"{tok}.npy", feats)
        n_with_obj += int((feats[:, 0] + feats[:, 1] > 0).any())
    print(f"cached {len(tokens)} keyframes ({args.box_source} boxes) to {out_dir} | {n_with_obj} keyframes have at least one object slot filled")


if __name__ == "__main__":
    main()
