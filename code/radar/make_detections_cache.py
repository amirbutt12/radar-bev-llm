"""Run the trained radar detector over every keyframe of a split and cache its DECODED, predicted boxes
(not ground truth) to disk: cache/detected_boxes/<tag>/<token>.npy, each (M, 7) float32
[class, x, y, w, l, yaw, score] -- the trailing score lets motion_tokens.py give the object token a confidence
feature instead of treating every detection as equally real.

Used to rebuild the motion-token cache from the detector's own predictions instead of oracle ground-truth boxes,
to test whether the pedestrian-motion finding survives a realistic end-to-end pipeline (make_motion_cache.py
--box-source detected).

    python make_detections_cache.py --version v1.0-trainval --ckpt D:/radar-bev-llm/checkpoints/radar_pillar_trainval.pt --tag trainval_det
"""
import argparse

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes

from model import RadarPillarNet
from pillar_dataset import GRID, ROOT, RadarPillarDataset, collate
from train import decode


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--min-score", type=float, default=0.3, help="pedestrians are a harder radar target and tend to score lower than cars; try a lower value to catch more of them")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    nusc = NuScenes(args.version, str(ROOT / "data" / "nuscenes"), verbose=False)
    net = RadarPillarNet().to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    net.eval()

    out_dir = ROOT / "cache" / "detected_boxes" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    n, n_with_det, n_boxes = 0, 0, 0
    for split in args.splits:
        ds = RadarPillarDataset(nusc, split)                       # every keyframe, including keyframe 0
        for i in range(0, len(ds), 16):
            items = [ds[j] for j in range(i, min(i + 16, len(ds)))]
            batch = collate(items)
            out = net(batch["points"].to(device), batch["mask"].to(device))
            preds = decode(out, GRID, min_score=args.min_score)
            for token, p in zip(batch["token"], preds):
                boxes = p[:, [0, 2, 3, 4, 5, 6, 1]].astype(np.float32)   # reorder -> [cls, x, y, w, l, yaw, score]
                np.save(out_dir / f"{token}.npy", boxes)
                n += 1
                n_with_det += int(len(boxes) > 0)
                n_boxes += len(boxes)
    print(f"cached detections for {n} keyframes to {out_dir} | {n_with_det} have >=1 detection | "
          f"{n_boxes} boxes total | min_score={args.min_score}")


if __name__ == "__main__":
    main()
