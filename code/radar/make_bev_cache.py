"""Run a trained radar network over every nuScenes-mini keyframe and cache its BEV features.

The features are the (64, 128, 128) map before the detection heads, average-pooled by `--pool` (default 4) to (64, 32, 32)
and stored as float16, one .npy file per keyframe: cache/bev_features/<tag>/<sample_token>.npy

    python make_bev_cache.py --ckpt D:/radar-bev-llm/checkpoints/radar_pillar_mini_mirror.pt --tag mini_mirror
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from nuscenes.nuscenes import NuScenes

from model import RadarPillarNet
from pillar_dataset import ROOT, RadarPillarDataset, collate


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--splits", nargs="+", default=["mini_train", "mini_val"])
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    nusc = NuScenes(args.version, str(ROOT / "data" / "nuscenes"), verbose=False)
    net = RadarPillarNet().to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()

    out_dir = ROOT / "cache" / "bev_features" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for split in args.splits:
        ds = RadarPillarDataset(nusc, split)                       # every keyframe, including keyframe 0
        for i in range(0, len(ds), 16):
            items = [ds[j] for j in range(i, min(i + 16, len(ds)))]
            batch = collate(items)
            feats = net.features(batch["points"].to(device), batch["mask"].to(device))
            feats = F.avg_pool2d(feats, args.pool).half().cpu().numpy()
            for token, f in zip(batch["token"], feats):
                np.save(out_dir / f"{token}.npy", f)
                n += 1
    print(f"cached {n} keyframes to {out_dir} | feature shape per keyframe {feats.shape[1:]}")


if __name__ == "__main__":
    main()
