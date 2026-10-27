"""Train the radar pillar network and save a checkpoint.

    python train_radar.py --mirror --epochs 25 --out D:/radar-bev-llm/checkpoints/radar_pillar_mini_mirror.pt
    python train_radar.py --version v1.0-trainval --train-split train --val-split val --mirror --epochs 8 \
        --out D:/radar-bev-llm/checkpoints/radar_pillar_trainval.pt

Items are built on the fly (not held in memory): the full train split does not fit in RAM as pre-built tensors.
"""
import argparse
from pathlib import Path

import torch
from nuscenes.nuscenes import NuScenes

import train as T
from model import RadarPillarNet
from pillar_dataset import ROOT, RadarPillarDataset


class Lazy:
    """List-like view of a dataset: supports len(), items[i], and items[i:j] (what train.py uses)."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.ds[i] for i in range(*idx.indices(len(self.ds)))]
        return self.ds[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--train-split", default="mini_train")
    ap.add_argument("--val-split", default="mini_val")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--mirror", action="store_true", help="random left-right flip of every training frame")
    ap.add_argument("--lidar-aux", action="store_true",
                     help="auxiliary training-time loss: predict LiDAR BEV occupancy from radar features (LiDAR never used at inference)")
    ap.add_argument("--lidar-weight", type=float, default=1.0, help="weight of the auxiliary LiDAR-occupancy loss")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "radar_pillar_mini.pt"))
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    nusc = NuScenes(args.version, str(ROOT / "data" / "nuscenes"), verbose=False)
    train_items = Lazy(RadarPillarDataset(nusc, args.train_split, drop_first=True, with_targets=True, with_lidar_aux=args.lidar_aux))
    val_items = Lazy(RadarPillarDataset(nusc, args.val_split, with_targets=True))
    print(f"train {len(train_items)} keyframes | val {len(val_items)} keyframes", flush=True)

    torch.manual_seed(args.seed)
    net = RadarPillarNet(lidar_aux=args.lidar_aux).to(device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    T.train(net, train_items, val_items, device, epochs=args.epochs, seed=args.seed, mirror=args.mirror, ckpt_path=args.out,
            lidar_weight=args.lidar_weight)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
