"""Train the LiDAR pillar network (the distillation teacher) and save a checkpoint.

Never run at inference -- this exists purely to produce a stronger BEV feature map for the radar detector to
distill against (see distill_radar.py). Same task, targets, and evaluation as the radar detector, so its AP is
directly comparable and establishes the ceiling the student is reaching toward.

    python train_lidar.py --version v1.0-trainval --train-split train --val-split val --mirror --epochs 8 \
        --out D:/radar-bev-llm/checkpoints/lidar_pillar_trainval.pt

LiDAR keyframes carry ~30-100x more points than radar (30k+ vs a few hundred); --bs defaults lower than
train_radar.py's to fit in memory, and the first epoch is slower while the point cache is built.
"""
import argparse
from pathlib import Path

import torch
from nuscenes.nuscenes import NuScenes

import train as T
from model import LidarPillarNet
from pillar_dataset import ROOT, LidarPillarDataset
from train_radar import Lazy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--train-split", default="mini_train")
    ap.add_argument("--val-split", default="mini_val")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--mirror", action="store_true", help="random left-right flip of every training frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "lidar_pillar_mini.pt"))
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    nusc = NuScenes(args.version, str(ROOT / "data" / "nuscenes"), verbose=False)
    train_items = Lazy(LidarPillarDataset(nusc, args.train_split, drop_first=True, with_targets=True))
    val_items = Lazy(LidarPillarDataset(nusc, args.val_split, with_targets=True))
    print(f"train {len(train_items)} keyframes | val {len(val_items)} keyframes", flush=True)

    torch.manual_seed(args.seed)
    net = LidarPillarNet().to(device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    T.train(net, train_items, val_items, device, epochs=args.epochs, batch_size=args.bs, seed=args.seed,
            mirror=args.mirror, ckpt_path=args.out)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
