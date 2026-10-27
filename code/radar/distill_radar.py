"""Distill the LiDAR teacher (train_lidar.py) into the radar detector: CRKD-style feature + response distillation,
on top of the normal detection loss. The teacher is frozen and only ever used during this training run -- at
inference the student is a plain RadarPillarNet, radar-only, identical in shape to every other checkpoint in this
project.

Two extra loss terms, both comparing the student's outputs to the (frozen) teacher's outputs on the SAME keyframe:
  - feature distillation: cosine distance between the two BEV feature maps (both (B, 64, 128, 128) -- the two
    networks share the same backbone shape, so no projection/adapter layer is needed to compare them).
  - response distillation: BCE between the student's heatmap logits and the teacher's (sigmoid) heatmap as a soft
    target -- a richer training signal than the hard 0/1 box labels alone.

    python distill_radar.py --version v1.0-trainval --train-split train --val-split val --mirror --epochs 8 \
        --teacher-ckpt D:/radar-bev-llm/checkpoints/lidar_pillar_trainval.pt \
        --out D:/radar-bev-llm/checkpoints/radar_pillar_trainval_distill.pt
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from nuscenes.nuscenes import NuScenes

from model import LidarPillarNet, RadarPillarNet
from pillar_dataset import GRID, LidarPillarDataset, RadarPillarDataset, ROOT, collate
from train import compute_loss, evaluate, to_device
from train_radar import Lazy


class DistillDataset:
    """Pairs a RadarPillarDataset and a LidarPillarDataset item-for-item, same split/drop_first -> identical,
    identically-ordered token lists, so index i is the same keyframe in both."""

    def __init__(self, radar_ds, lidar_ds):
        assert len(radar_ds) == len(lidar_ds)
        self.radar_ds, self.lidar_ds = radar_ds, lidar_ds

    def __len__(self):
        return len(self.radar_ds)

    def __getitem__(self, i):
        r, l = self.radar_ds[i], self.lidar_ds[i]
        assert r["token"] == l["token"]
        return dict(token=r["token"], radar_points=r["points"], lidar_points=l["points"],
                    heatmap=r["heatmap"], reg=r["reg"], reg_mask=r["reg_mask"], boxes=r["boxes"])


def distill_collate(batch):
    radar_batch = collate([dict(token=b["token"], points=b["radar_points"], heatmap=b["heatmap"], reg=b["reg"],
                                 reg_mask=b["reg_mask"], boxes=b["boxes"]) for b in batch])
    lidar_batch = collate([dict(token=b["token"], points=b["lidar_points"]) for b in batch])
    radar_batch["lidar_points"], radar_batch["lidar_mask"] = lidar_batch["points"], lidar_batch["mask"]
    return radar_batch


def mirror_joint_batch(batch, flip):
    f = flip.to(batch["points"].device)
    batch["points"][f, :, 1] *= -1
    batch["points"][f, :, 4] *= -1
    batch["lidar_points"][f, :, 1] *= -1
    batch["heatmap"][f] = batch["heatmap"][f].flip(-1)
    batch["reg"][f] = batch["reg"][f].flip(-1)
    batch["reg"][f, 1] *= -1
    batch["reg"][f, 4] *= -1
    batch["reg_mask"][f] = batch["reg_mask"][f].flip(-1)
    return batch


def distill_losses(student_feat, student_hm, teacher_feat, teacher_hm):
    """(feature distillation, response distillation), both comparing student to the frozen teacher."""
    feat_loss = 1 - F.cosine_similarity(student_feat, teacher_feat, dim=1).mean()
    resp_loss = F.binary_cross_entropy_with_logits(student_hm, teacher_hm.sigmoid())
    return feat_loss, resp_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--train-split", default="mini_train")
    ap.add_argument("--val-split", default="mini_val")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--feat-weight", type=float, default=1.0)
    ap.add_argument("--resp-weight", type=float, default=1.0)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--teacher-ckpt", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    nusc = NuScenes(args.version, str(ROOT / "data" / "nuscenes"), verbose=False)

    radar_train = RadarPillarDataset(nusc, args.train_split, drop_first=True, with_targets=True)
    lidar_train = LidarPillarDataset(nusc, args.train_split, drop_first=True, with_targets=False)
    train_items = Lazy(DistillDataset(radar_train, lidar_train))
    val_items = Lazy(RadarPillarDataset(nusc, args.val_split, with_targets=True))     # student-only eval, as usual
    print(f"train {len(train_items)} keyframes | val {len(val_items)} keyframes", flush=True)

    torch.manual_seed(args.seed)
    student = RadarPillarNet().to(device)
    teacher = LidarPillarNet().to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device, weights_only=True))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = args.epochs * int(np.ceil(len(train_items) / args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
    rng = np.random.default_rng(args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        student.train()
        t0, run = time.time(), []
        order = rng.permutation(len(train_items))
        for i in range(0, len(order), args.bs):
            batch = to_device(distill_collate([train_items[j] for j in order[i:i + args.bs]]), device)
            if args.mirror:
                batch = mirror_joint_batch(batch, torch.rand(len(batch["token"])) < 0.5)

            with torch.no_grad():
                teacher_feat = teacher.features(batch["lidar_points"], batch["lidar_mask"])
                teacher_hm = teacher.hm_head(teacher_feat)

            student_feat = student.features(batch["points"], batch["mask"])
            out = dict(hm=student.hm_head(student_feat), reg=student.reg_head(student_feat))
            det_loss, hm_l, rg_l = compute_loss(out, batch)
            feat_loss, resp_loss = distill_losses(student_feat, out["hm"], teacher_feat, teacher_hm)
            loss = det_loss + args.feat_weight * feat_loss + args.resp_weight * resp_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
            opt.step()
            sched.step()
            run.append([loss.item(), hm_l.item(), rg_l.item(), feat_loss.item(), resp_loss.item()])

        tr = np.mean(run, 0)
        va = evaluate(student, val_items, device, args.bs)
        print(f"epoch {epoch + 1:2d}/{args.epochs} | train loss {tr[0]:.3f} (hm {tr[1]:.3f} reg {tr[2]:.3f} "
              f"feat {tr[3]:.3f} resp {tr[4]:.3f}) | val AP@2m car {va['ap_car']:.3f} ped {va['ap_ped']:.3f} "
              f"| {time.time() - t0:.0f}s", flush=True)
        torch.save(student.state_dict(), args.out)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
