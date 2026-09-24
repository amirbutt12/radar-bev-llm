"""Losses, decoding, centre-distance AP and a small training loop for RadarPillarNet."""
import time

import numpy as np
import torch
import torch.nn.functional as F

from pillar_dataset import GRID, collate


def focal_loss(logits, gt):
    """CenterNet focal loss. gt is the Gaussian heatmap: exactly 1 at object centres, <1 around them."""
    p = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    pos_loss = torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * (1 - pos)     # squares near a centre are penalised less
    return -(pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1)


def reg_loss(pred, gt, mask):
    """L1 on the 6 regression values, only at object-centre squares."""
    m = mask[:, None]
    return F.l1_loss(pred * m, gt * m, reduction="sum") / (m.sum() * pred.shape[1]).clamp(min=1)


def compute_loss(out, batch):
    hm, rg = focal_loss(out["hm"], batch["heatmap"]), reg_loss(out["reg"], batch["reg"], batch["reg_mask"])
    return hm + rg, hm.detach(), rg.detach()


@torch.no_grad()
def decode(out, grid=GRID, k=100, min_score=0.05):
    """Network output -> per frame an array (K, 7) of [class, score, x, y, w, l, yaw] (ego frame, metres)."""
    hm = out["hm"].sigmoid()
    hm = hm * (F.max_pool2d(hm, 3, 1, 1) == hm)                          # keep only local peaks
    B, C, H, W = hm.shape
    scores, idx = hm.view(B, -1).topk(k)
    results = []
    for b in range(B):
        cls, ix, iy = idx[b] // (H * W), (idx[b] % (H * W)) // W, idx[b] % W
        keep = scores[b] >= min_score
        cls, ix, iy, sc = cls[keep], ix[keep], iy[keep], scores[b][keep]
        r = out["reg"][b][:, ix, iy]                                     # (6, K)
        x = grid["x0"] + (ix + 0.5 + r[0]) * grid["cell"]
        y = grid["y0"] + (iy + 0.5 + r[1]) * grid["cell"]
        yaw = torch.atan2(r[4], r[5])
        results.append(torch.stack([cls.float(), sc, x, y, r[2].exp(), r[3].exp(), yaw], 1).cpu().numpy())
    return results


def average_precision(preds, gts, cls, thr):
    """AP for one class, a detection is right if its centre is within `thr` metres of a not-yet-matched true centre."""
    truth = [g[g[:, 0] == cls][:, 1:3] for g in gts]
    n_gt = sum(len(t) for t in truth)
    if n_gt == 0:
        return float("nan")
    dets = sorted(((p[1], f, p[2], p[3]) for f, fp in enumerate(preds) for p in fp if p[0] == cls), reverse=True)
    used = [np.zeros(len(t), bool) for t in truth]
    tp = np.zeros(len(dets))
    for i, (_, f, x, y) in enumerate(dets):
        if len(truth[f]):
            d = np.hypot(truth[f][:, 0] - x, truth[f][:, 1] - y)
            d[used[f]] = np.inf
            j = int(np.argmin(d))
            if d[j] < thr:
                used[f][j], tp[i] = True, 1
    tp_c, fp_c = np.cumsum(tp), np.cumsum(1 - tp)
    recall, precision = tp_c / n_gt, tp_c / np.maximum(tp_c + fp_c, 1e-9)
    precision = np.maximum.accumulate(precision[::-1])[::-1]              # make precision non-increasing
    return float(np.sum(np.diff(np.concatenate([[0], recall])) * precision)) if len(dets) else 0.0


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, items, device, batch_size=8, thr=2.0):
    """Mean loss and per-class AP (centre distance < thr metres) over a list of dataset items."""
    model.eval()
    losses, preds, gts = [], [], []
    for i in range(0, len(items), batch_size):
        batch = to_device(collate(items[i:i + batch_size]), device)
        out = model(batch["points"], batch["mask"])
        losses.append([t.item() for t in compute_loss(out, batch)])
        preds += decode(out)
        gts += [b.numpy() for b in batch["boxes"]]
    total, hm, rg = np.mean(losses, 0)
    return dict(loss=total, hm=hm, reg=rg, ap_car=average_precision(preds, gts, 0, thr), ap_ped=average_precision(preds, gts, 1, thr))


def train(model, train_items, val_items, device, epochs=20, batch_size=8, lr=2e-3, seed=0, log=print):
    """Plain AdamW + cosine schedule. Returns one dict per epoch."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * int(np.ceil(len(train_items) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    history = []
    for epoch in range(epochs):
        model.train()
        t0, run = time.time(), []
        order = rng.permutation(len(train_items))
        for i in range(0, len(order), batch_size):
            batch = to_device(collate([train_items[j] for j in order[i:i + batch_size]]), device)
            loss, hm, rg = compute_loss(model(batch["points"], batch["mask"]), batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            run.append([loss.item(), hm.item(), rg.item()])
        tr = np.mean(run, 0)
        va = evaluate(model, val_items, device, batch_size)
        history.append(dict(epoch=epoch + 1, train_loss=tr[0], train_hm=tr[1], train_reg=tr[2], **{f"val_{k}": v for k, v in va.items()}))
        log(f"epoch {epoch + 1:2d}/{epochs} | train loss {tr[0]:.3f} (hm {tr[1]:.3f} reg {tr[2]:.3f}) | val loss {va['loss']:.3f} "
            f"| val AP@2m car {va['ap_car']:.3f} ped {va['ap_ped']:.3f} | {time.time() - t0:.0f}s")
    return history
