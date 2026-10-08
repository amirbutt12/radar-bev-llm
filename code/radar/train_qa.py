"""Train the radar-language model on nuScenes-QA (mini keyframes) and measure exact-match answer accuracy.

    python train_qa.py --name radar --feat-tag mini_mirror --epochs 5
    python train_qa.py --name blind --feat-tag mini_mirror --epochs 5 --no-vision     # same model, radar tokens removed
"""
import argparse
import collections
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from qa_data import ROOT, QADataset, collate_qa
from qa_model import RadarLLM


def majority_per_type(train_ds, val_ds):
    """Per question type: accuracy of always answering with the most common TRAINING answer of that type."""
    common = {}
    by_type = collections.defaultdict(collections.Counter)
    for q in train_ds.questions:
        by_type[q["template_type"]][q["answer"]] += 1
    for t, c in by_type.items():
        common[t] = c.most_common(1)[0][0]
    hits = collections.defaultdict(list)
    for q in val_ds.questions:
        hits[q["template_type"]].append(common.get(q["template_type"]) == q["answer"])
    return {t: float(np.mean(v)) for t, v in hits.items()}


def majority_baseline(train_ds, val_ds):
    """Answer every question with the most common training answer of its template type (no radar, no language model)."""
    per_type = majority_per_type(train_ds, val_ds)
    by_type = collections.Counter(q["template_type"] for q in val_ds.questions)
    n = sum(by_type.values())
    return sum(per_type[t] * c for t, c in by_type.items()) / n


@torch.no_grad()
def evaluate(model, ds, batch_size=32, shuffle_feats=False, zero=(), seed=0):
    """Exact-match accuracy overall and per question type.
    shuffle_feats=True gives every question ALL its sensor features from a random OTHER keyframe: the "wrong frame" control.
    zero=('feats','cam_feats','motion_feats', ...) blanks out just those modalities (sets them to 0) so the model has no
    information from them at all, everything else kept correct: isolates whether that one modality is being used."""
    model.eval()
    rng = np.random.default_rng(seed)
    tokens = sorted(ds.feats)
    preds, hits = [], []
    for i in range(0, len(ds), batch_size):
        items = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
        batch = collate_qa(items)
        if shuffle_feats:
            other = [tokens[rng.integers(len(tokens))] for _ in items]
            batch["feats"] = torch.stack([ds.feats[t] for t in other])
            if ds.cam_feats is not None:
                batch["cam_feats"] = torch.stack([ds.cam_feats[t] for t in other])
            if ds.motion_feats is not None:
                batch["motion_feats"] = torch.stack([ds.motion_feats[t] for t in other])
        for key in zero:
            batch[key] = torch.zeros_like(batch[key])
        out = model.answer(batch["question"], batch["feats"], batch.get("cam_feats"), batch.get("motion_feats"))
        preds += out
        hits += [o == a.lower() for o, a in zip(out, batch["answer"])]
    per_type = collections.defaultdict(list)
    for q, h in zip(ds.questions, hits):
        per_type[q["template_type"]].append(h)
    return dict(acc=float(np.mean(hits)), per_type={k: float(np.mean(v)) for k, v in per_type.items()},
                n_per_type={k: len(v) for k, v in per_type.items()}, preds=preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--feat-tag", default="mini_mirror")
    ap.add_argument("--camera-tag", default="", help="e.g. mini_clip; empty = no camera tokens")
    ap.add_argument("--motion-tag", default="", help="e.g. mini_obj; empty = no explicit per-object motion tokens")
    ap.add_argument("--motion-qa", action="store_true", help="also train/eval on the auto-generated speed questions (make_motion_qa.py)")
    ap.add_argument("--motion-qa-tag", default="motion", help="matches make_motion_qa.py's --out-tag")
    ap.add_argument("--llm", default=str(ROOT / "weights" / "Qwen2.5-0.5B-Instruct"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr-proj", type=float, default=1e-3)
    ap.add_argument("--lr-lora", type=float, default=2e-4)
    ap.add_argument("--no-vision", action="store_true", help="disable radar tokens")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N training questions (quick tests)")
    ap.add_argument("--subsample", type=int, default=0, help="train on a random N of the training questions")
    ap.add_argument("--val-subsample", type=int, default=0, help="evaluate on a random N of the validation questions")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    feat_dir = ROOT / "cache" / "bev_features" / args.feat_tag
    cam_dir = ROOT / "cache" / "camera_features" / args.camera_tag if args.camera_tag else None
    motion_dir = ROOT / "cache" / "motion_tokens" / args.motion_tag if args.motion_tag else None
    extra = lambda split: [ROOT / "cache" / f"nuscenes_qa_{args.motion_qa_tag}_{split}.json"] if args.motion_qa else []
    train_ds = QADataset("train", feat_dir, cam_dir, extra("train"), motion_dir)
    val_ds = QADataset("val", feat_dir, cam_dir, extra("val"), motion_dir)
    if args.limit:
        train_ds.questions = train_ds.questions[:args.limit]
    sub_rng = np.random.default_rng(args.seed)
    if args.subsample and args.subsample < len(train_ds.questions):
        keep = sorted(sub_rng.choice(len(train_ds.questions), args.subsample, replace=False))
        train_ds.questions = [train_ds.questions[i] for i in keep]
    if args.val_subsample and args.val_subsample < len(val_ds.questions):
        keep = sorted(sub_rng.choice(len(val_ds.questions), args.val_subsample, replace=False))
        val_ds.questions = [val_ds.questions[i] for i in keep]
    use_camera, use_motion = bool(args.camera_tag), bool(args.motion_tag)
    print(f"{args.name}: {len(train_ds)} train / {len(val_ds)} val questions | radar: {not args.no_vision} | camera: {use_camera} | object tokens: {use_motion}", flush=True)
    print(f"majority-answer-per-question-type baseline on val: {majority_baseline(train_ds, val_ds):.3f}", flush=True)

    model = RadarLLM(args.llm, use_vision=not args.no_vision, use_camera=use_camera, use_motion=use_motion).to(device)
    lora = [p for n, p in model.llm.named_parameters() if p.requires_grad]
    proj = list(model.projector.parameters()) + [model.pos]
    if use_camera:
        proj += list(model.cam_projector.parameters()) + [model.cam_pos]
    if use_motion:
        proj += list(model.motion_projector.parameters()) + [model.motion_pos]
    use_prefix = not args.no_vision or use_camera or use_motion
    groups = [dict(params=lora, lr=args.lr_lora)] + ([dict(params=proj, lr=args.lr_proj)] if use_prefix else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train_ds) / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.1)
    print(f"trainable parameters: LoRA {sum(p.numel() for p in lora):,} + projector {sum(p.numel() for p in proj):,}", flush=True)

    history = []
    loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, collate_fn=collate_qa, drop_last=False)
    for epoch in range(args.epochs):
        model.train()
        t0, losses = time.time(), []
        for batch in loader:
            loss = model.loss(batch["question"], batch["answer"], batch["feats"], batch.get("cam_feats"), batch.get("motion_feats"))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        res = evaluate(model, val_ds)
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), val_acc=res["acc"]))
        print(f"epoch {epoch + 1}/{args.epochs} | train loss {np.mean(losses):.3f} | val accuracy {res['acc']:.3f} | {time.time() - t0:.0f}s", flush=True)
        # save after every epoch: a hang or crash mid-run then loses minutes, not hours
        keep = {n: p for n, p in model.named_parameters() if p.requires_grad}
        torch.save(keep, ROOT / "checkpoints" / f"qa_{args.name}.pt")
        (ROOT / "outputs").mkdir(exist_ok=True)
        json.dump(dict(name=args.name, vision=not args.no_vision, camera=use_camera, motion=use_motion,
                       motion_qa=args.motion_qa, history=history, in_progress=True),
                  open(ROOT / "outputs" / f"qa_{args.name}.json", "w"), indent=1)

    final = evaluate(model, val_ds)
    result = dict(name=args.name, vision=not args.no_vision, camera=use_camera, motion=use_motion, motion_qa=args.motion_qa,
                  history=history, val_acc=final["acc"],
                  per_type=final["per_type"], n_per_type=final["n_per_type"], majority_baseline=majority_baseline(train_ds, val_ds),
                  preds=final["preds"][:200], answers=[q["answer"] for q in val_ds.questions[:200]])
    if use_prefix:
        result["val_acc_shuffled_radar"] = evaluate(model, val_ds, shuffle_feats=True)["acc"]
        print(f"control: same model with sensor features of a RANDOM other keyframe: {result['val_acc_shuffled_radar']:.3f}", flush=True)
    (ROOT / "outputs").mkdir(exist_ok=True)
    json.dump(result, open(ROOT / "outputs" / f"qa_{args.name}.json", "w"), indent=1)
    keep = {n: p for n, p in model.named_parameters() if p.requires_grad}
    torch.save(keep, ROOT / "checkpoints" / f"qa_{args.name}.pt")
    print(f"final val accuracy {final['acc']:.3f} | per type: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(final["per_type"].items())), flush=True)


if __name__ == "__main__":
    main()
