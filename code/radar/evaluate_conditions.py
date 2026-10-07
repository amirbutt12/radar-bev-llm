"""Accuracy by scene condition (clear, night, rain) for a trained run, on the same validation subsample training used.

    python evaluate_conditions.py --name fusion_obj_trainval

Reads the run's config from outputs/qa_<name>.json, rebuilds the validation subsample with the same seed as train_qa.py,
and reports exact-match accuracy per condition, plus the wrong-frame control per condition when the model uses sensors.
"""
import argparse
import collections
import json

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes

from qa_data import ROOT, QADataset, load_mini_questions
from qa_model import RadarLLM
from train_qa import evaluate


def condition(desc):
    d = desc.lower()
    if "night" in d:
        return "night"
    if "rain" in d:
        return "rain"
    return "clear"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--subsample", type=int, default=60000)
    ap.add_argument("--val-subsample", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    cfg = json.load(open(ROOT / "outputs" / f"qa_{args.name}.json", encoding="utf-8"))
    feat_dir = ROOT / "cache" / "bev_features" / "trainval_bev"
    cam_dir = ROOT / "cache" / "camera_features" / "trainval_clip" if cfg["camera"] else None
    motion_dir = ROOT / "cache" / "motion_tokens" / "trainval_obj" if cfg["motion"] else None

    # consume the rng exactly as train_qa.py does, so the validation subsample is the same one training evaluated on
    rng = np.random.default_rng(args.seed)
    n_train = len(load_mini_questions("train", feat_dir))
    if args.subsample and args.subsample < n_train:
        rng.choice(n_train, args.subsample, replace=False)
    n_val = len(load_mini_questions("val", feat_dir))
    keep = sorted(rng.choice(n_val, args.val_subsample, replace=False)) if args.val_subsample < n_val else list(range(n_val))

    val_ds = QADataset("val", feat_dir, cam_dir, (), motion_dir)
    val_ds.questions = [val_ds.questions[i] for i in keep]

    nusc = NuScenes(args.version, str(ROOT / "data" / "nuscenes"), verbose=False)
    tok_cond = {}
    for s in nusc.scene:
        c, t = condition(s["description"]), s["first_sample_token"]
        while t:
            tok_cond[t] = c
            t = nusc.get("sample", t)["next"]

    device = torch.device(f"cuda:{args.gpu}")
    model = RadarLLM(str(ROOT / "weights" / "Qwen2.5-0.5B-Instruct"), use_vision=cfg["vision"], use_camera=cfg["camera"],
                     use_motion=cfg["motion"]).to(device)
    model.load_state_dict(torch.load(ROOT / "checkpoints" / f"qa_{args.name}.pt", map_location=device), strict=False)
    model.eval()

    answers = [q["answer"].lower() for q in val_ds.questions]
    conds = [tok_cond[q["sample_token"]] for q in val_ds.questions]
    sensors = cfg["vision"] or cfg["camera"] or cfg["motion"]

    res = evaluate(model, val_ds)
    hits = np.array([p == a for p, a in zip(res["preds"], answers)], dtype=float)
    out = dict(name=args.name, n_val=len(val_ds), by_condition={}, counts={})
    wf_hits = None
    if sensors:
        wf = evaluate(model, val_ds, shuffle_feats=True)
        wf_hits = np.array([p == a for p, a in zip(wf["preds"], answers)], dtype=float)

    for c in ("clear", "night", "rain"):
        m = np.array([x == c for x in conds])
        if m.sum() == 0:
            out["by_condition"][c] = None
            out["counts"][c] = 0
            continue
        row = dict(acc=float(hits[m].mean()))
        if wf_hits is not None:
            row["wrong_frame_acc"] = float(wf_hits[m].mean())
            row["gap"] = row["acc"] - row["wrong_frame_acc"]
        out["by_condition"][c] = row
        out["counts"][c] = int(m.sum())

    (ROOT / "outputs").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "outputs" / f"conditions_{args.name}.json", "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
