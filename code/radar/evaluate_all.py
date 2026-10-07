"""Reload every trained radar-language checkpoint and evaluate them all on the SAME question sets, with the same set
of counterfactual controls, so the numbers are directly comparable. Writes outputs/qa_all_results.json for the notebook.

    python evaluate_all.py
"""
import json

import torch

from qa_data import ROOT, QADataset
from qa_model import RadarLLM
from train_qa import evaluate, majority_baseline, majority_per_type

LLM_DIR = str(ROOT / "weights" / "Qwen2.5-0.5B-Instruct")
FEAT_DIR = ROOT / "cache" / "bev_features" / "mini_mirror"
CAM_DIR = ROOT / "cache" / "camera_features" / "mini_clip"
MOTION_DIR = ROOT / "cache" / "motion_tokens" / "mini_obj"

RUNS = ["blind", "radar", "camera", "fusion", "blind_motion", "radar_motion", "camera_motion", "fusion_motion",
        "radar_obj", "motion_only", "fusion_obj"]


def load_model(cfg, device):
    model = RadarLLM(LLM_DIR, use_vision=cfg["vision"], use_camera=cfg.get("camera", False), use_motion=cfg.get("motion", False)).to(device)
    state = torch.load(ROOT / "checkpoints" / f"qa_{cfg['name']}.pt", map_location=device)
    model.load_state_dict(state, strict=False)
    return model.eval()


def main():
    device = torch.device("cuda:0")
    results = {}
    ds_cache = {}   # (motion_qa,) -> (train_ds, val_ds)  so we only build each question set once

    for name in RUNS:
        cfg_path = ROOT / "outputs" / f"qa_{name}.json"
        if not cfg_path.exists():
            print(f"skip {name}: no outputs/qa_{name}.json"); continue
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        cfg["name"] = name
        motion_qa = cfg.get("motion_qa", False)

        if motion_qa not in ds_cache:
            extra = lambda split: [ROOT / "cache" / f"nuscenes_qa_motion_{split}.json"] if motion_qa else []
            ds_cache[motion_qa] = (QADataset("train", FEAT_DIR, CAM_DIR, extra("train"), MOTION_DIR),
                                    QADataset("val", FEAT_DIR, CAM_DIR, extra("val"), MOTION_DIR))
        train_ds, val_ds = ds_cache[motion_qa]

        model = load_model(cfg, device)
        row = dict(vision=cfg["vision"], camera=cfg.get("camera", False), motion=cfg.get("motion", False), motion_qa=motion_qa,
                   majority_baseline=majority_baseline(train_ds, val_ds), majority_per_type=majority_per_type(train_ds, val_ds))
        row["correct"] = evaluate(model, val_ds)
        modalities = [k for k, on in [("feats", cfg["vision"]), ("cam_feats", cfg.get("camera", False)), ("motion_feats", cfg.get("motion", False))] if on]
        if len(modalities) >= 1:
            row["wrong_frame"] = evaluate(model, val_ds, shuffle_feats=True)
        if len(modalities) >= 2:                                        # per-modality zeroing only tells us something new with 2+ modalities
            row["zeroed"] = {m: evaluate(model, val_ds, zero=(m,)) for m in modalities}
        results[name] = row
        print(f"{name}: correct {row['correct']['acc']:.3f}" +
              (f" | wrong-frame {row['wrong_frame']['acc']:.3f}" if "wrong_frame" in row else "") +
              (f" | zeroed: " + ", ".join(f"{m.replace('_feats','')} {v['acc']:.3f}" for m, v in row["zeroed"].items()) if "zeroed" in row else ""),
              flush=True)
        del model
        torch.cuda.empty_cache()

    json.dump(results, open(ROOT / "outputs" / "qa_all_results.json", "w"), indent=1)
    print("wrote", ROOT / "outputs" / "qa_all_results.json")


if __name__ == "__main__":
    main()
