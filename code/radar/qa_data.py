"""nuScenes-QA questions for the keyframes we have cached radar features for (nuScenes mini)."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(r"D:\radar-bev-llm")


def load_mini_questions(split, feat_dir):
    """Questions of the official nuScenes-QA split ('train' / 'val') whose keyframe has cached features.
    The filtered list is cached on disk because the full question files are large (100+ MB)."""
    feat_dir = Path(feat_dir)
    cache = ROOT / "cache" / f"nuscenes_qa_{feat_dir.name}_{split}.json"
    if cache.exists():
        return json.load(open(cache, encoding="utf-8"))
    tokens = {p.stem for p in feat_dir.glob("*.npy")}
    questions = json.load(open(ROOT / "data" / "nuscenes_qa" / f"NuScenes_{split}_questions.json", encoding="utf-8"))["questions"]
    kept = [q for q in questions if q["sample_token"] in tokens]
    json.dump(kept, open(cache, "w", encoding="utf-8"))
    return kept


class LazyFeats:
    """Per-keyframe feature files read from disk on access, so the whole cache never sits in RAM."""

    def __init__(self, folder, tokens):
        self.folder, self.tokens = Path(folder), sorted(tokens)

    def __getitem__(self, tok):
        return torch.from_numpy(np.load(self.folder / f"{tok}.npy"))

    def __iter__(self):
        return iter(self.tokens)


class QADataset(Dataset):
    """Item: dict(question, answer, template_type, num_hop, token, feats (64, 32, 32) float16[, cam_feats (49, 768) float16])."""

    def __init__(self, split, feat_dir, cam_feat_dir=None, extra_files=(), motion_feat_dir=None):
        self.questions = load_mini_questions(split, feat_dir)
        for path in extra_files:
            self.questions += json.load(open(path, encoding="utf-8"))
        toks = {q["sample_token"] for q in self.questions}
        self.feats = LazyFeats(feat_dir, toks)
        self.cam_feats = LazyFeats(cam_feat_dir, toks) if cam_feat_dir else None
        self.motion_feats = {t: torch.from_numpy(np.load(Path(motion_feat_dir) / f"{t}.npy")) for t in toks} if motion_feat_dir else None

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, i):
        q = self.questions[i]
        tok = q["sample_token"]
        item = dict(question=q["question"], answer=q["answer"], template_type=q["template_type"], num_hop=q["num_hop"],
                    token=tok, feats=self.feats[tok])
        if self.cam_feats is not None:
            item["cam_feats"] = self.cam_feats[tok]
        if self.motion_feats is not None:
            item["motion_feats"] = self.motion_feats[tok]
        return item


def collate_qa(batch):
    out = dict(question=[b["question"] for b in batch], answer=[b["answer"] for b in batch],
               template_type=[b["template_type"] for b in batch], token=[b["token"] for b in batch],
               feats=torch.stack([b["feats"] for b in batch]))
    if "cam_feats" in batch[0]:
        out["cam_feats"] = torch.stack([b["cam_feats"] for b in batch])
    if "motion_feats" in batch[0]:
        out["motion_feats"] = torch.stack([b["motion_feats"] for b in batch])
    return out
