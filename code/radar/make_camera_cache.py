"""Run a frozen CLIP vision encoder over every nuScenes-mini keyframe's front camera image and cache its patch tokens.

CLIP ViT-B/32 turns a 224x224 image into 7x7 = 49 patches, each a 768-number description of that patch. We save those 49
tokens (drop the whole-image CLS token) as float16, one .npy file per keyframe: cache/camera_features/<tag>/<sample_token>.npy

    python make_camera_cache.py --tag mini_clip
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel

from pillar_dataset import ROOT


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--clip-dir", default=str(ROOT / "weights" / "clip-vit-base-patch32"))
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    nusc = NuScenes(args.version, str(ROOT / "data" / "nuscenes"), verbose=False)
    proc = CLIPImageProcessor.from_pretrained(args.clip_dir)
    clip = CLIPVisionModel.from_pretrained(args.clip_dir).to(device).eval()

    out_dir = ROOT / "cache" / "camera_features" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = []
    for s in nusc.scene:
        t = s["first_sample_token"]
        while t:
            tokens.append(t)
            t = nusc.get("sample", t)["next"]

    n, shape = 0, None
    for i in range(0, len(tokens), 32):
        batch_tok = tokens[i:i + 32]
        imgs = []
        for tok in batch_tok:
            sd = nusc.get("sample_data", nusc.get("sample", tok)["data"]["CAM_FRONT"])
            imgs.append(Image.open(Path(nusc.dataroot) / sd["filename"]).convert("RGB"))
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        out = clip(pixel_values=px).last_hidden_state[:, 1:].half().cpu().numpy()      # drop CLS token -> (B, 49, 768)
        shape = out.shape[1:]
        for tok, f in zip(batch_tok, out):
            np.save(out_dir / f"{tok}.npy", f)
            n += 1
    print(f"cached {n} keyframes to {out_dir} | feature shape per keyframe {shape}")


if __name__ == "__main__":
    main()
