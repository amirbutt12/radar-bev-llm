"""Auto-generate motion/speed questions from ground-truth box velocities, to test whether the model can read
*how fast* things are moving -- something radar measures directly (Doppler) but camera/LiDAR must infer across frames.

Unlike nuScenes-QA's questions (existence, count, object, status, comparison), these are new template types:
    moving_count_car / moving_count_pedestrian   "How many cars/pedestrians are moving in the scene?"   -> a count
    top_speed_car                                "How fast is the fastest moving car: slow or fast?"    -> none / slow / fast

Written to cache/nuscenes_qa_motion_<split>.json, split the same way as the official nuScenes-QA cache (by looking up
which split each keyframe already belongs to there), so a keyframe's motion questions and its regular questions agree.

    python make_motion_qa.py
"""
import json

import numpy as np
from nuscenes.nuscenes import NuScenes

from bev_targets import get_velocities
from pillar_dataset import GRID, ROOT

MOVING = 0.5     # m/s: below this a GT box is "parked" (matches nuScenes-QA's own moving/parked convention)
FAST = 8.0       # m/s: above this a moving car counts as "fast" rather than "slow" (close to the median moving-car speed on mini, for a balanced split)


def main():
    nusc = NuScenes("v1.0-mini", str(ROOT / "data" / "nuscenes"), verbose=False)

    token_split = {}
    for split in ("train", "val"):
        for q in json.load(open(ROOT / "cache" / f"nuscenes_qa_mini_mirror_{split}.json", encoding="utf-8")):
            token_split[q["sample_token"]] = split

    out = {"train": [], "val": []}
    for tok, split in token_split.items():
        rows = get_velocities(nusc, tok, GRID)
        speed = np.hypot(rows[:, 1], rows[:, 2]) if len(rows) else np.zeros(0)
        cars = speed[rows[:, 0] == 0] if len(rows) else np.zeros(0)
        peds = speed[rows[:, 0] == 1] if len(rows) else np.zeros(0)

        moving_cars, moving_peds = int((cars > MOVING).sum()), int((peds > MOVING).sum())
        top = cars[cars > MOVING].max() if moving_cars else None
        top_bucket = "none" if top is None else ("fast" if top > FAST else "slow")

        base = dict(sample_token=tok, num_hop=0)
        out[split] += [
            dict(base, question="How many cars are moving in the scene?", answer=str(moving_cars), template_type="moving_count_car"),
            dict(base, question="How many pedestrians are moving in the scene?", answer=str(moving_peds), template_type="moving_count_pedestrian"),
            dict(base, question="How fast is the fastest moving car: slow or fast?", answer=top_bucket, template_type="top_speed_car"),
        ]

    for split, qs in out.items():
        path = ROOT / "cache" / f"nuscenes_qa_motion_{split}.json"
        json.dump(qs, open(path, "w", encoding="utf-8"))
        n_tok = len({q["sample_token"] for q in qs})
        print(f"{split}: {len(qs)} motion questions over {n_tok} keyframes -> {path}")


if __name__ == "__main__":
    main()
