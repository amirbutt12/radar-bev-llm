"""Fill the raw radar-point cache for every keyframe of the train and val scenes of a nuScenes version.

    python make_points_cache.py --version v1.0-trainval
"""
import argparse

from nuscenes.nuscenes import NuScenes

from pillar_dataset import ROOT, RadarPillarDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()

    nusc = NuScenes(args.version, str(ROOT / "data" / "nuscenes"), verbose=False)
    for split in args.splits:
        ds = RadarPillarDataset(nusc, split)
        secs = ds.prepare()
        print(f"{split}: {len(ds)} keyframes cached in {secs / 60:.1f} min -> {ds.cache_dir}", flush=True)


if __name__ == "__main__":
    main()
