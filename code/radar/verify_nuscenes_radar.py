"""Load the 5 nuScenes radars for one keyframe, extract the fields the pillar encoder needs,
sanity-check them, and plot a BEV next to CAM_FRONT.

Per point (ego frame, x forward / y left): range, azimuth, compensated radial Doppler, RCS,
plus the compensated velocity vector (vx_comp, vy_comp).

    python verify_nuscenes_radar.py --scene-name scene-1077 --frame 10
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from PIL import Image
from pyquaternion import Quaternion

RADARS = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]
COLORS = dict(zip(RADARS, ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple"]))

# RadarPointCloud column order: x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid
# ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
IDX = dict(x=0, y=1, z=2, dyn_prop=3, rcs=5, vx=6, vy=7, vx_comp=8, vy_comp=9)


def load_radar(nusc, sample, sensor, invalid_states=None, dynprop_states=None, ambig_states=None):
    """Return a dict of arrays in the EGO frame for one radar of one keyframe."""
    sd = nusc.get("sample_data", sample["data"][sensor])
    pc = RadarPointCloud.from_file(
        str(Path(nusc.dataroot) / sd["filename"]),
        invalid_states=invalid_states, dynprop_states=dynprop_states, ambig_states=ambig_states,
    )
    p = pc.points  # (18, N), sensor frame
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    R = Quaternion(cs["rotation"]).rotation_matrix
    t = np.array(cs["translation"])

    xyz_sensor = p[[IDX["x"], IDX["y"], IDX["z"]]]
    xyz_ego = R @ xyz_sensor + t[:, None]
    v_comp_sensor = np.stack([p[IDX["vx_comp"]], p[IDX["vy_comp"]], np.zeros(p.shape[1])])
    v_comp_ego = R @ v_comp_sensor

    # Radial Doppler measured by the radar: project the compensated velocity on the line of sight.
    los = xyz_sensor[:2] / np.maximum(np.linalg.norm(xyz_sensor[:2], axis=0), 1e-6)
    v_radial = (p[[IDX["vx_comp"], IDX["vy_comp"]]] * los).sum(0)
    # For collinearity check: the residual perpendicular to the line of sight should be ~0.
    v_perp = p[IDX["vx_comp"]] * (-los[1]) + p[IDX["vy_comp"]] * los[0]

    return dict(
        sensor=sensor,
        x=xyz_ego[0], y=xyz_ego[1], z=xyz_ego[2],
        rng=np.linalg.norm(xyz_ego[:2], axis=0),
        azimuth=np.arctan2(xyz_ego[1], xyz_ego[0]),
        v_radial=v_radial, v_perp=v_perp,
        vx_comp=v_comp_ego[0], vy_comp=v_comp_ego[1],
        rcs=p[IDX["rcs"]], dyn_prop=p[IDX["dyn_prop"]],
    )


def main():
    plt.switch_backend("Agg")  # script mode: write PNGs; importing this module must not change the notebook backend
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default=r"D:\radar-bev-llm\data\nuscenes")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scene-name", default=None, help="e.g. scene-1077; default: first scene")
    ap.add_argument("--frame", type=int, default=10, help="keyframe index within the scene")
    ap.add_argument("--range", type=float, default=100.0, help="BEV half-extent in metres")
    ap.add_argument("--out", default=r"D:\radar-bev-llm\outputs")
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print("Scenes:")
    for s in nusc.scene:
        print(f"  {s['name']}  ({s['nbr_samples']} samples)  {s['description']}")

    scene = next((s for s in nusc.scene if s["name"] == args.scene_name), nusc.scene[0])
    token = scene["first_sample_token"]
    for _ in range(args.frame):
        token = nusc.get("sample", token)["next"]
    sample = nusc.get("sample", token)
    print(f"\nUsing {scene['name']}: {scene['description']!r}, keyframe {args.frame}")

    # Devkit defaults (drop invalid/ambiguous clusters) vs. everything the sensor reported.
    filt = {s: load_radar(nusc, sample, s) for s in RADARS}
    raw = {s: load_radar(nusc, sample, s, invalid_states=list(range(19)),
                         dynprop_states=list(range(8)), ambig_states=list(range(5))) for s in RADARS}

    print(f"\n{'sensor':18s}{'raw':>6s}{'kept':>6s}{'range m':>16s}{'v_radial m/s':>18s}{'rcs dBsm':>16s}{'|v_perp| max':>14s}")
    for s in RADARS:
        d, r = filt[s], raw[s]
        print(f"{s:18s}{len(r['x']):6d}{len(d['x']):6d}"
              f"{d['rng'].min():8.1f}..{d['rng'].max():5.1f}"
              f"{d['v_radial'].min():9.1f}..{d['v_radial'].max():5.1f}"
              f"{d['rcs'].min():8.1f}..{d['rcs'].max():4.1f}"
              f"{np.abs(d['v_perp']).max():14.4f}")
    total = sum(len(d["x"]) for d in filt.values())
    print(f"\nTotal points kept across 5 radars: {total}")

    # BEV plot
    fig, (ax_cam, ax) = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": [1.2, 1]})
    cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    ax_cam.imshow(Image.open(Path(args.dataroot) / cam["filename"]))
    ax_cam.set_title(f"CAM_FRONT | {scene['name']} | {scene['description']}", fontsize=9)
    ax_cam.axis("off")

    for s in RADARS:
        d = filt[s]
        sc = ax.scatter(d["y"], d["x"], s=14, color=COLORS[s], label=f"{s.replace('RADAR_', '')} ({len(d['x'])})", zorder=3)
        # short arrow = compensated velocity vector (scaled), i.e. how fast the point moves in the world
        ax.quiver(d["y"], d["x"], d["vy_comp"], d["vx_comp"], angles="xy", scale_units="xy", scale=1.0,
                  color=COLORS[s], width=0.002, alpha=0.7, zorder=2)
    ax.scatter([0], [0], marker="^", color="k", s=120, zorder=5)
    for r in (25, 50, 75, 100):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ls=":", color="grey", lw=0.6))
    ax.set_xlim(args.range, -args.range)   # ego-left on the left of the plot
    ax.set_ylim(-args.range * 0.6, args.range)
    ax.set_aspect("equal")
    ax.set_xlabel("y (m, left +)")
    ax.set_ylabel("x (m, forward +)")
    ax.set_title(f"Radar BEV, {total} pts. Arrows: ego-compensated velocity (m/s = m)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.2)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"radar_bev_{scene['name']}_f{args.frame}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
