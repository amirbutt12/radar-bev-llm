"""Qualitative dataset figures for the paper, built from a real trainval keyframe.
Fig A: camera + radar BEV (with GT boxes) -- what the pipeline's raw inputs actually look like.
Fig B: radar vs LiDAR point density in the same scene -- the visual case for the sensor-ceiling finding.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from nuscenes.nuscenes import NuScenes
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "radar"))
from bev_targets import get_boxes
from pillar_dataset import GRID, ROOT, pillarize

TOK = "0bcb08a96d264c8ca9f2119c6b0dbeb2"
OUT = Path(__file__).resolve().parent / "figures"

nusc = NuScenes("v1.0-trainval", str(ROOT / "data" / "nuscenes"), verbose=False)
sample = nusc.get("sample", TOK)

cam_sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
cam_img = Image.open(ROOT / "data" / "nuscenes" / cam_sd["filename"])
radar_pts = np.load(ROOT / "cache" / "radar_points" / "v1.0-trainval_sweeps5" / f"{TOK}.npy")
lidar_pts = np.load(ROOT / "cache" / "lidar_points" / "v1.0-trainval" / f"{TOK}.npy")
boxes = get_boxes(nusc, TOK, GRID)

# the specific sparse-radar pedestrian identified by find_example.py (the geometrically-nearest pedestrian
# overall is a different, less sparse one, so this is found by its known approximate position, not by distance)
ped_boxes = boxes[boxes[:, 0] == 1]
target = np.array([12.78, -9.56])
target_ped = ped_boxes[np.argmin(np.hypot(ped_boxes[:, 1] - target[0], ped_boxes[:, 2] - target[1]))]
_, tpx, tpy, tpw, tpl, _ = target_ped
tr = max(tpw, tpl) / 2 + 1.0
radar_near = int((np.hypot(radar_pts[:, 0] - tpx, radar_pts[:, 1] - tpy) < tr).sum())
lidar_near = int((np.hypot(lidar_pts[:, 0] - tpx, lidar_pts[:, 1] - tpy) < tr).sum())
print(f"target pedestrian at ({tpx:.1f}, {tpy:.1f}): radar points within {tr:.1f}m = {radar_near}, lidar points = {lidar_near}")


def draw_boxes(ax, boxes):
    for cls, x, y, w, l, yaw in boxes:
        c, s = np.cos(yaw), np.sin(yaw)
        xy = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]]) @ np.array([[c, s], [-s, c]]) + [x, y]
        ec = "#228833" if int(cls) == 0 else "#CC3377"
        ax.add_patch(Polygon(np.stack([xy[:, 1], xy[:, 0]], 1), closed=True, fill=False, ec=ec, lw=1.4, zorder=5))


def callout(ax, fontsize=7.5):
    ax.annotate(f"{radar_near if ax.get_gid() == 'radar' else lidar_near} pts here", xy=(tpy, tpx),
                xytext=(tpy - 9, tpx + 9), fontsize=fontsize, color="#CC3377", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#CC3377", lw=1.2))


# ---------------------------------------------------------------- Fig A: camera + radar BEV with GT boxes
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

axes[0].imshow(cam_img)
axes[0].axis("off")
axes[0].set_title("Front camera", fontsize=10)

ax = axes[1]
ax.set_gid("radar")
ext = [GRID["y0"] + GRID["ny"] * GRID["cell"], GRID["y0"], GRID["x0"], GRID["x0"] + GRID["nx"] * GRID["cell"]]
ax.imshow(np.minimum(pillarize(radar_pts)[0], 3)[::-1, ::-1], extent=ext, cmap="Greys", vmin=0, vmax=3)
draw_boxes(ax, boxes)
ax.plot(0, 0, "r^", ms=9)
callout(ax)
ax.set_xlim(25, -25)
ax.set_ylim(-5, 45)
ax.set_xlabel("y (m), left of ego")
ax.set_ylabel("x (m), forward")
ax.set_title(f"Radar points (ego frame) + ground-truth boxes\n{len(radar_pts)} points, green=car magenta=pedestrian", fontsize=9.5)
plt.tight_layout()
plt.savefig(OUT / "fig_data_example.pdf")
plt.close()
print("wrote fig_data_example.pdf")

# ---------------------------------------------------------------- Fig B: radar vs lidar density, same scene
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))
for gid, ax, pts, title, color in zip(
    ["radar", "lidar"], axes, [radar_pts, lidar_pts],
    [f"Radar\n{len(radar_pts)} points (5-sweep accumulated)", f"LiDAR\n{len(lidar_pts)} points (single sweep)"],
    ["#4477AA", "#999999"]):
    ax.set_gid(gid)
    ax.scatter(pts[:, 1], pts[:, 0], s=3, c=color, alpha=0.6, linewidths=0)
    draw_boxes(ax, boxes)
    ax.plot(0, 0, "r^", ms=8, zorder=6)
    callout(ax)
    ax.set_xlim(20, -20)
    ax.set_ylim(-5, 35)
    ax.set_xlabel("y (m)")
    ax.set_title(title, fontsize=9.5)
axes[0].set_ylabel("x (m), forward")
plt.tight_layout()
plt.savefig(OUT / "fig_radar_vs_lidar.pdf")
plt.close()
print("wrote fig_radar_vs_lidar.pdf")
