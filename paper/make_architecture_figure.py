"""Pipeline architecture overview, left-to-right: radar + camera streams -> projectors -> LLM prefix -> answer.
Two radar sub-lanes (detection/object-token path and dense-token path) share one backbone box, plus a camera lane."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size": 7.6})

COL_RADAR = "#4477AA"
COL_CAM = "#DDCC77"
COL_OBJ = "#AA3377"
COL_LLM = "#228833"
COL_NEUTRAL = "#BBBBBB"
EDGE = "#333333"


def box(ax, cx, cy, w, h, text, color, fontsize=7.2, textcolor="black", fontweight="normal"):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.035",
                        linewidth=1.0, edgecolor=EDGE, facecolor=color, alpha=0.9, zorder=2)
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, color=textcolor, zorder=3,
            fontweight=fontweight, linespacing=1.25)
    return (cx, cy, w, h)


def edge_pt(b, side):
    cx, cy, w, h = b
    return {"r": (cx + w / 2, cy), "l": (cx - w / 2, cy), "t": (cx, cy + h / 2), "b": (cx, cy - h / 2)}[side]


def arrow(ax, p1, p2, color=EDGE, lw=1.0):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=8, linewidth=lw, color=color, zorder=1,
                         shrinkA=1, shrinkB=1, connectionstyle="arc3,rad=0.0")
    ax.add_patch(a)


fig, ax = plt.subplots(figsize=(14.3, 7.0))
ax.set_xlim(0, 188)
ax.set_ylim(-2, 62)
ax.axis("off")

Y_TOP, Y_MID, Y_BOT = 50, 32, 12

# Stage 1: raw inputs
b_radar_in = box(ax, 12, Y_MID + 9, 20, 7.5, "Radar: 5 sensors\nx 5 sweeps", COL_RADAR, textcolor="white")
b_cam_in = box(ax, 12, Y_BOT, 18, 6.5, "Camera image", COL_CAM)

# Stage 2: radar backbone (shared), camera encoder
b_backbone = box(ax, 40, Y_MID + 9, 24, 10,
                  "Radar backbone:\nego-frame + pillarize\n(128x128) + U-Net\n(~1.85M params)", COL_RADAR,
                  fontsize=6.6, textcolor="white")
b_clip = box(ax, 40, Y_BOT, 20, 6.5, "Frozen CLIP\nViT-B/32", COL_CAM, fontsize=6.9)

arrow(ax, edge_pt(b_radar_in, "r"), edge_pt(b_backbone, "l"))
arrow(ax, edge_pt(b_cam_in, "r"), edge_pt(b_clip, "l"))

# Stage 3: branch -- top: detection heads: bottom: pooled tokens; camera: patch tokens
b_heads = box(ax, 65, Y_TOP, 20, 7.5, "CenterNet heads:\nboxes (oracle or\ndetected)", COL_RADAR, fontsize=6.6,
              textcolor="white")
b_pool = box(ax, 65, Y_MID, 20, 6.5, "Adaptive pool\n-> 256 tokens", COL_RADAR, fontsize=6.9, textcolor="white")
b_camtok = box(ax, 65, Y_BOT, 20, 6.5, "49 camera\npatch tokens", COL_CAM, fontsize=6.9)

arrow(ax, (52, Y_MID + 9 + 2), edge_pt(b_heads, "l"))
arrow(ax, (52, Y_MID + 9 - 2), edge_pt(b_pool, "l"))
arrow(ax, edge_pt(b_clip, "r"), edge_pt(b_camtok, "l"))

# Stage 4: motion token extraction (top only)
b_motion = box(ax, 92, Y_TOP, 24, 8.5, "Motion token\nextraction: 8 slots\nx 9-dim (radar vel.\n+ confidence)", COL_OBJ,
               fontsize=5.9, textcolor="white")
arrow(ax, edge_pt(b_heads, "r"), edge_pt(b_motion, "l"))

# Stage 5: projectors
b_proj_obj = box(ax, 116, Y_TOP, 16, 6.5, "Motion\nprojector", COL_OBJ, fontsize=6.9, textcolor="white")
b_proj_radar = box(ax, 112, Y_MID, 16, 6.5, "Radar\nprojector", COL_RADAR, fontsize=6.9, textcolor="white")
b_proj_cam = box(ax, 112, Y_BOT, 16, 6.5, "Camera\nprojector", COL_CAM, fontsize=6.9)

arrow(ax, edge_pt(b_motion, "r"), edge_pt(b_proj_obj, "l"))
arrow(ax, edge_pt(b_pool, "r"), edge_pt(b_proj_radar, "l"))
arrow(ax, edge_pt(b_camtok, "r"), edge_pt(b_proj_cam, "l"))

# Stage 6: concatenated prefix
b_prefix = box(ax, 133, Y_MID, 16, 26, "Concatenated\nprefix tokens\n(+ positional\nembeddings)", COL_NEUTRAL,
               fontsize=6.3)
arrow(ax, edge_pt(b_proj_obj, "r"), (125, 45))
arrow(ax, edge_pt(b_proj_radar, "r"), (125, Y_MID))
arrow(ax, edge_pt(b_proj_cam, "r"), (125, 19))

# Stage 7: LLM + answer
b_llm = box(ax, 156, Y_MID, 18, 14, "Qwen2.5-0.5B-\nInstruct (frozen)\n+ LoRA (r=16)\n+ question text", COL_LLM,
            fontsize=6.3, textcolor="white")
arrow(ax, edge_pt(b_prefix, "r"), edge_pt(b_llm, "l"))
ax.annotate("Answer", xy=(182, Y_MID), xytext=(167, Y_MID), fontsize=7.4, fontweight="bold", va="center",
            arrowprops=dict(arrowstyle="-|>", mutation_scale=8, linewidth=1.0, color=EDGE))

plt.tight_layout()
plt.savefig("figures/fig1_architecture.pdf", bbox_inches="tight")
plt.close()
print("wrote figures/fig1_architecture.pdf")
