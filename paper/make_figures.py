"""Generate all result figures for the paper, straight from the numbers verified in PROJECT_NOTES.md Part V/VII.
No numbers here are invented; every array below matches a table already checked against training logs."""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})
OUT = "figures"

# ---------------------------------------------------------------- Fig 1: modality ladder, acc + wrong-frame gap
models = ["Blind", "Camera", "Radar\n(dense)", "Object\ntokens", "Radar +\nobject", "Full\nfusion"]
acc = [0.513, 0.544, 0.545, 0.520, 0.558, 0.552]
gap = [None, 0.053, 0.041, 0.055, 0.075, 0.070]

fig, ax1 = plt.subplots(figsize=(6.5, 3.0))
x = np.arange(len(models))
w = 0.35
ax1.bar(x - w / 2, acc, w, label="Overall accuracy", color="#4477AA")
ax1.axhline(0.0, color="k", lw=0.8)
gap_vals = [g if g is not None else 0 for g in gap]
ax2 = ax1.twinx()
ax2.bar(x + w / 2, gap_vals, w, label="Wrong-frame gap", color="#CC6677")
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=8)
ax1.set_ylabel("Overall accuracy")
ax2.set_ylabel("Wrong-frame gap")
ax1.set_ylim(0, 0.62)
ax2.set_ylim(0, 0.085)
for xi, v in zip(x, acc):
    ax1.text(xi - w / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=6.5)
for xi, v in zip(x, gap):
    if v is not None:
        ax2.text(xi + w / 2, v + 0.002, f"+{v:.3f}", ha="center", fontsize=6.5)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7.5, ncol=2)
ax1.set_title("Modality ablation on motion/speed questions (oracle boxes)", fontsize=9.5)
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_modality_ladder.pdf")
plt.close()

# ---------------------------------------------------------------- Fig 2: per-question-type breakdown
qtypes = ["Cars moving", "Pedestrians moving", "Speed judgment"]
baseline = [0.195, 0.340, 0.345]
model_names = ["Blind", "Camera", "Radar (dense)", "Object tokens", "Radar+object", "Full fusion"]
vals = {
    "Blind": [0.19, 0.34, 0.35],
    "Camera": [0.39, 0.33, 0.68],
    "Radar (dense)": [0.35, 0.36, 0.66],
    "Object tokens": [0.29, 0.43, 0.60],
    "Radar+object": [0.37, 0.45, 0.73],
    "Full fusion": [0.39, 0.47, 0.68],
}
colors = ["#888888", "#DDCC77", "#4477AA", "#AA3377", "#228833", "#CC6677"]

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
for i, (ax, qt) in enumerate(zip(axes, qtypes)):
    heights = [vals[m][i] for m in model_names]
    bars = ax.bar(np.arange(len(model_names)), heights, color=colors)
    ax.axhline(baseline[i], color="k", ls="--", lw=1.1, label="Majority baseline" if i == 0 else None)
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, rotation=60, ha="right", fontsize=6.5)
    ax.set_title(qt, fontsize=8.5)
    for xi, v in enumerate(heights):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=6)
    if i == 0:
        ax.set_ylabel("Accuracy")
axes[1].text(0.95, 0.03, "Camera-only is the only\nmodel below baseline here", transform=axes[1].transAxes,
             ha="right", va="bottom", fontsize=6, style="italic", color="#333333")
axes[0].set_ylim(0, 0.82)
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=1, fontsize=7)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_per_question_type.pdf")
plt.close()

# ---------------------------------------------------------------- Fig 3: day/night/rain wrong-frame gap
models3 = ["Camera", "Radar\n(dense)", "Object\ntokens", "Radar+\nobject", "Full\nfusion"]
clear = [0.045, 0.036, 0.050, 0.069, 0.064]
night = [0.091, 0.079, 0.110, 0.139, 0.144]
rain = [0.079, 0.054, 0.053, 0.078, 0.068]

fig, ax = plt.subplots(figsize=(6.5, 2.9))
x = np.arange(len(models3))
w = 0.26
ax.bar(x - w, clear, w, label="Clear", color="#88CCEE")
ax.bar(x, night, w, label="Night", color="#332288")
ax.bar(x + w, rain, w, label="Rain", color="#44AA99")
ax.set_xticks(x)
ax.set_xticklabels(models3, fontsize=8)
ax.set_ylabel("Wrong-frame gap")
ax.set_title("Wrong-frame gap by scene condition (night n=353, rain n=987, clear n=4660)", fontsize=9)
ax.legend(fontsize=8)
for xi, v in zip(x, night):
    ax.text(xi, v + 0.003, f"{v:.3f}", ha="center", fontsize=6.5, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_condition_split.pdf")
plt.close()

# ---------------------------------------------------------------- Fig 4: pedestrian-counting stress test
configs = ["Oracle\nboxes", "Detected\n0.3 thr,\nbalanced", "Detected\n0.15 thr,\nbalanced",
           "Detected\n0.3 thr,\n+confidence", "Detected\n0.15 thr,\n+confidence"]
ped_vals = [0.45, 0.37, 0.36, 0.36, 0.35]
ccolors = ["#228833", "#CC6677", "#CC6677", "#4477AA", "#4477AA"]

fig, ax = plt.subplots(figsize=(6.5, 3.0))
bars = ax.bar(np.arange(len(configs)), ped_vals, color=ccolors)
ax.axhline(0.340, color="k", ls="--", lw=1.2, label="Majority baseline (0.340)")
ax.set_xticks(np.arange(len(configs)))
ax.set_xticklabels(configs, fontsize=7.5)
ax.set_ylabel("Pedestrian-moving-count accuracy")
ax.set_ylim(0, 0.52)
for xi, v in enumerate(ped_vals):
    ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
ax.annotate("Real, clear margin\nunder oracle boxes", xy=(0, 0.45), xytext=(0.3, 0.485),
            fontsize=7, arrowprops=dict(arrowstyle="->", lw=1))
ax.annotate("Four independent fixes tried;\nall land in the same narrow band", xy=(2.5, 0.36), xytext=(2.5, 0.42),
            fontsize=7, ha="center", arrowprops=dict(arrowstyle="->", lw=1))
ax.legend(fontsize=7.5, loc="upper right")
ax.set_title("Pedestrian-counting collapses under real detections, resists four independent fixes", fontsize=8.8)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_stress_test.pdf")
plt.close()

# ---------------------------------------------------------------- Fig 5: PR sweep, car vs pedestrian (retrained detector)
thr = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
car_prec = [0.094, 0.125, 0.219, 0.325, 0.436, 0.537, 0.708]
car_rec = [0.605, 0.580, 0.516, 0.466, 0.429, 0.395, 0.329]
ped_prec = [0.041, 0.055, 0.120, 0.220, 0.348, 0.471, 0.679]
ped_rec = [0.267, 0.240, 0.177, 0.146, 0.123, 0.104, 0.081]

fig, ax = plt.subplots(figsize=(4.4, 3.4))
ax.plot(car_rec, car_prec, "o-", color="#4477AA", label="Car", ms=4)
ax.plot(ped_rec, ped_prec, "s-", color="#CC6677", label="Pedestrian", ms=4)
for r, p, t in zip(car_rec, car_prec, thr):
    if t in (0.15, 0.30):
        ax.annotate(f"{t}", (r, p), textcoords="offset points", xytext=(4, 4), fontsize=6, color="#4477AA")
for r, p, t in zip(ped_rec, ped_prec, thr):
    if t in (0.15, 0.30):
        ax.annotate(f"{t}", (r, p), textcoords="offset points", xytext=(4, -8), fontsize=6, color="#CC6677")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision/recall sweep, retrained\ndetector (20 epochs)", fontsize=9)
ax.legend(fontsize=8)
ax.set_xlim(0, 0.65)
ax.set_ylim(0, 0.75)
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_pr_sweep.pdf")
plt.close()

print("wrote 5 figures to", OUT)
