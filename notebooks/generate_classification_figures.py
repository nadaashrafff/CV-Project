"""
Classification System Comparison Figures
Generates three figures in figures/final/:
  1. classification_per_class_f1.png  — grouped bar per class
  2. classification_radar.png         — radar / spider chart
  3. classification_dim_tradeoff.png  — feature-dim vs Macro-F1 scatter
Run: python3 notebooks/generate_classification_figures.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

os.makedirs("figures/final", exist_ok=True)

CLASSES   = ["BIODEG.", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
CLASSES_F = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]

# ── Per-class F1 data (from classification_results.csv & per-class CSVs) ────
PER_CLASS = {
    "E2\nClassical RF":      [0.8857, 0.7442, 0.5678, 0.5666, 0.5335, 0.5880],
    "E3\nDeep SVM":          [0.9249, 0.8706, 0.7294, 0.7654, 0.7419, 0.6767],
    "E5\nPCA Fusion":        [0.9340, 0.8934, 0.7681, 0.7848, 0.7373, 0.7253],
    "E11\nYOLO+Classical":   [0.9181, 0.8455, 0.6677, 0.6842, 0.6523, 0.6393],
    "E12\nThree-Way":        [0.9352, 0.9016, 0.7692, 0.7821, 0.7485, 0.7416],
}

EXP_COLORS = {
    "E2\nClassical RF":    "#4C72B0",
    "E3\nDeep SVM":        "#4C72B0",
    "E5\nPCA Fusion":      "#55A868",
    "E11\nYOLO+Classical": "#DD8452",
    "E12\nThree-Way":      "#DD8452",
}
# darken E12 vs E11, E3 vs E2
EXP_COLORS["E2\nClassical RF"]    = "#9DB8D2"
EXP_COLORS["E3\nDeep SVM"]        = "#2A5A9C"
EXP_COLORS["E5\nPCA Fusion"]      = "#2E7D50"
EXP_COLORS["E11\nYOLO+Classical"] = "#E8A87C"
EXP_COLORS["E12\nThree-Way"]      = "#C0550A"

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Per-class F1 grouped bar chart
# ══════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(13, 5.5))
fig1.patch.set_facecolor("#FAFAFA")
ax1.set_facecolor("#FAFAFA")

n_exp    = len(PER_CLASS)
n_cls    = len(CLASSES)
bw       = 0.14
offsets  = np.linspace(-(n_exp - 1) * bw / 2, (n_exp - 1) * bw / 2, n_exp)
x        = np.arange(n_cls)

for i, (exp, vals) in enumerate(PER_CLASS.items()):
    bars = ax1.bar(x + offsets[i], vals, width=bw,
                   color=EXP_COLORS[exp], label=exp.replace("\n", " "),
                   zorder=3, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        if v < 0.68:
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                     f"{v:.2f}", ha="center", va="bottom",
                     fontsize=5.5, color="#333333", rotation=90)

ax1.set_xticks(x)
ax1.set_xticklabels(CLASSES, fontsize=11)
ax1.set_ylabel("F1 Score", fontsize=11)
ax1.set_ylim(0.45, 1.00)
ax1.set_title("Per-Class F1: Key Experiments Comparison", fontsize=13, fontweight="bold", pad=12)
ax1.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax1.set_axisbelow(True)
for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)

# annotate hardest class
ax1.annotate("Hardest class\nacross all methods",
             xy=(2, 0.535), xytext=(2.3, 0.535),
             fontsize=8, color="#555",
             arrowprops=dict(arrowstyle="->", color="#888", lw=0.8))

ax1.legend(loc="upper right", fontsize=8.5, ncol=5,
           framealpha=0.9, edgecolor="#ccc",
           columnspacing=0.8, handlelength=1.2)

plt.tight_layout()
out1 = "figures/final/classification_per_class_f1.png"
plt.savefig(out1, dpi=180, bbox_inches="tight")
print(f"Saved → {out1}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Radar / Spider chart (top 4 methods)
# ══════════════════════════════════════════════════════════════════════════════
RADAR_DATA = {
    "E3: Deep SVM":       [0.9249, 0.8706, 0.7294, 0.7654, 0.7419, 0.6767],
    "E5: PCA Fusion":     [0.9340, 0.8934, 0.7681, 0.7848, 0.7373, 0.7253],
    "E11: YOLO+Classical":[0.9181, 0.8455, 0.6677, 0.6842, 0.6523, 0.6393],
    "E12: Three-Way":     [0.9352, 0.9016, 0.7692, 0.7821, 0.7485, 0.7416],
}
RADAR_COLORS = ["#2A5A9C", "#2E7D50", "#E8A87C", "#C0550A"]

N = len(CLASSES)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig2, ax2 = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
fig2.patch.set_facecolor("#FAFAFA")
ax2.set_facecolor("#F5F5F5")

for (name, vals), color in zip(RADAR_DATA.items(), RADAR_COLORS):
    vals_closed = vals + vals[:1]
    ax2.plot(angles, vals_closed, "o-", linewidth=2, color=color,
             label=name, markersize=4, zorder=3)
    ax2.fill(angles, vals_closed, alpha=0.08, color=color)

ax2.set_thetagrids(np.degrees(angles[:-1]), CLASSES_F, fontsize=10)
ax2.set_ylim(0.5, 1.0)
yticks = [0.55, 0.65, 0.75, 0.85, 0.95]
ax2.set_yticks(yticks)
ax2.set_yticklabels([f"{v:.2f}" for v in yticks], fontsize=7, color="#666")
ax2.set_rlabel_position(30)
ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
ax2.xaxis.grid(True, linestyle="-", alpha=0.25)
ax2.spines["polar"].set_visible(False)

ax2.set_title("Per-Class F1 Profile — Top Experiments",
              fontsize=13, fontweight="bold", pad=20)
ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.13),
           ncol=2, fontsize=9, framealpha=0.9, edgecolor="#ccc")

plt.tight_layout()
out2 = "figures/final/classification_radar.png"
plt.savefig(out2, dpi=180, bbox_inches="tight")
print(f"Saved → {out2}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Feature-dim vs Macro-F1 scatter (compression tradeoff)
# ══════════════════════════════════════════════════════════════════════════════
# (dim, macro_f1, inference_ms, category, label)
SCATTER = [
    (252,  0.6476,  0.01,  "single",  "E2\nClassical RF"),
    (1280, 0.7848, 21.27,  "single",  "E3\nDeep SVM"),
    (1532, 0.6960,  0.01,  "early",   "E4\nRaw Concat"),
    (939,  0.8071, 10.00,  "early",   "E5\nPCA Fusion"),
    (256,  0.7705,  3.17,  "early",   "E6\nAutoenc."),
    (200,  0.7081,  0.01,  "early",   "E7\nFeat. Sel."),
    (1532, 0.7820,  0.01,  "late",    "E8\nAvg Vote"),
    (1532, 0.7832,  0.01,  "late",    "E9\nWt. Vote"),
    (128,  0.7752,  3.17,  "late",    "E10\nAttention"),
    (508,  0.7345,  5.62,  "hybrid",  "E11\nYOLO+Cls"),
    (1088, 0.8130, 23.71,  "hybrid",  "E12\nThree-Way"),
]

CAT_COLOR = {"single": "#2A5A9C", "early": "#2E7D50",
             "late": "#C44E52",   "hybrid": "#C0550A"}

fig3, ax3 = plt.subplots(figsize=(10, 6))
fig3.patch.set_facecolor("#FAFAFA")
ax3.set_facecolor("#FAFAFA")

# bubble size proportional to inference time (floored at 60 for visibility)
for (dim, f1, ms, cat, lbl) in SCATTER:
    size = max(ms * 14, 60)
    ax3.scatter(dim, f1, s=size, color=CAT_COLOR[cat],
                alpha=0.82, edgecolors="white", linewidths=1.2, zorder=3)
    # label placement offset by category
    dx = {"single": 18, "early": -18, "late": 20, "hybrid": 18}[cat]
    dy = {"single": 0.003, "early": -0.010, "late": 0.003, "hybrid": -0.010}[cat]
    ax3.text(dim + dx, f1 + dy, lbl.replace("\n", " "),
             fontsize=7.5, va="center", ha="left", color="#222")

# pareto frontier line (manually identified: E2→E7→E6→E11→E5→E12 on dim-axis,
#                        but on Pareto: E2, E6, E5, E12 — smallest dim for each F1 level)
pareto = sorted([(252, 0.6476), (256, 0.7705), (939, 0.8071), (1088, 0.8130)],
                key=lambda x: x[0])
px, py = zip(*pareto)
ax3.plot(px, py, "k--", linewidth=1.0, alpha=0.45, zorder=2, label="Pareto frontier")

# reference bands
ax3.axhline(0.8071, color="#2E7D50", linewidth=0.8, linestyle=":", alpha=0.6)
ax3.axhline(0.7848, color="#2A5A9C", linewidth=0.8, linestyle=":", alpha=0.6)
ax3.text(1400, 0.8075, "E5 (two-way best)", fontsize=7.5, color="#2E7D50", style="italic")
ax3.text(1400, 0.7852, "E3 (single-modal best)", fontsize=7.5, color="#2A5A9C", style="italic")

ax3.set_xlabel("Post-compression Feature Dimension", fontsize=11)
ax3.set_ylabel("Macro-F1 (test set)", fontsize=11)
ax3.set_title("Compression–Accuracy Tradeoff\nBubble size ∝ inference time (ms/crop)",
              fontsize=13, fontweight="bold", pad=12)
ax3.set_xlim(60, 1750)
ax3.set_ylim(0.62, 0.835)
ax3.xaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
ax3.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
ax3.set_axisbelow(True)
for spine in ["top", "right"]:
    ax3.spines[spine].set_visible(False)

legend_handles = [
    mpatches.Patch(color=CAT_COLOR["single"], label="Single modality"),
    mpatches.Patch(color=CAT_COLOR["early"],  label="Early fusion"),
    mpatches.Patch(color=CAT_COLOR["late"],   label="Late fusion"),
    mpatches.Patch(color=CAT_COLOR["hybrid"], label="Hybrid (YOLO)"),
    plt.Line2D([0], [0], color="k", linestyle="--", alpha=0.5, label="Pareto frontier"),
]
ax3.legend(handles=legend_handles, loc="lower right", fontsize=9,
           framealpha=0.9, edgecolor="#ccc")

plt.tight_layout()
out3 = "figures/final/classification_dim_tradeoff.png"
plt.savefig(out3, dpi=180, bbox_inches="tight")
print(f"Saved → {out3}")
plt.close()

print("All done.")
