"""
Classification Ablation Study Figure
Generates figures/final/classification_ablation.png
Run: python3 notebooks/generate_classification_ablation.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

os.makedirs("figures/final", exist_ok=True)

# ── Data ────────────────────────────────────────────────────────────────────
# (sorted ascending so horizontal bars read best → worst top to bottom)
experiments = [
    # name                      macro_f1   category       modalities
    ("E2: Classical RF",         0.6476,   "single",      "Classical (252-dim)"),
    ("E4: Raw Concat",           0.6960,   "early",       "Classical + Deep"),
    ("E7: Feat. Selection",      0.7081,   "early",       "Top-200 (Classical+Deep)"),
    ("E11: YOLO+Classical",      0.7345,   "hybrid",      "YOLO SPPF + Classical"),
    ("E6: Autoencoder",          0.7705,   "early",       "AE-256 (Classical+Deep)"),
    ("E10: Attention Fusion",    0.7752,   "late",        "Gated (Classical+Deep)"),
    ("E8: Avg Voting",           0.7820,   "late",        "Classical + Deep (late)"),
    ("E9: Weighted Voting",      0.7832,   "late",        "Classical + Deep (late)"),
    ("E3: Deep SVM",             0.7848,   "single",      "EfficientNetB0 (1280-dim)"),
    ("E5: PCA Fusion",           0.8071,   "early",       "Classical + Deep (PCA)"),
    ("E12: Three-Way Hybrid",    0.8130,   "hybrid",      "YOLO + Deep + Classical"),
]

labels   = [e[0] for e in experiments]
values   = [e[1] for e in experiments]
cats     = [e[2] for e in experiments]
modals   = [e[3] for e in experiments]

PALETTE = {
    "single": "#4C72B0",   # blue
    "early":  "#55A868",   # green
    "late":   "#C44E52",   # red
    "hybrid": "#DD8452",   # orange
}
colors = [PALETTE[c] for c in cats]

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FAFAFA")

y_pos = np.arange(len(labels))
bars = ax.barh(y_pos, values, height=0.62, color=colors, zorder=3,
               edgecolor="white", linewidth=0.6)

# reference lines
for xv, lbl, ls in [
    (0.6476, "E2 baseline\n(classical only)", ":"),
    (0.7848, "E3 single-deep", "--"),
    (0.8071, "E5 two-way PCA", "--"),
]:
    ax.axvline(xv, color="#888888", linewidth=0.9, linestyle=ls, zorder=2)
    ax.text(xv + 0.001, len(labels) - 0.15, lbl,
            fontsize=6.5, color="#666666", va="top", ha="left",
            style="italic")

# value labels on bars
for i, (bar, val) in enumerate(zip(bars, values)):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=8.5,
            fontweight="bold" if val >= 0.8071 else "normal",
            color="#222222")

# modality sub-labels inside bar
for i, (bar, modal) in enumerate(zip(bars, modals)):
    ax.text(0.637, bar.get_y() + bar.get_height() / 2,
            modal, va="center", ha="left", fontsize=6.8, color="white",
            alpha=0.92)

# ── Δ annotations for top-3 ──────────────────────────────────────────────────
# E12 vs E5
ax.annotate("", xy=(0.8130, 10), xytext=(0.8071, 10),
            arrowprops=dict(arrowstyle="<->", color="#DD8452", lw=1.5))
ax.text(0.8100, 10.38, "+0.0059", ha="center", va="bottom",
        fontsize=7.5, color="#DD8452", fontweight="bold")

# E5 vs E3
ax.annotate("", xy=(0.8071, 9), xytext=(0.7848, 9),
            arrowprops=dict(arrowstyle="<->", color="#55A868", lw=1.5))
ax.text(0.7960, 9.38, "+0.0223", ha="center", va="bottom",
        fontsize=7.5, color="#55A868", fontweight="bold")

# ── Axes styling ─────────────────────────────────────────────────────────────
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel("Macro-F1 Score (test set)", fontsize=11)
ax.set_title("Classification Track Ablation Study\nAll Experiments — Ranked by Macro-F1",
             fontsize=13, fontweight="bold", pad=14)
ax.set_xlim(0.63, 0.845)
ax.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=PALETTE["single"], label="Single modality"),
    mpatches.Patch(color=PALETTE["early"],  label="Early fusion"),
    mpatches.Patch(color=PALETTE["late"],   label="Late fusion"),
    mpatches.Patch(color=PALETTE["hybrid"], label="Hybrid (YOLO)"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
          framealpha=0.85, edgecolor="#cccccc")

plt.tight_layout()
out = "figures/final/classification_ablation.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved → {out}")
plt.close()
