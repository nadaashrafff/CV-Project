"""
Run the full preprocessing pipeline locally.

Usage:
    python scripts/run_preprocessing.py

Outputs:
    data/processed/dataset_split_70_15_15/   YOLO split + data.yaml
    data/processed/dataset_crops/            224x224 crops by class
    figures/preprocessing/                   Fig_01 through Fig_07
    results/metrics/01_cleaning_report.csv
    results/metrics/02_split_stats.csv
    results/metrics/03_crop_metadata.csv
"""
import sys, hashlib, yaml, re
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.seed import set_seed, SEED
from src.utils.paths import CLASSES
from src.utils.logger import get_logger
from src.data.split_dataset import build_image_inventory, make_split, get_dominant_class

set_seed()
logger = get_logger("preprocessing")

DATASET_ROOT = REPO_ROOT / "Trash_Dataset" / "GARBAGE CLASSIFICATION"
SPLIT_OUT    = REPO_ROOT / "data" / "processed" / "dataset_split_70_15_15"
CROPS_OUT    = REPO_ROOT / "data" / "processed" / "dataset_crops"
FIG_DIR      = REPO_ROOT / "figures" / "preprocessing"
METRICS_DIR  = REPO_ROOT / "results" / "metrics"

for d in [SPLIT_OUT, CROPS_OUT, FIG_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

HEX_LIST = ["#2ecc71", "#e67e22", "#3498db", "#95a5a6", "#f5cba7", "#9b59b6"]

assert DATASET_ROOT.exists(), f"Dataset not found: {DATASET_ROOT}"
logger.info(f"Dataset root: {DATASET_ROOT}")


# ─── STEP 1: AUDIT ───────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1 — DATASET AUDIT")
print("=" * 70)

img_exts  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
all_files = list(DATASET_ROOT.rglob("*"))
all_images = [f for f in all_files if f.is_file() and f.suffix.lower() in img_exts]
all_labels = [f for f in all_files if f.is_file() and f.suffix.lower() == ".txt"
              and "README" not in f.name]
print(f"  Images: {len(all_images)}   Labels: {len(all_labels)}")

for sp in ["train", "valid", "test"]:
    ni = len(list((DATASET_ROOT / sp / "images").glob("*.jpg")))
    nl = len(list((DATASET_ROOT / sp / "labels").glob("*.txt")))
    print(f"  {sp:<8}: {ni:>5} images  {nl:>5} labels")

with open(DATASET_ROOT / "data.yaml") as f:
    yaml_data = yaml.safe_load(f)
print(f"  YAML nc={yaml_data.get('nc')}  names={yaml_data.get('names')}")

# Fig_01: raw class distribution per vendor split
scc = {}
for sp in ["train", "valid", "test"]:
    c = Counter()
    for lf in (DATASET_ROOT / sp / "labels").glob("*.txt"):
        try:
            for ln in lf.read_text().strip().splitlines():
                p = ln.split()
                if p:
                    c[int(p[0])] += 1
        except Exception:
            pass
    scc[sp] = c

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Fig_01 — Raw Dataset: Box Count per Class per Vendor Split",
             fontsize=13, fontweight="bold")
for ax, sp in zip(axes, ["train", "valid", "test"]):
    vals = [scc[sp].get(i, 0) for i in range(6)]
    bars = ax.bar(range(6), vals, color=HEX_LIST, edgecolor="black")
    ax.set_title(f"{sp.upper()}  ({sum(vals):,} boxes)", fontweight="bold")
    ax.set_xticks(range(6))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Box count")
    for b in bars:
        h = b.get_height()
        if h > 0:
            ax.annotate(str(int(h)), xy=(b.get_x() + b.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "Fig_01_raw_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: Fig_01_raw_class_distribution.png")


# ─── STEP 2: CLEANING ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2 — DEEP CLEANING")
print("=" * 70)

df_raw = build_image_inventory(DATASET_ROOT)
print(f"  Raw inventory: {len(df_raw)} matched pairs")

zero_area_log       = []
cleaned_label_cache = {}
for _, row in df_raw.iterrows():
    lbl   = Path(row["label"])
    lines = lbl.read_text().strip().splitlines()
    good  = []
    for ln in lines:
        parts = ln.strip().split()
        if len(parts) == 5:
            try:
                w, h = float(parts[3]), float(parts[4])
                if w > 0 and h > 0:
                    good.append(ln)
                else:
                    zero_area_log.append({"image": row["stem"], "line": ln})
            except ValueError:
                pass
    cleaned_label_cache[row["stem"]] = (row["image"], lbl, good)
print(f"  Zero-area boxes: {len(zero_area_log)}")

corrupt_stems = set()
for stem, (img, _, _) in cleaned_label_cache.items():
    try:
        Image.open(img).verify()
    except Exception:
        corrupt_stems.add(stem)
print(f"  Corrupt images: {len(corrupt_stems)}")

md5_map = defaultdict(list)
for stem, (img, _, _) in cleaned_label_cache.items():
    if stem in corrupt_stems:
        continue
    try:
        h = hashlib.md5(Path(img).read_bytes()).hexdigest()
        md5_map[h].append(stem)
    except Exception:
        corrupt_stems.add(stem)
dup_stems = {s for stems in md5_map.values() for s in stems[1:]}
print(f"  Exact duplicates: {len(dup_stems)}")

skip = corrupt_stems | dup_stems
clean_records = []
for stem in sorted(cleaned_label_cache):
    if stem in skip:
        continue
    img, lbl, good = cleaned_label_cache[stem]
    if not good:
        continue
    dom = get_dominant_class(Path(lbl))
    if dom == -1:
        continue
    clean_records.append({"stem": stem, "image": img, "label": lbl, "dominant_class": dom})

df_clean  = pd.DataFrame(clean_records)
n_removed = len(df_raw) - len(df_clean)
print(f"  Clean pool: {len(df_clean)}  (removed {n_removed})")

pd.DataFrame([
    {"stage": "zero_area_boxes_removed", "count": len(zero_area_log)},
    {"stage": "corrupt_images",          "count": len(corrupt_stems)},
    {"stage": "duplicate_images",        "count": len(dup_stems)},
    {"stage": "total_removed",           "count": n_removed},
    {"stage": "clean_pool_size",         "count": len(df_clean)},
]).to_csv(METRICS_DIR / "01_cleaning_report.csv", index=False)
print(f"  Saved: 01_cleaning_report.csv")

# Fig_02: before/after cleaning
raw_dist   = Counter(df_raw["dominant_class"].tolist())
clean_dist = Counter(df_clean["dominant_class"].tolist())
x, w = np.arange(6), 0.35
fig, ax = plt.subplots(figsize=(12, 5))
b1 = ax.bar(x - w/2, [raw_dist.get(i, 0)   for i in range(6)], w,
            label="Before", color=HEX_LIST, edgecolor="black", alpha=0.55)
b2 = ax.bar(x + w/2, [clean_dist.get(i, 0) for i in range(6)], w,
            label="After",  color=HEX_LIST, edgecolor="black")
ax.set_title("Fig_02 — Before vs After Cleaning", fontsize=13, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=20, ha="right"); ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "Fig_02_before_after_cleaning.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: Fig_02_before_after_cleaning.png")


# ─── STEP 3: SPLIT ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3 — STRATIFIED 70/15/15 SPLIT")
print("=" * 70)

DATA_YAML = SPLIT_OUT / "data.yaml"
if DATA_YAML.exists():
    print(f"  Split already exists — skipping file copy.")
else:
    make_split(df_clean, SPLIT_OUT)
    yaml_cfg = {"path": str(SPLIT_OUT), "train": "train/images",
                "val": "val/images", "test": "test/images",
                "nc": 6, "names": CLASSES}
    with open(DATA_YAML, "w") as f:
        yaml.dump(yaml_cfg, f, default_flow_style=None, sort_keys=False)
    print(f"  data.yaml written: {DATA_YAML}")

split_rows = []
for sp in ["train", "val", "test"]:
    ni = len(list((SPLIT_OUT / sp / "images").glob("*.jpg")))
    nl = len(list((SPLIT_OUT / sp / "labels").glob("*.txt")))
    flag = "" if ni == nl else " <- MISMATCH"
    print(f"  {sp:<8}: {ni:>5} images  {nl:>5} labels{flag}")
    dom_counts = Counter()
    for lf in (SPLIT_OUT / sp / "labels").glob("*.txt"):
        try:
            d = get_dominant_class(lf)
            if d != -1:
                dom_counts[d] += 1
        except Exception:
            pass
    for i, cls in enumerate(CLASSES):
        split_rows.append({"split": sp, "class": cls, "dominant_images": dom_counts.get(i, 0)})

split_df = pd.DataFrame(split_rows)
split_df.to_csv(METRICS_DIR / "02_split_stats.csv", index=False)
print(f"  Saved: 02_split_stats.csv")

pivot = split_df.pivot(index="split", columns="class", values="dominant_images")\
                .reindex(["train", "val", "test"])[CLASSES]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Fig_03 — 70/15/15 Split: Class Distribution", fontsize=13, fontweight="bold")
xp, wp = np.arange(6), 0.25
for i, (sp, col) in enumerate(zip(["train", "val", "test"], ["#2980b9", "#27ae60", "#e74c3c"])):
    vals = [pivot.loc[sp, cls] for cls in CLASSES]
    axes[0].bar(xp + i*wp, vals, wp, label=sp.capitalize(), color=col, edgecolor="black")
axes[0].set_xticks(xp + wp); axes[0].set_xticklabels(CLASSES, rotation=20, ha="right")
axes[0].set_ylabel("Images"); axes[0].set_title("Grouped"); axes[0].legend()
pivot.plot(kind="bar", stacked=True, ax=axes[1], color=HEX_LIST, edgecolor="black")
axes[1].set_xticklabels(["Train", "Val", "Test"], rotation=0); axes[1].set_ylabel("Images")
axes[1].set_title("Stacked"); axes[1].legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(FIG_DIR / "Fig_03_split_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: Fig_03_split_class_distribution.png")


# ─── STEP 4: VISUAL VERIFICATION ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4 — VISUAL VERIFICATION")
print("=" * 70)

import random as _rnd
_rnd.seed(SEED)
train_imgs = list((SPLIT_OUT / "train" / "images").glob("*.jpg"))
samples    = _rnd.sample(train_imgs, min(5, len(train_imgs)))
train_lbls = SPLIT_OUT / "train" / "labels"

fig, axes = plt.subplots(1, 5, figsize=(25, 6))
fig.suptitle("Fig_04 — Train Samples with Ground-Truth Bounding Boxes",
             fontsize=13, fontweight="bold")
for ai, img_path in enumerate(samples):
    stem     = img_path.stem
    lbl_path = train_lbls / f"{stem}.txt"
    img      = Image.open(img_path).convert("RGB")
    draw     = ImageDraw.Draw(img)
    iw, ih   = img.size
    boxes    = []
    if lbl_path.exists():
        for ln in lbl_path.read_text().strip().splitlines():
            parts = ln.split()
            if len(parts) == 5:
                try:
                    cls = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])
                    x1 = max(0, int((xc - bw/2) * iw))
                    y1 = max(0, int((yc - bh/2) * ih))
                    x2 = min(iw, int((xc + bw/2) * iw))
                    y2 = min(ih, int((yc + bh/2) * ih))
                    draw.rectangle([x1, y1, x2, y2], outline=HEX_LIST[cls % 6], width=3)
                    draw.text((x1+2, y1+2), CLASSES[cls][:4], fill=HEX_LIST[cls % 6])
                    boxes.append(cls)
                except Exception:
                    pass
    axes[ai].imshow(img)
    axes[ai].set_title(f"{stem[:22]}\n{len(boxes)} boxes", fontsize=8)
    axes[ai].axis("off")
plt.tight_layout()
plt.savefig(FIG_DIR / "Fig_04_bbox_verification_samples.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: Fig_04_bbox_verification_samples.png")

print("\n  Class coverage check:")
for sp in ["train", "val", "test"]:
    cls_ids = set()
    for lf in (SPLIT_OUT / sp / "labels").glob("*.txt"):
        for ln in lf.read_text().strip().splitlines():
            p = ln.split()
            if p:
                try: cls_ids.add(int(p[0]))
                except ValueError: pass
    missing = set(range(6)) - cls_ids
    status  = "OK" if not missing else f"MISSING {sorted(missing)}"
    print(f"  {sp:<8}: {len(cls_ids)}/6 classes  [{status}]")


# ─── STEP 5: CROP EXTRACTION ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5 — CROP EXTRACTION (224x224)")
print("=" * 70)

CROP_SIZE       = (224, 224)
MIN_CROP_PIXELS = 16

existing_crops = list(CROPS_OUT.rglob("*.jpg"))
if existing_crops:
    print(f"  Crops already exist ({len(existing_crops):,}) — skipping extraction.")
    crop_records = []
    for sp in ["train", "val", "test"]:
        for cls in CLASSES:
            cd = CROPS_OUT / sp / cls
            if not cd.exists():
                continue
            for p in cd.glob("*.jpg"):
                crop_records.append({"split": sp, "class_name": cls,
                                     "crop_filename": p.name, "class_id": CLASSES.index(cls),
                                     "source_image": p.stem.rsplit("_obj", 1)[0], "bbox_px": []})
    crop_df     = pd.DataFrame(crop_records)
    total_crops = len(crop_df)
    rejected    = {"too_small": 0, "out_of_bounds": 0, "read_error": 0}
else:
    crop_records = []
    total_crops  = 0
    rejected     = {"too_small": 0, "out_of_bounds": 0, "read_error": 0}

    for sp in ["train", "val", "test"]:
        img_dir  = SPLIT_OUT / sp / "images"
        lbl_dir  = SPLIT_OUT / sp / "labels"
        sp_count = 0

        for cls in CLASSES:
            (CROPS_OUT / sp / cls).mkdir(parents=True, exist_ok=True)

        for img_path in sorted(img_dir.glob("*.jpg")):
            stem     = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"
            if not lbl_path.exists():
                continue
            try:
                img_obj     = Image.open(img_path).convert("RGB")
                iw, ih      = img_obj.size
            except Exception:
                rejected["read_error"] += 1
                continue

            obj_idx = 0
            for ln in lbl_path.read_text().strip().splitlines():
                parts = ln.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    cid = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])
                except ValueError:
                    continue
                if cid >= len(CLASSES):
                    continue
                x1 = max(0, int((xc - bw/2) * iw))
                y1 = max(0, int((yc - bh/2) * ih))
                x2 = min(iw, int((xc + bw/2) * iw))
                y2 = min(ih, int((yc + bh/2) * ih))
                if (x2 - x1) < MIN_CROP_PIXELS or (y2 - y1) < MIN_CROP_PIXELS:
                    rejected["too_small"] += 1
                    continue
                if x2 <= x1 or y2 <= y1:
                    rejected["out_of_bounds"] += 1
                    continue
                cls_name = CLASSES[cid]
                crop     = img_obj.crop((x1, y1, x2, y2)).resize(CROP_SIZE, Image.Resampling.BILINEAR)
                fname    = f"{stem}_obj{obj_idx:02d}_cls{cid}.jpg"
                crop.save(CROPS_OUT / sp / cls_name / fname, quality=95)
                crop_records.append({"split": sp, "source_image": stem, "crop_filename": fname,
                                     "class_id": cid, "class_name": cls_name,
                                     "bbox_px": [x1, y1, x2, y2]})
                obj_idx     += 1
                sp_count    += 1
                total_crops += 1

        print(f"  {sp:<6}: {sp_count:>6} crops")

    crop_df = pd.DataFrame(crop_records)

print(f"  Total: {total_crops:,}  rejected: {sum(rejected.values()):,}")
crop_df.to_csv(METRICS_DIR / "03_crop_metadata.csv", index=False)
print(f"  Saved: 03_crop_metadata.csv")

overall = crop_df["class_name"].value_counts().reindex(CLASSES, fill_value=0)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(6), overall.values, color=HEX_LIST, edgecolor="black")
ax.set_title("Fig_05 — Crop Count per Class", fontsize=13, fontweight="bold")
ax.set_xticks(range(6)); ax.set_xticklabels(CLASSES, rotation=20, ha="right")
ax.set_ylabel("Crop count")
for b in bars:
    h = b.get_height()
    if h > 0:
        ax.annotate(f"{int(h):,}", xy=(b.get_x() + b.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "Fig_05_crop_count_per_class.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: Fig_05_crop_count_per_class.png")

scc2 = (crop_df.groupby(["split", "class_name"]).size()
        .unstack(fill_value=0)
        .reindex(["train", "val", "test"])
        .reindex(columns=CLASSES, fill_value=0))
fig, ax = plt.subplots(figsize=(8, 5))
scc2.plot(kind="bar", stacked=True, ax=ax, color=HEX_LIST, edgecolor="black")
ax.set_title("Fig_06 — Crop Count per Split", fontsize=13, fontweight="bold")
ax.set_xticklabels(["Train", "Val", "Test"], rotation=0); ax.set_ylabel("Crop count")
ax.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(FIG_DIR / "Fig_06_crop_count_per_split.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: Fig_06_crop_count_per_split.png")

fig, axes = plt.subplots(6, 6, figsize=(18, 18))
fig.suptitle("Fig_07 — Sample Crops: 6 per Class (train)", fontsize=14, fontweight="bold")
for ri, cls in enumerate(CLASSES):
    crop_dir   = CROPS_OUT / "train" / cls
    all_crops  = sorted(crop_dir.glob("*.jpg")) if crop_dir.exists() else []
    crop_samps = all_crops[:6]
    for ci, cp in enumerate(crop_samps):
        axes[ri, ci].imshow(Image.open(cp))
        axes[ri, ci].set_title(f"{cls}\n{cp.stem[-18:]}", fontsize=6)
        axes[ri, ci].axis("off")
    for ci in range(len(crop_samps), 6):
        axes[ri, ci].axis("off")
plt.tight_layout()
plt.savefig(FIG_DIR / "Fig_07_sample_crops_per_class.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"  Saved: Fig_07_sample_crops_per_class.png")


# ─── SUMMARY ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)
train_n = len(list((SPLIT_OUT / "train" / "images").glob("*.jpg")))
val_n   = len(list((SPLIT_OUT / "val"   / "images").glob("*.jpg")))
test_n  = len(list((SPLIT_OUT / "test"  / "images").glob("*.jpg")))
print(f"  Raw pool       : {len(df_raw)}")
print(f"  Clean pool     : {len(df_clean)}  (removed {len(df_raw)-len(df_clean)})")
print(f"  Split  train   : {train_n}")
print(f"         val     : {val_n}")
print(f"         test    : {test_n}")
print(f"  Crops          : {total_crops:,}")
print(f"  data.yaml      : {DATA_YAML}")
print(f"  Figures        : {FIG_DIR}")
print(f"  Reports        : {METRICS_DIR}")
print("\n  Ready for: scripts/run_yolo_e1.py")
