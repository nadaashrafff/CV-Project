"""
E2 — Classical Features + Random Forest  (with training augmentation)

Augmentation strategy (training split only):
  Each crop yields 5 feature vectors:
    1. Original
    2. Horizontal flip
    3. 90° rotation
    4. 180° rotation
    5. 270° rotation
  Rationale: waste objects appear in arbitrary orientations; rotational and
  flip augmentation directly addresses this without distorting texture/shape
  descriptors.  Val/test use original only (clean evaluation).

Pipeline:
  1. Extract + augment 252-dim classical features from all crops
  2. Cache .npy files — skip extraction if cache exists (--force to redo)
  3. Train Random Forest (200 estimators, class_weight='balanced')
  4. Evaluate on test split — full metric suite
  5. Save model, metrics, predictions, confusion matrix, F1 bar chart

Prerequisites:
  python scripts/run_preprocessing.py

Usage:
  python scripts/run_e2_classical.py
  python scripts/run_e2_classical.py --force   # re-extract features

Saves:
  data/processed/features/classical_{train,val,test}_{X,y}.npy
  models/classifiers/random_forest_E2.pkl
  results/metrics/classification_results.csv
  results/metrics/E2_per_class_metrics.csv
  results/predictions/E2_classical_predictions.csv
  figures/classification/E2_confusion_matrix.png
  figures/classification/E2_confusion_matrix_norm.png
  figures/classification/E2_class_f1_bar.png
"""
import sys, time, argparse
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from PIL import Image
from src.features.classical_features import extract_classical_from_array, FEATURE_DIM
from src.models.train_random_forest import train as train_rf, save as save_rf
from src.evaluation.classification_metrics import (
    compute_all_metrics, print_report, get_confusion_matrix, save_metrics_row,
    CLASSES,
)
from src.evaluation.plots import plot_confusion_matrix, plot_per_class_f1
from src.utils.paths import CLASS_TO_IDX
from src.utils.seed import set_seed
from src.utils.logger import get_logger

set_seed()
logger = get_logger("e2_classical")

# ── Paths ─────────────────────────────────────────────────────────────────────
CROPS_DIR    = REPO_ROOT / "data" / "processed" / "dataset_crops"
FEATURES_DIR = REPO_ROOT / "data" / "processed" / "features"
MODELS_DIR   = REPO_ROOT / "models" / "classifiers"
RESULTS_DIR  = REPO_ROOT / "results" / "metrics"
PREDS_DIR    = REPO_ROOT / "results" / "predictions"
FIGS_DIR     = REPO_ROOT / "figures" / "classification"

for d in [FEATURES_DIR, MODELS_DIR, RESULTS_DIR, PREDS_DIR, FIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "random_forest_E2.pkl"

# ── Augmentation ──────────────────────────────────────────────────────────────
def augment_variants(img: np.ndarray) -> list:
    """5 views: original + hflip + 3 rotations. Training split only."""
    return [
        img,
        np.fliplr(img),
        np.rot90(img, k=1),
        np.rot90(img, k=2),
        np.rot90(img, k=3),
    ]

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_crop_paths(split: str):
    paths, labels = [], []
    for cls in CLASSES:
        cls_dir = CROPS_DIR / split / cls
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.glob("*.jpg")):
            paths.append(str(p))
            labels.append(CLASS_TO_IDX[cls])
    return paths, labels


def extract_split(split: str, force: bool = False):
    X_path = FEATURES_DIR / f"classical_{split}_X.npy"
    y_path = FEATURES_DIR / f"classical_{split}_y.npy"

    if X_path.exists() and y_path.exists() and not force:
        logger.info(f"[{split}] Cache found — loading {X_path.name}")
        return np.load(X_path), np.load(y_path), 0.0

    paths, labels = load_crop_paths(split)
    assert paths, f"No crops found in {CROPS_DIR / split}"

    augment   = (split == "train")
    n_variants = 5 if augment else 1
    logger.info(f"[{split}] {len(paths):,} crops × {n_variants} = "
                f"{len(paths)*n_variants:,} vectors ...")

    X_list, y_list = [], []
    errors = 0
    t0 = time.perf_counter()

    for i, (p, label) in enumerate(zip(paths, labels)):
        try:
            img      = np.array(Image.open(p).convert("RGB").resize((224, 224)))
            variants = augment_variants(img) if augment else [img]
            for v in variants:
                X_list.append(extract_classical_from_array(v))
                y_list.append(label)
        except Exception as e:
            logger.warning(f"  skip {Path(p).name}: {e}")
            errors += 1

        if (i + 1) % 2000 == 0 or (i + 1) == len(paths):
            elapsed = time.perf_counter() - t0
            rate    = (i + 1) / max(elapsed, 1e-6)
            eta     = (len(paths) - i - 1) / rate
            print(f"  {i+1:>6}/{len(paths)}  {rate:.0f} crops/s  "
                  f"ETA {eta/60:.1f} min")

    elapsed = time.perf_counter() - t0
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list,  dtype=np.int32)
    np.save(X_path, X)
    np.save(y_path, y)
    logger.info(f"[{split}] {X.shape}  {elapsed/60:.1f} min  ({errors} err)")
    return X, y, elapsed


# ── Main ──────────────────────────────────────────────────────────────────────
def main(force: bool = False):
    print("=" * 70)
    print("E2  Classical Features + Random Forest  [augmented training]")
    print("=" * 70)

    assert CROPS_DIR.exists(), (
        f"Crops not found: {CROPS_DIR}\n"
        "Run: python scripts/run_preprocessing.py"
    )

    # ── 1. Feature extraction ─────────────────────────────────────────────────
    print("\n[1/3] FEATURE EXTRACTION")
    X_train, y_train, ext_time_s = extract_split("train", force)
    X_val,   y_val,   _          = extract_split("val",   force)
    X_test,  y_test,  _          = extract_split("test",  force)
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # ── 2. Train ──────────────────────────────────────────────────────────────
    print("\n[2/3] TRAINING RANDOM FOREST")
    t0  = time.perf_counter()
    clf = train_rf(X_train, y_train)
    print(f"  Trained in {time.perf_counter()-t0:.1f} s")
    save_rf(clf, str(MODEL_PATH))
    print(f"  Saved: {MODEL_PATH}")

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    print("\n[3/3] EVALUATION  (test split — no augmentation)")
    t_infer_start = time.perf_counter()
    y_pred = clf.predict(X_test)
    infer_time_s = time.perf_counter() - t_infer_start

    # Probability estimates for AUC-ROC
    y_prob = clf.predict_proba(X_test)

    print_report(y_test, y_pred)

    row = compute_all_metrics(
        y_true=y_test, y_pred=y_pred, y_prob=y_prob,
        experiment_id="E2", model_name="RandomForest",
        feature_dim=FEATURE_DIM,
        extraction_time_s=ext_time_s,
        n_train_samples=len(y_train),
        classification_time_s=infer_time_s,
        n_test_samples=len(y_test),
        augmentation="hflip+rot90+rot180+rot270",
    )
    row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    # classification_results.csv
    save_metrics_row(row, str(RESULTS_DIR / "classification_results.csv"), "E2")
    print(f"  Saved: classification_results.csv")

    # per-class detailed CSV
    per_class_rows = []
    for cls in CLASSES:
        per_class_rows.append({
            "experiment": "E2", "class": cls,
            "f1"       : row[f"f1_{cls}"],
            "precision": row[f"prec_{cls}"],
            "recall"   : row[f"rec_{cls}"],
        })
    pd.DataFrame(per_class_rows).to_csv(
        RESULTS_DIR / "E2_per_class_metrics.csv", index=False)
    print("  Saved: E2_per_class_metrics.csv")

    # predictions CSV
    test_paths, _ = load_crop_paths("test")
    pd.DataFrame([{
        "image"     : Path(p).name,
        "true_class": CLASSES[y_test[i]],
        "predicted" : CLASSES[y_pred[i]],
        "correct"   : bool(y_test[i] == y_pred[i]),
        "confidence": round(float(y_prob[i].max()), 4),
    } for i, p in enumerate(test_paths)]
    ).to_csv(PREDS_DIR / "E2_classical_predictions.csv", index=False)
    print("  Saved: E2_classical_predictions.csv")

    # confusion matrix (raw counts)
    cm = get_confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "E2 — Classical Features + Random Forest",
                          str(FIGS_DIR / "E2_confusion_matrix.png"))

    # confusion matrix (row-normalised %)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("E2 — Normalised Confusion Matrix")
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "E2_confusion_matrix_norm.png"), dpi=150)
    plt.close()
    print("  Saved: E2_confusion_matrix.png  +  E2_confusion_matrix_norm.png")

    per_class_f1 = [row[f"f1_{c}"] for c in CLASSES]
    plot_per_class_f1({"E2 Classical+RF": per_class_f1},
                      str(FIGS_DIR / "E2_class_f1_bar.png"))
    print("  Saved: E2_class_f1_bar.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("E2  Classical + Random Forest  |  RESULTS")
    print("=" * 70)
    print(f"  Accuracy          : {row['accuracy']:.4f}")
    print(f"  Macro-F1          : {row['macro_f1']:.4f}")
    print(f"  Weighted-F1       : {row['weighted_f1']:.4f}")
    print(f"  Balanced Accuracy : {row['balanced_accuracy']:.4f}")
    print(f"  MCC               : {row['mcc']:.4f}")
    print(f"  Cohen Kappa       : {row['cohen_kappa']:.4f}")
    print(f"  AUC-ROC (macro)   : {row['auc_roc_macro']:.4f}")
    print(f"  Top-2 Accuracy    : {row['top2_accuracy']:.4f}")
    print(f"  Infer/crop        : {row['inference_ms_per_crop']:.3f} ms")
    print(f"  Extract throughput: {row['extraction_throughput_cps']:.0f} crops/s")
    print(f"  Feature dim       : {FEATURE_DIM}")
    print(f"  Train size        : {row['n_train_samples']:,}  (5× aug)")
    print()
    print("  Per-class F1:")
    for c in CLASSES:
        v = row[f"f1_{c}"]
        bar = "#" * int(v * 40)
        print(f"    {c:<16} {v:.4f}  {bar}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Re-extract features even if cache exists")
    args = ap.parse_args()
    main(force=args.force)
