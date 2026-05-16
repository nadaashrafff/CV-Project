"""
E3 — Deep Features (EfficientNetB0) + SVM

Loads 1,280-dim features saved by run_e3_features_gpu.py, scales them,
trains SVC (RBF), evaluates with full metric suite, saves all outputs.

Prerequisites:
    python scripts/run_e3_features_gpu.py   (on Colab T4 or GPU)

Usage:
    python scripts/run_e3_svm.py
    python scripts/run_e3_svm.py --linear   # LinearSVC if RBF too slow on CPU

Note on training time:
    RBF SVC on ~35k samples can take 2–6 hrs on CPU.
    Use --linear for a fast linear approximation.

Saves:
    models/classifiers/svm_E3.pkl
    results/metrics/classification_results.csv     (E3 row)
    results/metrics/E3_per_class_metrics.csv
    results/predictions/E3_deep_predictions.csv
    figures/classification/E3_confusion_matrix.png
    figures/classification/E3_confusion_matrix_norm.png
    figures/classification/E3_class_f1_bar.png
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
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.evaluation.classification_metrics import (
    compute_all_metrics, print_report, get_confusion_matrix, save_metrics_row,
    CLASSES,
)
from src.evaluation.plots import plot_confusion_matrix, plot_per_class_f1
from src.features.deep_features import DEEP_DIM
from src.utils.seed import set_seed, SEED
from src.utils.logger import get_logger

set_seed()
logger = get_logger("e3_svm")

FEATURES_DIR = REPO_ROOT / "data" / "processed" / "features"
MODELS_DIR   = REPO_ROOT / "models" / "classifiers"
RESULTS_DIR  = REPO_ROOT / "results" / "metrics"
PREDS_DIR    = REPO_ROOT / "results" / "predictions"
FIGS_DIR     = REPO_ROOT / "figures" / "classification"

for d in [MODELS_DIR, RESULTS_DIR, PREDS_DIR, FIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_features(split: str):
    X_path = FEATURES_DIR / f"deep_{split}_X.npy"
    y_path = FEATURES_DIR / f"deep_{split}_y.npy"
    assert X_path.exists(), (
        f"Features not found: {X_path}\n"
        "Run first on GPU: python scripts/run_e3_features_gpu.py"
    )
    return np.load(X_path), np.load(y_path)


def load_crop_paths(split: str):
    crops_dir = REPO_ROOT / "data" / "processed" / "dataset_crops"
    paths = []
    for cls in CLASSES:
        cls_dir = crops_dir / split / cls
        if cls_dir.exists():
            paths.extend(sorted(str(p) for p in cls_dir.glob("*.jpg")))
    return paths


def main(use_linear: bool = False):
    print("=" * 70)
    print("E3  EfficientNetB0 Deep Features + SVM")
    print("=" * 70)

    # ── 1. Load features ──────────────────────────────────────────────────────
    print("\n[1/3] LOADING FEATURES")
    X_train, y_train = load_features("train")
    X_val,   y_val   = load_features("val")
    X_test,  y_test  = load_features("test")
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    assert X_train.shape[1] == DEEP_DIM, \
        f"Expected {DEEP_DIM} dims, got {X_train.shape[1]}"

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── 2. Train ──────────────────────────────────────────────────────────────
    print("\n[2/3] TRAINING SVM")
    t0 = time.perf_counter()
    if use_linear:
        print("  LinearSVC + CalibratedClassifierCV (gives probabilities for AUC-ROC)")
        base = LinearSVC(C=1.0, class_weight="balanced",
                         max_iter=3000, random_state=SEED)
        clf = CalibratedClassifierCV(base, cv=5, method="sigmoid")
        model_name = "LinearSVC_calibrated"
        y_prob = None          # assigned after fit
    else:
        print(f"  SVC(kernel='rbf') on {len(y_train):,} samples — may take hours on CPU")
        clf = SVC(C=1.0, kernel="rbf", gamma="scale",
                  class_weight="balanced", probability=True,
                  cache_size=2000, random_state=SEED)
        model_name = "SVC_rbf"

    clf.fit(X_train, y_train)
    train_min = (time.perf_counter() - t0) / 60
    print(f"  Trained in {train_min:.1f} min")

    joblib.dump({"clf": clf, "scaler": scaler}, str(MODELS_DIR / "svm_E3.pkl"))
    print(f"  Saved: models/classifiers/svm_E3.pkl")

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    print("\n[3/3] EVALUATION  (test split)")
    t_infer = time.perf_counter()
    y_pred  = clf.predict(X_test)
    infer_s = time.perf_counter() - t_infer

    if not use_linear:
        y_prob = clf.predict_proba(X_test)
    else:
        y_prob = None

    print_report(y_test, y_pred)

    row = compute_all_metrics(
        y_true=y_test, y_pred=y_pred, y_prob=y_prob,
        experiment_id="E3", model_name=model_name,
        feature_dim=DEEP_DIM,
        extraction_time_s=0.0,        # extraction was done separately on GPU
        n_train_samples=len(y_train),
        classification_time_s=infer_s,
        n_test_samples=len(y_test),
        augmentation="RandomHFlip+RandomVFlip+Rotation15+ColorJitter",
    )
    row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    save_metrics_row(row, str(RESULTS_DIR / "classification_results.csv"), "E3")
    print(f"  Saved: classification_results.csv")

    pd.DataFrame([{
        "experiment": "E3", "class": cls,
        "f1"        : row[f"f1_{cls}"],
        "precision" : row[f"prec_{cls}"],
        "recall"    : row[f"rec_{cls}"],
    } for cls in CLASSES]).to_csv(RESULTS_DIR / "E3_per_class_metrics.csv", index=False)
    print("  Saved: E3_per_class_metrics.csv")

    test_paths = load_crop_paths("test")
    pred_rows = [{
        "image"     : Path(p).name,
        "true_class": CLASSES[y_test[i]],
        "predicted" : CLASSES[y_pred[i]],
        "correct"   : bool(y_test[i] == y_pred[i]),
        "confidence": round(float(y_prob[i].max()), 4) if y_prob is not None else float("nan"),
    } for i, p in enumerate(test_paths)]
    pd.DataFrame(pred_rows).to_csv(
        PREDS_DIR / "E3_deep_predictions.csv", index=False)
    print("  Saved: E3_deep_predictions.csv")

    cm = get_confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, f"E3 — EfficientNetB0 + {model_name}",
                          str(FIGS_DIR / "E3_confusion_matrix.png"))

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"E3 — Normalised Confusion Matrix ({model_name})")
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "E3_confusion_matrix_norm.png"), dpi=150)
    plt.close()
    print("  Saved: E3_confusion_matrix.png  +  E3_confusion_matrix_norm.png")

    per_class_f1 = [row[f"f1_{c}"] for c in CLASSES]
    plot_per_class_f1({f"E3 Deep+{model_name}": per_class_f1},
                      str(FIGS_DIR / "E3_class_f1_bar.png"))
    print("  Saved: E3_class_f1_bar.png")

    print("\n" + "=" * 70)
    print(f"E3  EfficientNetB0 + {model_name}  |  RESULTS")
    print("=" * 70)
    print(f"  Accuracy          : {row['accuracy']:.4f}")
    print(f"  Macro-F1          : {row['macro_f1']:.4f}")
    print(f"  Weighted-F1       : {row['weighted_f1']:.4f}")
    print(f"  Balanced Accuracy : {row['balanced_accuracy']:.4f}")
    print(f"  MCC               : {row['mcc']:.4f}")
    print(f"  Cohen Kappa       : {row['cohen_kappa']:.4f}")
    auc = row['auc_roc_macro']
    print(f"  AUC-ROC (macro)   : {auc:.4f}" if auc == auc else
          "  AUC-ROC (macro)   : N/A (LinearSVC, no probabilities)")
    print(f"  Top-2 Accuracy    : {row['top2_accuracy']:.4f}"
          if row['top2_accuracy'] == row['top2_accuracy'] else
          "  Top-2 Accuracy    : N/A")
    print(f"  Infer/crop        : {row['inference_ms_per_crop']:.3f} ms")
    print(f"  Feature dim       : {DEEP_DIM}")
    print()
    print("  Per-class F1:")
    for c in CLASSES:
        v = row[f"f1_{c}"]
        bar = "#" * int(v * 40)
        print(f"    {c:<16} {v:.4f}  {bar}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", action="store_true",
                    help="Use LinearSVC instead of RBF SVM (much faster)")
    args = ap.parse_args()
    main(use_linear=args.linear)
