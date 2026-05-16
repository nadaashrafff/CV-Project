"""Classification metrics for E2–E10.

Required by project spec (Track B):
  accuracy, macro-F1, per-class F1, confusion matrix,
  feature dimension, feature extraction time, classification time

Additional metrics for comprehensive evaluation:
  weighted-F1, balanced accuracy, MCC, Cohen's kappa,
  AUC-ROC (macro OvR), per-class precision, per-class recall,
  top-2 accuracy, feature extraction throughput (crops/sec)
"""
import time
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    top_k_accuracy_score,
)

CLASSES = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
N_CLASSES = len(CLASSES)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,       # (N, C) probability matrix — required for AUC-ROC
    experiment_id: str,
    model_name: str,
    feature_dim: int,
    extraction_time_s: float,        # total feature extraction time in seconds
    n_train_samples: int,
    classification_time_s: float,    # total classification time on test set in seconds
    n_test_samples: int,
    augmentation: str = "none",
) -> dict:
    """Return a flat dict of all metrics, ready to append to a CSV row."""

    infer_ms  = (classification_time_s / n_test_samples) * 1000
    extract_ms = (extraction_time_s / n_train_samples) * 1000 if n_train_samples > 0 else float("nan")
    extract_throughput = n_train_samples / max(extraction_time_s, 1e-9)  # crops/sec

    per_cls_f1   = f1_score(y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0)
    per_cls_prec = precision_score(y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0)
    per_cls_rec  = recall_score(y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0)

    # AUC-ROC requires probability estimates
    auc_macro = float("nan")
    if y_prob is not None:
        try:
            auc_macro = float(roc_auc_score(y_true, y_prob, multi_class="ovr",
                                             average="macro"))
        except Exception:
            pass

    # Top-2 accuracy
    top2_acc = float("nan")
    if y_prob is not None:
        try:
            top2_acc = float(top_k_accuracy_score(y_true, y_prob, k=2))
        except Exception:
            pass

    row = {
        # ── Identity ────────────────────────────────────────────────────────
        "experiment"           : experiment_id,
        "model"                : model_name,
        "augmentation"         : augmentation,

        # ── Core metrics (project spec) ──────────────────────────────────────
        "accuracy"             : round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1"             : round(float(f1_score(y_true, y_pred, average="macro",  zero_division=0)), 4),
        "weighted_f1"          : round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),

        # ── Imbalance-robust metrics ─────────────────────────────────────────
        "balanced_accuracy"    : round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "mcc"                  : round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "cohen_kappa"          : round(float(cohen_kappa_score(y_true, y_pred)), 4),
        "auc_roc_macro"        : round(auc_macro, 4) if not np.isnan(auc_macro) else float("nan"),
        "top2_accuracy"        : round(top2_acc, 4) if not np.isnan(top2_acc) else float("nan"),

        # ── Efficiency (project spec) ────────────────────────────────────────
        "feature_dim"          : feature_dim,
        "n_train_samples"      : n_train_samples,
        "inference_ms_per_crop": round(infer_ms, 4),
        "extraction_ms_per_crop": round(extract_ms, 4),
        "extraction_throughput_cps": round(extract_throughput, 1),  # crops per second

        # ── Per-class F1 ─────────────────────────────────────────────────────
        **{f"f1_{CLASSES[i]}": round(float(v), 4)  for i, v in enumerate(per_cls_f1)},

        # ── Per-class Precision ───────────────────────────────────────────────
        **{f"prec_{CLASSES[i]}": round(float(v), 4) for i, v in enumerate(per_cls_prec)},

        # ── Per-class Recall ──────────────────────────────────────────────────
        **{f"rec_{CLASSES[i]}": round(float(v), 4)  for i, v in enumerate(per_cls_rec)},
    }
    return row


def print_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))


def save_metrics_row(row: dict, csv_path: str, experiment_id: str) -> None:
    """Append or replace the row for experiment_id in the CSV."""
    import pandas as pd
    from pathlib import Path
    p = Path(csv_path)
    if p.exists():
        df = pd.read_csv(p)
        df = df[df["experiment"] != experiment_id]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(p, index=False)
