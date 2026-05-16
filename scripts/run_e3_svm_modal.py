"""
E3 — SVM training on Modal (large CPU, 32 GB RAM).

Uploads the .npy feature files already extracted by run_e3_features_modal.py,
trains SVC(kernel='rbf', probability=True) on Modal's faster CPU, then
downloads all outputs (model, metrics, predictions, figures) back locally.

Prerequisites:
    modal run scripts/run_e3_features_modal.py   (features already extracted)

Usage:
    modal run scripts/run_e3_svm_modal.py

Downloads to local repo:
    models/classifiers/svm_E3.pkl
    results/metrics/classification_results.csv   (E3 row)
    results/metrics/E3_per_class_metrics.csv
    results/predictions/E3_deep_predictions.csv
    figures/classification/E3_confusion_matrix.png
    figures/classification/E3_confusion_matrix_norm.png
    figures/classification/E3_class_f1_bar.png

Changes v2 (expert audit):
    - Validation set now scaled + evaluated (val_accuracy, val_macro_f1 added to row)
    - Per-class AUC-ROC added (auc_{CLASS} for all 6 classes, computed on test set)
"""
import io
from datetime import datetime
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("e3-svm-training")

FEATURES_VOL = modal.Volume.from_name("e3-features-out")
RESULTS_VOL  = modal.Volume.from_name("e3-svm-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "joblib>=1.3.0",
    )
)

CLASSES  = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
DEEP_DIM = 1280
SEED     = 42


@app.function(
    image=image,
    cpu=8,
    memory=32768,   # 32 GB — RBF kernel matrix for 45k samples needs headroom
    timeout=7200,   # 2 hrs max
    volumes={
        "/features": FEATURES_VOL,
        "/results" : RESULTS_VOL,
    },
)
def train_svm():
    import time
    import numpy as np
    import pandas as pd
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        precision_score, recall_score, classification_report,
        confusion_matrix, matthews_corrcoef, cohen_kappa_score,
        roc_auc_score, top_k_accuracy_score,
    )

    FEATURES_DIR = Path("/features")
    RESULTS_DIR  = Path("/results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(exist_ok=True)

    # ── Load features ──────────────────────────────────────────────────────────
    print("[1/3] Loading features ...")
    X_train = np.load(str(FEATURES_DIR / "deep_train_X.npy"))
    y_train = np.load(str(FEATURES_DIR / "deep_train_y.npy"))
    X_val   = np.load(str(FEATURES_DIR / "deep_val_X.npy"))
    y_val   = np.load(str(FEATURES_DIR / "deep_val_y.npy"))
    X_test  = np.load(str(FEATURES_DIR / "deep_test_X.npy"))
    y_test  = np.load(str(FEATURES_DIR / "deep_test_y.npy"))
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)   # fit on train only — no data leakage
    X_test  = scaler.transform(X_test)

    # ── Train RBF SVM ──────────────────────────────────────────────────────────
    print(f"[2/3] Training SVC(kernel='rbf') on {len(y_train):,} samples ...")
    t0 = time.perf_counter()
    clf = SVC(C=1.0, kernel="rbf", gamma="scale",
              class_weight="balanced", probability=True,
              cache_size=8000, random_state=SEED)
    clf.fit(X_train, y_train)
    train_min = (time.perf_counter() - t0) / 60
    print(f"  Trained in {train_min:.1f} min")

    joblib.dump({"clf": clf, "scaler": scaler},
                str(RESULTS_DIR / "svm_E3.pkl"))
    print("  Saved: svm_E3.pkl")

    # ── Evaluate — validation set (overfitting check) ──────────────────────────
    print("[3a/3] Evaluating on validation split ...")
    y_pred_val = clf.predict(X_val)
    y_prob_val = clf.predict_proba(X_val)
    val_accuracy  = float(accuracy_score(y_val, y_pred_val))
    val_macro_f1  = float(f1_score(y_val, y_pred_val, average="macro", zero_division=0))
    print(f"  Val accuracy : {val_accuracy:.4f}")
    print(f"  Val macro-F1 : {val_macro_f1:.4f}")

    # ── Evaluate — test set (official metrics) ─────────────────────────────────
    print("[3b/3] Evaluating on test split ...")
    t_infer = time.perf_counter()
    y_pred  = clf.predict(X_test)
    infer_s = time.perf_counter() - t_infer
    y_prob  = clf.predict_proba(X_test)

    print(classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))

    N = len(y_test)
    n_cls = len(CLASSES)
    per_f1   = f1_score(y_test, y_pred, average=None, labels=list(range(n_cls)), zero_division=0)
    per_prec = precision_score(y_test, y_pred, average=None, labels=list(range(n_cls)), zero_division=0)
    per_rec  = recall_score(y_test, y_pred, average=None, labels=list(range(n_cls)), zero_division=0)

    try:
        auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        auc = float("nan")
    try:
        per_class_auc = list(roc_auc_score(y_test, y_prob, multi_class="ovr", average=None))
    except Exception:
        per_class_auc = [float("nan")] * n_cls
    try:
        top2 = float(top_k_accuracy_score(y_test, y_prob, k=2))
    except Exception:
        top2 = float("nan")

    row = {
        "experiment"           : "E3",
        "model"                : "SVC_rbf",
        "augmentation"         : "RandomHFlip+RandomVFlip+Rotation15+ColorJitter",
        "timestamp"            : datetime.now().strftime("%Y-%m-%d %H:%M"),
        "accuracy"             : round(float(accuracy_score(y_test, y_pred)), 4),
        "macro_f1"             : round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1"          : round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "balanced_accuracy"    : round(float(balanced_accuracy_score(y_test, y_pred)), 4),
        "mcc"                  : round(float(matthews_corrcoef(y_test, y_pred)), 4),
        "cohen_kappa"          : round(float(cohen_kappa_score(y_test, y_pred)), 4),
        "auc_roc_macro"        : round(auc, 4) if auc == auc else float("nan"),
        "top2_accuracy"        : round(top2, 4) if top2 == top2 else float("nan"),
        "val_accuracy"         : round(val_accuracy, 4),
        "val_macro_f1"         : round(val_macro_f1, 4),
        "feature_dim"          : DEEP_DIM,
        "n_train_samples"      : len(y_train),
        "inference_ms_per_crop": round((infer_s / N) * 1000, 4),
        "extraction_ms_per_crop": float("nan"),    # extracted in separate GPU job
        "extraction_throughput_cps": float("nan"),
        **{f"f1_{CLASSES[i]}"  : round(float(v), 4) for i, v in enumerate(per_f1)},
        **{f"prec_{CLASSES[i]}": round(float(v), 4) for i, v in enumerate(per_prec)},
        **{f"rec_{CLASSES[i]}" : round(float(v), 4) for i, v in enumerate(per_rec)},
        **{f"auc_{CLASSES[i]}" : round(float(v), 4) if v == v else float("nan")
           for i, v in enumerate(per_class_auc)},
    }

    # classification_results.csv
    csv_path = RESULTS_DIR / "classification_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df["experiment"] != "E3"]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)

    pd.DataFrame([{
        "experiment": "E3", "class": c,
        "f1": row[f"f1_{c}"], "precision": row[f"prec_{c}"], "recall": row[f"rec_{c}"],
        "auc": row[f"auc_{c}"],
    } for c in CLASSES]).to_csv(RESULTS_DIR / "E3_per_class_metrics.csv", index=False)

    # predictions CSV (no image paths available on Modal — use index)
    pd.DataFrame([{
        "index"     : i,
        "true_class": CLASSES[y_test[i]],
        "predicted" : CLASSES[y_pred[i]],
        "correct"   : bool(y_test[i] == y_pred[i]),
        "confidence": round(float(y_prob[i].max()), 4),
    } for i in range(N)]).to_csv(RESULTS_DIR / "E3_deep_predictions.csv", index=False)

    # Confusion matrix (raw)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(n_cls)))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("E3 — EfficientNetB0 + SVC_rbf")
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "figures" / "E3_confusion_matrix.png"), dpi=150)
    plt.close()

    # Confusion matrix (normalised)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("E3 — Normalised Confusion Matrix (SVC_rbf)")
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "figures" / "E3_confusion_matrix_norm.png"), dpi=150)
    plt.close()

    # Per-class F1 bar
    per_class_f1 = [row[f"f1_{c}"] for c in CLASSES]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(CLASSES, per_class_f1, color="steelblue")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 Score")
    ax.set_title("E3 Deep+SVC_rbf — Per-class F1")
    for bar, v in zip(bars, per_class_f1):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "figures" / "E3_class_f1_bar.png"), dpi=150)
    plt.close()

    RESULTS_VOL.commit()
    print("\nAll outputs saved to Modal Volume.")
    return row


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("E3  SVC(rbf) Training  [Modal 32GB CPU]")
    print("=" * 60)

    row = train_svm.remote()

    print("\n" + "=" * 60)
    print("E3  SVC_rbf  |  RESULTS")
    print("=" * 60)
    print(f"  Accuracy          : {row['accuracy']:.4f}")
    print(f"  Macro-F1          : {row['macro_f1']:.4f}")
    print(f"  Weighted-F1       : {row['weighted_f1']:.4f}")
    print(f"  Balanced Accuracy : {row['balanced_accuracy']:.4f}")
    print(f"  MCC               : {row['mcc']:.4f}")
    print(f"  Cohen Kappa       : {row['cohen_kappa']:.4f}")
    auc = row["auc_roc_macro"]
    print(f"  AUC-ROC (macro)   : {auc:.4f}" if auc == auc else
          "  AUC-ROC (macro)   : N/A")
    print(f"  Top-2 Accuracy    : {row['top2_accuracy']:.4f}")
    print(f"  Infer/crop        : {row['inference_ms_per_crop']:.3f} ms")
    print(f"\n  Overfitting check (val vs test):")
    print(f"    Val  macro-F1  : {row['val_macro_f1']:.4f}")
    print(f"    Test macro-F1  : {row['macro_f1']:.4f}")
    gap = abs(row['val_macro_f1'] - row['macro_f1'])
    print(f"    Gap            : {gap:.4f}  {'(OK)' if gap < 0.05 else '(WARNING: possible overfit)'}")
    print("\n  Per-class F1 / AUC:")
    for c in CLASSES:
        v   = row[f"f1_{c}"]
        au  = row[f"auc_{c}"]
        bar = "#" * int(v * 40)
        auc_str = f"{au:.4f}" if au == au else "  N/A"
        print(f"    {c:<16} F1={v:.4f}  AUC={auc_str}  {bar}")

    # ── Download all outputs ───────────────────────────────────────────────────
    vol = modal.Volume.from_name("e3-svm-results")

    downloads = {
        "svm_E3.pkl"                 : REPO_ROOT / "models" / "classifiers" / "svm_E3.pkl",
        "classification_results.csv" : REPO_ROOT / "results" / "metrics" / "classification_results.csv",
        "E3_per_class_metrics.csv"   : REPO_ROOT / "results" / "metrics" / "E3_per_class_metrics.csv",
        "E3_deep_predictions.csv"    : REPO_ROOT / "results" / "predictions" / "E3_deep_predictions.csv",
        "figures/E3_confusion_matrix.png"     : REPO_ROOT / "figures" / "classification" / "E3_confusion_matrix.png",
        "figures/E3_confusion_matrix_norm.png": REPO_ROOT / "figures" / "classification" / "E3_confusion_matrix_norm.png",
        "figures/E3_class_f1_bar.png"         : REPO_ROOT / "figures" / "classification" / "E3_class_f1_bar.png",
    }

    print("\nDownloading outputs ...")
    for remote, local in downloads.items():
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {remote} ...", end="", flush=True)

        # classification_results.csv: merge E3 row into existing local CSV
        if remote == "classification_results.csv" and local.exists():
            import pandas as pd
            buf = io.BytesIO()
            for chunk in vol.read_file(remote):
                buf.write(chunk)
            buf.seek(0)
            remote_df = pd.read_csv(buf)
            local_df  = pd.read_csv(local)
            local_df  = local_df[local_df["experiment"] != "E3"]
            merged    = pd.concat([local_df, remote_df[remote_df["experiment"] == "E3"]],
                                  ignore_index=True)
            merged.to_csv(local, index=False)
            print("  merged")
            continue

        with open(local, "wb") as f:
            for chunk in vol.read_file(remote):
                f.write(chunk)
        size = local.stat().st_size / 1024
        print(f"  {size:.0f} KB")

    print("\nE3 complete. All outputs in local repo.")
