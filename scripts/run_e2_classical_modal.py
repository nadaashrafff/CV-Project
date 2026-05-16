"""
E2 — Classical Features (252-dim) + Random Forest  [Modal 16-core CPU, 32 GB RAM]

Feature extraction uses joblib.Parallel across all 16 cores.
Gabor kernels are pre-computed once at module level to avoid per-crop overhead.
Reuses dataset_crops.zip already in the 'e3-crops-v1' Volume from E3.

Prerequisites:
    modal run scripts/run_e3_features_modal.py   (uploads crops — already done)

Usage:
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run scripts/run_e2_classical_modal.py

Downloads to local repo:
    data/processed/features/classical_train_X.npy
    data/processed/features/classical_train_y.npy
    data/processed/features/classical_val_X.npy
    data/processed/features/classical_val_y.npy
    data/processed/features/classical_test_X.npy
    data/processed/features/classical_test_y.npy
    data/processed/features/classical_feature_manifest.json
    models/classifiers/random_forest_E2.pkl
    results/metrics/classification_results.csv   (E2 row merged in)
    results/metrics/E2_per_class_metrics.csv
    results/predictions/E2_classical_predictions.csv
    figures/classification/E2_confusion_matrix.png
    figures/classification/E2_confusion_matrix_norm.png
    figures/classification/E2_class_f1_bar.png
"""
import io
from datetime import datetime
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("e2-classical-rf")

CROPS_VOL   = modal.Volume.from_name("e3-crops-v1")
RESULTS_VOL = modal.Volume.from_name("e2-rf-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        "opencv-python-headless>=4.8.0",  # headless: no libGL dependency
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "Pillow>=10.0.0",
        "joblib>=1.3.0",
        "scipy>=1.11.0",
    )
)

CLASSES      = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
FEATURE_DIM  = 252
SEED         = 42

# ── Pre-computed Gabor kernels (module-level — computed once, shared across workers) ──
# Defined here so they're available in remote container without recomputation per crop.
# Populated lazily inside the remote function to avoid cv2 import at local entrypoint time.
_GABOR_KERNELS = None  # list of cv2 kernels, shape (24,)


def _build_gabor_kernels():
    import cv2
    import numpy as np
    kernels = []
    for freq in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        for theta in [0, 1 * 3.14159265 / 4, 2 * 3.14159265 / 4, 3 * 3.14159265 / 4]:
            k = cv2.getGaborKernel((21, 21), 5, theta, 1 / freq, 0.5, 0, ktype=cv2.CV_32F)
            kernels.append(k)
    return kernels  # 24 kernels


def _extract_features(img_rgb, gabor_kernels):
    """252-dim feature vector from a 224×224 RGB uint8 ndarray."""
    import cv2
    import numpy as np
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Color histogram — 96 dims (HSV, 32 bins/channel)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    color_feats = []
    for ch in range(3):
        hist, _ = np.histogram(img_hsv[:, :, ch], bins=32, range=(0, 256), density=True)
        color_feats.append(hist)
    color_hist = np.concatenate(color_feats)

    # LBP — 26 dims
    lbp = local_binary_pattern(gray, 24, 3, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)

    # Gabor — 48 dims (pre-computed kernels, mean+std per filter)
    gray_f = gray.astype(np.float32)
    gabor_feats = []
    for k in gabor_kernels:
        filtered = cv2.filter2D(gray_f, cv2.CV_32F, k)
        gabor_feats.extend([filtered.mean(), filtered.std()])
    gabor = np.array(gabor_feats)

    # HOG — 36 dims
    hog_feats = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), feature_vector=True)
    target = 36
    if len(hog_feats) >= target:
        hog_out = hog_feats[:target]
    else:
        hog_out = np.pad(hog_feats, (0, target - len(hog_feats)))

    # Shape — 10 dims
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / (h + 1e-6)
        circ   = 4 * 3.14159265 * area / (perim ** 2 + 1e-6)
        extent = area / (w * h + 1e-6)
        hu     = cv2.HuMoments(cv2.moments(cnt)).flatten()
        shape  = np.array([aspect, circ, extent, *hu[:7]])
    else:
        shape = np.zeros(10)

    # GLCM — 36 dims
    glcm = graycomatrix(gray, distances=[1, 3],
                        angles=[0, 0.7853981, 1.5707963, 2.3561944],
                        levels=256, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    glcm_raw = np.concatenate([graycoprops(glcm, p).ravel() for p in props])
    glcm_out = glcm_raw[:36]

    feats = np.concatenate([color_hist, lbp_hist, gabor, hog_out, shape, glcm_out])
    assert len(feats) == FEATURE_DIM, f"Expected {FEATURE_DIM}, got {len(feats)}"
    return feats


def _process_crop(args):
    """Worker function — must be module-level for joblib loky pickling."""
    path, label, augment, gabor_kernels = args
    from PIL import Image as PILImage
    import numpy as np

    try:
        img = np.array(PILImage.open(path).convert("RGB").resize((224, 224), PILImage.BILINEAR))
        if augment:
            variants = [img, np.fliplr(img),
                        np.rot90(img, 1), np.rot90(img, 2), np.rot90(img, 3)]
        else:
            variants = [img]
        return [(_extract_features(v.copy(), gabor_kernels), label) for v in variants]
    except Exception:
        return []


@app.function(
    image=image,
    cpu=16,
    memory=65536,   # 64 GB: feature extraction + RF training need headroom
    timeout=43200,  # 12 hrs: classical feature extraction is CPU-heavy
    volumes={
        "/crops_vol": CROPS_VOL,
        "/results"  : RESULTS_VOL,
    },
)
def extract_and_train():
    import time, zipfile, random, json
    import numpy as np
    import pandas as pd
    import joblib as jl
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        precision_score, recall_score, classification_report,
        confusion_matrix, matthews_corrcoef, cohen_kappa_score,
        roc_auc_score, top_k_accuracy_score,
    )
    from joblib import Parallel, delayed

    random.seed(SEED)
    np.random.seed(SEED)

    CROPS_DIR   = Path("/tmp/dataset_crops")
    RESULTS_DIR = Path("/results")
    FEATURES_DIR = RESULTS_DIR / "features"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(exist_ok=True)

    # ── Extract crops zip ──────────────────────────────────────────────────────
    if not (CROPS_DIR / "train").exists():
        zip_path = Path("/crops_vol/dataset_crops.zip")
        assert zip_path.exists(), "dataset_crops.zip not found in e3-crops-v1 Volume"
        print("Extracting crops ...")
        CROPS_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(CROPS_DIR))
        print("  Done.")

    # Pre-compute Gabor kernels once for all workers
    gabor_kernels = _build_gabor_kernels()
    print(f"Pre-computed {len(gabor_kernels)} Gabor kernels.")

    def extract_split(split: str):
        paths, labels = [], []
        for cls in CLASSES:
            cls_dir = CROPS_DIR / split / cls
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.glob("*.jpg")):
                paths.append(str(p))
                labels.append(CLASS_TO_IDX[cls])

        augment    = (split == "train")
        n_variants = 5 if augment else 1
        print(f"[{split}] {len(paths):,} crops × {n_variants} = {len(paths)*n_variants:,} samples  (parallel, 16 cores)")

        t0 = time.perf_counter()
        args = [(p, l, augment, gabor_kernels) for p, l in zip(paths, labels)]

        results = Parallel(n_jobs=16, backend="loky", batch_size=200, verbose=5)(
            delayed(_process_crop)(a) for a in args
        )

        X_list = [feat for res in results for feat, _ in res]
        y_list = [lab  for res in results for _, lab  in res]

        elapsed = time.perf_counter() - t0
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        errors = len(paths) * n_variants - len(X_list)
        throughput = len(paths) / elapsed
        print(f"[{split}] done: {X.shape}  {elapsed/60:.1f} min  "
              f"{throughput:.0f} crops/s  {errors} errors")
        np.save(str(FEATURES_DIR / f"classical_{split}_X.npy"), X)
        np.save(str(FEATURES_DIR / f"classical_{split}_y.npy"), y)
        print(f"[{split}] saved feature arrays to features/classical_{split}_*.npy")
        return X, y, elapsed / max(len(paths), 1) * 1000

    # ── 1. Feature extraction ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[1/3] FEATURE EXTRACTION")
    print("=" * 60)
    X_train, y_train, train_ms = extract_split("train")
    X_val,   y_val,   _        = extract_split("val")
    X_test,  y_test,  _        = extract_split("test")
    extraction_throughput = 1000.0 / train_ms

    manifest = {
        "experiment": "E2",
        "feature_type": "classical_handcrafted",
        "feature_dim": FEATURE_DIM,
        "classes": [{"id": i, "name": c} for i, c in enumerate(CLASSES)],
        "label_encoding": "y arrays contain integer class IDs matching classes[].id",
        "splits": {
            "train": {
                "X": "classical_train_X.npy",
                "y": "classical_train_y.npy",
                "shape": list(X_train.shape),
                "augmentation": "original+hflip+rot90+rot180+rot270",
            },
            "val": {
                "X": "classical_val_X.npy",
                "y": "classical_val_y.npy",
                "shape": list(X_val.shape),
                "augmentation": "none",
            },
            "test": {
                "X": "classical_test_X.npy",
                "y": "classical_test_y.npy",
                "shape": list(X_test.shape),
                "augmentation": "none",
            },
        },
        "component_dims": {
            "hsv_color_histogram": 96,
            "lbp_uniform": 26,
            "gabor_mean_std": 48,
            "hog_truncated": 36,
            "shape": 10,
            "glcm_truncated": 36,
        },
    }
    (FEATURES_DIR / "classical_feature_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # ── 2. Train Random Forest ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[2/3] TRAINING RANDOM FOREST")
    print("=" * 60)
    t0 = time.perf_counter()
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )
    clf.fit(X_train, y_train)
    train_s = time.perf_counter() - t0
    print(f"  Trained in {train_s:.1f} s")
    jl.dump(clf, str(RESULTS_DIR / "random_forest_E2.pkl"))
    print("  Saved: random_forest_E2.pkl")

    # ── 3a. Validation (overfitting check) ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("[3/3] EVALUATION")
    print("=" * 60)
    y_pred_val   = clf.predict(X_val)
    val_accuracy = float(accuracy_score(y_val, y_pred_val))
    val_macro_f1 = float(f1_score(y_val, y_pred_val, average="macro", zero_division=0))
    print(f"  Val accuracy : {val_accuracy:.4f}")
    print(f"  Val macro-F1 : {val_macro_f1:.4f}")

    # ── 3b. Test (official metrics) ────────────────────────────────────────────
    t_infer = time.perf_counter()
    y_pred  = clf.predict(X_test)
    infer_s = time.perf_counter() - t_infer
    y_prob  = clf.predict_proba(X_test)

    print(classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))

    N            = len(y_test)
    n_cls        = len(CLASSES)
    labels_range = list(range(n_cls))
    per_f1   = f1_score(y_test, y_pred, average=None, labels=labels_range, zero_division=0)
    per_prec = precision_score(y_test, y_pred, average=None, labels=labels_range, zero_division=0)
    per_rec  = recall_score(y_test, y_pred, average=None, labels=labels_range, zero_division=0)

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
        "experiment"               : "E2",
        "model"                    : "RandomForest",
        "augmentation"             : "hflip+rot90+rot180+rot270",
        "timestamp"                : datetime.now().strftime("%Y-%m-%d %H:%M"),
        "accuracy"                 : round(float(accuracy_score(y_test, y_pred)), 4),
        "macro_f1"                 : round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1"              : round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "balanced_accuracy"        : round(float(balanced_accuracy_score(y_test, y_pred)), 4),
        "mcc"                      : round(float(matthews_corrcoef(y_test, y_pred)), 4),
        "cohen_kappa"              : round(float(cohen_kappa_score(y_test, y_pred)), 4),
        "auc_roc_macro"            : round(auc, 4) if auc == auc else float("nan"),
        "top2_accuracy"            : round(top2, 4) if top2 == top2 else float("nan"),
        "val_accuracy"             : round(val_accuracy, 4),
        "val_macro_f1"             : round(val_macro_f1, 4),
        "feature_dim"              : FEATURE_DIM,
        "n_train_samples"          : len(y_train),
        "inference_ms_per_crop"    : round((infer_s / N) * 1000, 4),
        "extraction_ms_per_crop"   : round(train_ms, 4),
        "extraction_throughput_cps": round(extraction_throughput, 2),
        **{f"f1_{CLASSES[i]}"  : round(float(v), 4) for i, v in enumerate(per_f1)},
        **{f"prec_{CLASSES[i]}": round(float(v), 4) for i, v in enumerate(per_prec)},
        **{f"rec_{CLASSES[i]}" : round(float(v), 4) for i, v in enumerate(per_rec)},
        **{f"auc_{CLASSES[i]}" : round(float(v), 4) if v == v else float("nan")
           for i, v in enumerate(per_class_auc)},
    }

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "classification_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df["experiment"] != "E2"]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)

    pd.DataFrame([{
        "experiment": "E2", "class": c,
        "f1"       : row[f"f1_{c}"],
        "precision": row[f"prec_{c}"],
        "recall"   : row[f"rec_{c}"],
        "auc"      : row[f"auc_{c}"],
    } for c in CLASSES]).to_csv(RESULTS_DIR / "E2_per_class_metrics.csv", index=False)

    pd.DataFrame([{
        "index"     : i,
        "true_class": CLASSES[y_test[i]],
        "predicted" : CLASSES[y_pred[i]],
        "correct"   : bool(y_test[i] == y_pred[i]),
        "confidence": round(float(y_prob[i].max()), 4),
    } for i in range(N)]).to_csv(RESULTS_DIR / "E2_classical_predictions.csv", index=False)

    # ── Figures ────────────────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred, labels=labels_range)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("E2 — Classical Features (252-dim) + Random Forest")
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "figures" / "E2_confusion_matrix.png"), dpi=150)
    plt.close()

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("E2 — Normalised Confusion Matrix (Random Forest)")
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "figures" / "E2_confusion_matrix_norm.png"), dpi=150)
    plt.close()

    per_class_f1 = [row[f"f1_{c}"] for c in CLASSES]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(CLASSES, per_class_f1, color="steelblue")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 Score")
    ax.set_title("E2 Classical+RF — Per-class F1")
    for bar, v in zip(bars, per_class_f1):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "figures" / "E2_class_f1_bar.png"), dpi=150)
    plt.close()

    RESULTS_VOL.commit()
    print("\nAll outputs saved to Modal Volume.")
    return row


@app.local_entrypoint()
def main():
    import pandas as pd

    print("=" * 60)
    print("E2  Classical Features + Random Forest  [Modal 16-core CPU]")
    print("=" * 60)

    print("Checking e3-crops-v1 Volume for dataset_crops.zip ...")
    entries = {e.path.lstrip("/") for e in CROPS_VOL.listdir("/")}
    if "dataset_crops.zip" not in entries:
        import zipfile
        crops_dir = REPO_ROOT / "data" / "processed" / "dataset_crops"
        zip_path  = REPO_ROOT / "data" / "processed" / "dataset_crops.zip"
        assert crops_dir.exists(), f"Crops not found: {crops_dir}"
        if not zip_path.exists():
            print("Creating crops zip ...")
            all_jpgs = sorted(crops_dir.rglob("*.jpg"))
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
                for i, f in enumerate(all_jpgs):
                    zf.write(f, f.relative_to(crops_dir))
                    if (i + 1) % 10000 == 0:
                        print(f"  {i+1:,} files zipped")
            print(f"  {zip_path.stat().st_size / 1e6:.0f} MB")
        print("Uploading to Modal Volume ...")
        with CROPS_VOL.batch_upload() as batch:
            batch.put_file(str(zip_path), "dataset_crops.zip")
        print("  Uploaded.")
    else:
        print("  dataset_crops.zip already in Volume — skipping upload.")

    print("\nLaunching extract_and_train on Modal ...")
    row = extract_and_train.remote()

    print("\n" + "=" * 60)
    print("E2  Classical+RF  |  RESULTS")
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
    print(f"  Extraction        : {row['extraction_ms_per_crop']:.2f} ms/crop  "
          f"({row['extraction_throughput_cps']:.0f} crops/s)")
    print(f"\n  Overfitting check (val vs test):")
    print(f"    Val  macro-F1  : {row['val_macro_f1']:.4f}")
    print(f"    Test macro-F1  : {row['macro_f1']:.4f}")
    gap = abs(row['val_macro_f1'] - row['macro_f1'])
    print(f"    Gap            : {gap:.4f}  {'(OK)' if gap < 0.05 else '(WARNING)'}")
    print("\n  Per-class F1 / AUC:")
    for c in CLASSES:
        v   = row[f"f1_{c}"]
        au  = row[f"auc_{c}"]
        bar = "#" * int(v * 40)
        auc_str = f"{au:.4f}" if au == au else "  N/A"
        print(f"    {c:<16} F1={v:.4f}  AUC={auc_str}  {bar}")

    # ── Download all outputs ───────────────────────────────────────────────────
    vol = modal.Volume.from_name("e2-rf-results")
    downloads = {
        "features/classical_train_X.npy"       : REPO_ROOT / "data" / "processed" / "features" / "classical_train_X.npy",
        "features/classical_train_y.npy"       : REPO_ROOT / "data" / "processed" / "features" / "classical_train_y.npy",
        "features/classical_val_X.npy"         : REPO_ROOT / "data" / "processed" / "features" / "classical_val_X.npy",
        "features/classical_val_y.npy"         : REPO_ROOT / "data" / "processed" / "features" / "classical_val_y.npy",
        "features/classical_test_X.npy"        : REPO_ROOT / "data" / "processed" / "features" / "classical_test_X.npy",
        "features/classical_test_y.npy"        : REPO_ROOT / "data" / "processed" / "features" / "classical_test_y.npy",
        "features/classical_feature_manifest.json": REPO_ROOT / "data" / "processed" / "features" / "classical_feature_manifest.json",
        "random_forest_E2.pkl"               : REPO_ROOT / "models" / "classifiers" / "random_forest_E2.pkl",
        "classification_results.csv"         : REPO_ROOT / "results" / "metrics" / "classification_results.csv",
        "E2_per_class_metrics.csv"           : REPO_ROOT / "results" / "metrics" / "E2_per_class_metrics.csv",
        "E2_classical_predictions.csv"       : REPO_ROOT / "results" / "predictions" / "E2_classical_predictions.csv",
        "figures/E2_confusion_matrix.png"    : REPO_ROOT / "figures" / "classification" / "E2_confusion_matrix.png",
        "figures/E2_confusion_matrix_norm.png": REPO_ROOT / "figures" / "classification" / "E2_confusion_matrix_norm.png",
        "figures/E2_class_f1_bar.png"        : REPO_ROOT / "figures" / "classification" / "E2_class_f1_bar.png",
    }

    print("\nDownloading outputs ...")
    for remote, local in downloads.items():
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {remote} ...", end="", flush=True)
        if remote == "classification_results.csv" and local.exists():
            buf = io.BytesIO()
            for chunk in vol.read_file(remote):
                buf.write(chunk)
            buf.seek(0)
            remote_df = pd.read_csv(buf)
            local_df  = pd.read_csv(local)
            local_df  = local_df[local_df["experiment"] != "E2"]
            merged    = pd.concat([local_df, remote_df[remote_df["experiment"] == "E2"]],
                                  ignore_index=True)
            merged.to_csv(local, index=False)
            print("  merged")
            continue
        with open(local, "wb") as f:
            for chunk in vol.read_file(remote):
                f.write(chunk)
        print(f"  {local.stat().st_size / 1024:.0f} KB")

    print("\nE2 complete. All outputs in local repo.")
