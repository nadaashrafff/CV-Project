# Compression-Aware Multiscale Feature Fusion for Waste Object Detection

A comparative study evaluating 11 feature extraction and fusion strategies for 6-class waste classification.
**Research question:** which feature representation achieves the best accuracy-to-cost tradeoff?

---

## Team

| Member | Role | Notebooks |
|--------|------|-----------|
| **Aly** | Data prep, preprocessing, E1–E3 baselines | 00, 01, 02, 03, 04 |
| **Ahmed** | Early fusion (E4–E7) | 05 |
| **Bayo** | Late fusion, attention fusion (E8–E10) | 06, 07 |
| **Nada** | YOLO hybrid (E11) | 08 |

---

## Dataset

**Garbage Classification (Roboflow v2)** — 10,464 images, 6 classes:
`BIODEGRADABLE` · `CARDBOARD` · `GLASS` · `METAL` · `PAPER` · `PLASTIC`

Dataset is stored in **Google Drive** and **Modal Volumes** — not committed to this repo (too large).
The stratified 70/15/15 split has 7,325 / 1,570 / 1,569 images per split.
65,665 valid 224×224 crops were extracted from ground-truth bounding boxes.

---

## Baseline Experiment Results (E1, E2, E3 — COMPLETE)

### E1 — YOLOv8-nano Detection Baseline

Trained on Modal H100 80GB · 100 epochs · imgsz=640 · lr0=0.001 · freeze=10 · close_mosaic=15

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.4559 |
| mAP@0.5:0.95 | 0.3164 |
| Precision | 0.6637 |
| Recall | 0.5063 |
| Model size | 6.26 MB |

Per-class mAP@0.5: BIODEGRADABLE=0.47 · CARDBOARD=0.44 · GLASS=0.46 · METAL=0.52 · PAPER=0.39 · PLASTIC=0.46

Outputs: `results/metrics/detection_results.csv` · `results/metrics/E1_per_class_metrics.csv` · `models/yolo/yolov8n_E1_best.pt` · `figures/yolo/`

### E2 — Classical Features + Random Forest

252-dim features (color hist + LBP + Gabor + HOG + shape + GLCM) · RF(n=200, balanced)
5× augmentation on training split only.

| Metric | Value |
|--------|-------|
| Accuracy | 0.7740 |
| Macro-F1 | 0.6476 |
| Weighted-F1 | 0.7551 |
| AUC-ROC (macro) | 0.9085 |
| Val Macro-F1 | 0.6426 (gap 0.005 — no overfitting) |

Per-class AUC: BIO=0.932 · CARD=0.971 · GLASS=0.829 · METAL=0.925 · PAPER=0.872 · PLASTIC=0.923

Outputs: `results/metrics/classification_results.csv` (E2 row) · `results/metrics/E2_per_class_metrics.csv` · `models/classifiers/random_forest_E2.pkl`

### E3 — EfficientNetB0 Deep Features + SVM (RBF)

1280-dim global-pooled features (`timm` EfficientNetB0, `num_classes=0`) · RBF SVM

| Metric | Value |
|--------|-------|
| Accuracy | 0.8522 |
| Macro-F1 | 0.7848 |
| Weighted-F1 | 0.8471 |
| AUC-ROC (macro) | 0.9566 |
| Val Macro-F1 | 0.7826 (gap 0.002 — no overfitting) |
| Inference | 21.3 ms/crop |

Per-class F1: BIO=0.925 · CARD=0.871 · GLASS=0.729 · METAL=0.765 · PAPER=0.742 · PLASTIC=0.677

Outputs: `results/metrics/classification_results.csv` (E3 row) · `results/metrics/E3_per_class_metrics.csv` · `models/classifiers/svm_E3.pkl`

---

## Feature Arrays — Ready for E4–E11

All arrays are in `data/processed/features/` and are **row-aligned** (same crop order across all modalities).

| File | Shape | Description |
|------|-------|-------------|
| `classical_train_clean_X.npy` | (45177, 252) | E2 classical — train, no augmentation |
| `classical_val_X.npy` | (9935, 252) | E2 classical — val |
| `classical_test_X.npy` | (10553, 252) | E2 classical — test |
| `deep_train_X.npy` | (45177, 1280) | E3 EfficientNetB0 — train |
| `deep_val_X.npy` | (9935, 1280) | E3 EfficientNetB0 — val |
| `deep_test_X.npy` | (10553, 1280) | E3 EfficientNetB0 — test |
| `yolo_train_X.npy` | (45177, 256) | E1 YOLO SPPF backbone — train |
| `yolo_val_X.npy` | (9935, 256) | E1 YOLO SPPF backbone — val |
| `yolo_test_X.npy` | (10553, 256) | E1 YOLO SPPF backbone — test |

Each split has a matching `_y.npy` with integer class labels (0–5 matching CLASSES order below).

> **Note:** `classical_train_X.npy` (shape 225885×252) is the 5× augmented E2 training matrix.
> It is NOT row-aligned with deep/YOLO features. Use `classical_train_clean_X.npy` for fusion.

Load example:
```python
import numpy as np
X_train = np.load("data/processed/features/classical_train_clean_X.npy")
y_train = np.load("data/processed/features/classical_train_clean_y.npy")
# Concatenate for early fusion:
X_fused = np.hstack([classical_X, deep_X])  # (45177, 1532)
```

---

## Experiment Overview

| ID | Name | Input | Classifier | Track | Status |
|----|------|-------|------------|-------|--------|
| E1 | YOLO Baseline | Full images (640px) | YOLOv8n | Detection | **COMPLETE** |
| E2 | Classical Only | 252-dim | Random Forest | Classification | **COMPLETE** |
| E3 | Deep Only | 1280-dim | RBF SVM | Classification | **COMPLETE** |
| E4 | Raw Concat Fusion | 252+1280=1532-dim | Random Forest | Classification | Pending |
| E5 | PCA Fusion | 1532→PCA(95%) | RF/SVM | Classification | Pending |
| E6 | Autoencoder Fusion | 1532→AE→256 | RF | Classification | Pending |
| E7 | Feature Selection | Top-200 | RF | Classification | Pending |
| E8 | Average Voting | E2+E3 proba avg | — | Classification | Pending |
| E9 | Weighted Voting | E2+E3 weighted | — | Classification | Pending |
| E10 | Attention Fusion | 252+1280 gating | MLP | Classification | Pending |
| E11 | YOLO Hybrid | 256+252=508-dim | MLP head | Detection+Class | Pending |

---

## Dependency Graph

```
notebooks/01_preprocessing_and_splits.ipynb  (Aly — DONE)
    │
    ├── notebooks/02_yolo_baseline_E1.ipynb  (Aly — DONE) ──────────────────────── E11 (Nada)
    │       └── yolo_train_X / val / test .npy  (256-dim)
    │
    ├── notebooks/03_classical_features_E2.ipynb  (Aly — DONE)
    │       └── classical_train_clean_X / val / test .npy  (252-dim)
    │
    └── notebooks/04_deep_features_E3.ipynb  (Aly — DONE)
            └── deep_train_X / val / test .npy  (1280-dim)
                    │
                    ├── notebooks/05_early_fusion_E4_E5_E6_E7.ipynb  (Ahmed)
                    ├── notebooks/06_late_fusion_E8_E9.ipynb         (Bayo)
                    ├── notebooks/07_attention_fusion_E10.ipynb      (Bayo)
                    └── notebooks/08_yolo_hybrid_E11.ipynb           (Nada)
```

**E4–E11 can start now** — all feature arrays are in `data/processed/features/`.

---

## How to Run

### Colab (notebooks)

All notebooks are designed for **Google Colab**. Local Python is not required.

```
1. Clone repo to: MyDrive/CV/repo/
2. Upload Trash_Dataset.zip to: MyDrive/CV/
3. Open the notebook in Colab → Runtime → Run all
```

Each notebook has a Drive-mount cell at the top. Run cells sequentially — each one builds on the previous.

### Modal (E1–E3 training scripts)

The baseline experiments were trained on Modal cloud GPUs. Scripts are in `scripts/`.

```bash
# Prerequisites
pip install modal
modal token new   # authenticate once

# E1 — YOLO detection training (H100, ~50 min)
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run scripts/run_e1_yolo_modal.py

# E1 — YOLO backbone feature extraction only (A10G, ~10 min, run after E1 training)
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run scripts/run_e1_extract_features_modal.py

# E2 — Classical features + Random Forest (CPU 64GB, ~2 hrs)
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run scripts/run_e2_classical_modal.py

# E3 — EfficientNetB0 feature extraction (A10G, ~30 min)
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run scripts/run_e3_features_modal.py

# E3 — RBF SVM training (CPU 32GB, ~2 hrs)
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run scripts/run_e3_svm_modal.py
```

> **Windows note:** The `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` prefix must be on the same line in bash.
> In PowerShell use: `$env:PYTHONUTF8="1"; $env:PYTHONIOENCODING="utf-8"; modal run ...`

---

## Locked Constants

Do not change these without updating all downstream notebooks.

| Variable | Value |
|----------|-------|
| Random seed | `42` |
| Dataset split | 70 / 15 / 15 stratified by dominant class |
| Crop size | 224×224 (BILINEAR) |
| Classical feature dim | **252** |
| Deep feature dim | **1280** (EfficientNetB0 global pool, `timm`, `num_classes=0`) |
| YOLO backbone feature dim | **256** (SPPF layer 9, global average pool) |
| YOLO conf threshold | 0.25 |
| YOLO NMS IoU threshold | 0.45 |
| E1 hyperparams | imgsz=640, lr0=0.001, freeze=10, close_mosaic=15, copy_paste=0.0 |
| Primary metric | **Macro-F1** (handles 10.3:1 class imbalance) |
| Class weights | `balanced` in all sklearn classifiers |

---

## Repo Structure

```
scripts/                    Modal training scripts (E1–E3)
  run_e1_yolo_modal.py      E1 detection training on H100
  run_e1_extract_features_modal.py  E1 YOLO backbone feature extraction
  run_e2_classical_modal.py E2 classical features + RF training
  run_e3_features_modal.py  E3 EfficientNetB0 feature extraction
  run_e3_svm_modal.py       E3 RBF SVM training
  archive/                  Superseded local scripts (reference only)

notebooks/                  One notebook per experiment (run in Colab)
  01_preprocessing_and_splits.ipynb
  02_yolo_baseline_E1.ipynb
  03_classical_features_E2.ipynb
  04_deep_features_E3.ipynb
  05_early_fusion_E4_E5_E6_E7.ipynb
  06_late_fusion_E8_E9.ipynb
  07_attention_fusion_E10.ipynb
  08_yolo_hybrid_E11.ipynb
  09_results_analysis.ipynb

src/
  data/                     Dataset prep, splitting, crop extraction
  features/                 classical_features.py, deep_features.py, yolo_fpn_features.py
  fusion/                   early, late, attention, dimensionality reduction
  models/                   Train wrappers
  evaluation/               Metrics, plots, efficiency benchmarks
  utils/                    seed.py, paths.py, config_loader.py, logger.py

data/processed/features/    Feature arrays (*.npy) + manifests (*.json)
  README.md                 Full feature array reference
  classical_feature_manifest.json
  deep_feature_manifest.json
  yolo_feature_manifest.json

models/
  yolo/yolov8n_E1_best.pt   E1 trained weights (gitignored — download from Modal)
  classifiers/random_forest_E2.pkl  (gitignored)
  classifiers/svm_E3.pkl            (gitignored)

results/metrics/            All experiment CSVs (committed)
  detection_results.csv     E1 detection metrics
  classification_results.csv  E2 + E3 classification metrics
  E1_per_class_metrics.csv
  E2_per_class_metrics.csv
  E3_per_class_metrics.csv

figures/
  preprocessing/            Figs 1–7 (dataset analysis, splits, crops)
  yolo/                     E1 training curves, confusion matrices, PR/F1 curves
  classification/           E2 + E3 confusion matrices, F1 bar charts

configs/                    YAML experiment configs
config.yaml                 All locked constants (single source of truth)
requirements.txt            Python dependencies
CLAUDE.md                   AI assistant context (detailed experiment history)
```

---

## Classes Reference

```python
CLASSES = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
# Integer labels: 0, 1, 2, 3, 4, 5
```

---

## Downloading Large Files (GitHub Release)

Large files (datasets, feature arrays, trained models) are attached to the **GitHub Release v1.0**.
They are not in the repo itself — download them manually after cloning.

### Quick download (PowerShell — Windows)

```powershell
# Install GitHub CLI first: winget install --id GitHub.cli
gh release download v1.0 --dir . --repo <REPO_URL>
```

### Files in the Release

| File | Size | Who needs it |
|------|------|--------------|
| `dataset_split_70_15_15.zip` | 190 MB | E1 retraining, E11 (Nada) |
| `dataset_crops.zip` | 736 MB | Re-extracting crops (optional) |
| `classical_train_clean_X.npy` | 43 MB | E4–E10 (Ahmed, Bayo) |
| `classical_val_X.npy` | 10 MB | E4–E10 |
| `classical_test_X.npy` | 10 MB | E4–E10 |
| `deep_train_X.npy` | 221 MB | E4–E10 (Ahmed, Bayo) |
| `deep_val_X.npy` | 49 MB | E4–E10 |
| `deep_test_X.npy` | 52 MB | E4–E10 |
| `yolo_train_X.npy` | 44 MB | E11 (Nada) |
| `yolo_val_X.npy` | 10 MB | E11 |
| `yolo_test_X.npy` | 10 MB | E11 |
| `random_forest_E2.pkl` | 451 MB | Inference / E8–E9 |
| `svm_E3.pkl` | 220 MB | Inference / E8–E9 |
| `yolov8n_E1_best.pt` | 6 MB | E11 (Nada) |

### After downloading, place files here

```
data/processed/
  dataset_split_70_15_15.zip
  dataset_crops.zip
  features/
    classical_train_clean_X.npy
    classical_val_X.npy
    classical_test_X.npy
    deep_train_X.npy
    deep_val_X.npy
    deep_test_X.npy
    yolo_train_X.npy
    yolo_val_X.npy
    yolo_test_X.npy

models/
  classifiers/
    random_forest_E2.pkl
    svm_E3.pkl
  yolo/
    yolov8n_E1_best.pt
```

---

## What's Gitignored (not in this repo)

- `Trash_Dataset/` and all raw/processed images — too large, stored in Google Drive
- `data/processed/features/*.npy` — regenerated by Modal scripts (~400 MB total)
- `models/yolo/*.pt`, `models/classifiers/*.pkl` — regenerated by Modal scripts
- `data/processed/*.zip` — dataset archives stored in Modal Volumes
