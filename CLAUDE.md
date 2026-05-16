# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Compression-Aware Multiscale Feature Fusion for Waste Object Detection and Classification**

A research project comparing feature extraction strategies (classical, deep, YOLO) and fusion methods (early, late, attention, hybrid) for 6-class waste classification. The primary research question: which feature representation achieves best accuracy at lowest computational cost?

- **Dataset**: Garbage Classification (Roboflow v2) — 10,464 images, 6 classes, YOLO bbox format
- **Classes**: `BIODEGRADABLE`(0), `CARDBOARD`(1), `GLASS`(2), `METAL`(3), `PAPER`(4), `PLASTIC`(5)
- **Environment**: Google Colab GPU runtime; local files here are the source notebooks and dataset

## Project Structure

```
notebooks/          — one notebook per experiment (00–09)
src/
  data/             — dataset prep, splitting, crop extraction
  features/         — classical (252-dim), deep EfficientNetB0 (1280-dim implemented), YOLO FPN
  fusion/           — early, late, attention fusion; PCA/autoencoder/feature-selection
  models/           — train_*.py wrappers for each experiment
  evaluation/       — metrics, plots, efficiency benchmarks
  utils/            — seed.py, paths.py, config_loader.py, logger.py
configs/            — per-topic YAML configs (yolo, features, training, experiments)
config.yaml         — all locked constants (single source of truth)
```

`Trash_Dataset/` is gitignored — it stays in place locally but is not pushed to GitHub.
`01_Preprocessing (2).ipynb` at the root is the source file copied to `notebooks/01_preprocessing_and_splits.ipynb`.

## Running Notebooks

All notebooks are designed to run in **Google Colab**, not locally. The standard workflow:

1. Clone repo to `MyDrive/CV/repo/` and upload `Trash_Dataset.zip` to `MyDrive/CV/`
2. Open any `notebooks/*.ipynb` in Colab — each has a Drive mount + `sys.path` setup cell
3. Run cells sequentially — each cell builds on the previous cell's outputs

Path constants live in `src/utils/paths.py`. All notebooks import from there — don't hardcode Drive paths in notebooks.

## Experiment Architecture

Two parallel tracks:

**Detection track** (uses full images, YOLO bounding boxes):
- `E1`: YOLOv8-nano baseline — mAP@0.5, mAP@0.5:0.95, FPS, model size
- `E11`: YOLO Hybrid — YOLOv8 FPN features + classical 252-dim features

**Classification track** (uses 224×224 crops extracted from ground-truth boxes):
- `E2`: Classical features only (252-dim) → Random Forest
- `E3`: EfficientNetB0 deep features only (1,280-dim implemented late pooled features) → SVM
- `E4–E7`: Early fusion variants (raw concat, PCA, Autoencoder, feature selection)
- `E8–E9`: Late fusion (average voting, weighted voting)
- `E10`: Attention/gating fusion (learns per-sample feature trust weights)

Primary metric for classification: **Macro-F1** (compensates for 10.3:1 class imbalance). Always use `class_weight='balanced'` in sklearn classifiers.

## Locked Controlled Variables

| Variable | Value |
|----------|-------|
| Random seed | `42` (all cells, all splits) |
| Dataset split | 70 / 15 / 15 stratified by dominant class |
| Crop size | 224×224 (BILINEAR resize) |
| YOLO conf threshold | 0.25 |
| YOLO NMS IoU threshold | 0.45 |
| Classical feature dim | 252 |
| Deep feature dim | 1,280 implemented (`timm` EfficientNetB0 global pooled output) |

Do not change these without updating all downstream notebooks.

## Preprocessing Pipeline (`01_Preprocessing.ipynb`)

Cells run in order — each produces artifacts consumed by the next:

- **Cell 1**: Mount Drive, extract ZIP, structural audit → discovers dataset root
- **Cell 2**: Parse `data.yaml`, verify 6-class taxonomy → GO/NO-GO verdict
- **Cell 3**: Clean zero-area boxes (17 removed), check image integrity, MD5 dedup, class distribution charts
- **Cell 4**: Discard vendor split (corrupted — classes absent from val/test); create stratified 70/15/15 split by dominant class at image level
- **Cell 5**: Extract 224×224 crops from ground-truth boxes (65,830 valid; 8,243 rejected as <16px)
- **Cell 6**: Classical feature extraction — fixed at 252 dims and completed through Modal E2 run
- **Cell 7**: EfficientNetB0 deep feature extraction — completed through Modal E3 feature run

## Drive Output Structure

```
/content/drive/MyDrive/CV/
├── Trash_Dataset.zip
├── 01_dataset_analysis/       # Cleaning CSVs + distribution plots (Figs 1–2)
├── 02_dataset_split/          # data.yaml + split stats + split distribution plots (Figs 3–4)
├── 03_yolo_baseline/          # E1 training outputs (weights, metrics)
└── 04_crops/                  # Crop metadata CSV + distribution plots (Figs 9–11)
```

## Known Issues

| Issue | Status |
|-------|--------|
| Cell 6 classical feature dim mismatch (283 vs 252) | Fixed by current `src/features/classical_features.py` and Modal E2 script; exported arrays are 252-dim |
| BIODEGRADABLE redundant annotations (4 boxes for 1 object) | Known, acceptable — class weights compensate |
| Vendor Roboflow split was corrupted (classes missing from val/test) | Fixed in Cell 4 |
| Deep feature dim mismatch in older docs (2048 vs 1280) | Current implementation/results use 1280-dim EfficientNetB0 pooled features |

## Team Notebook Assignments

- **Aly** (this repo): `01_Preprocessing.ipynb` — preprocessing + E1 YOLO baseline
- **Ahmed**: `04_E4_E7_Early_Fusion.ipynb` — depends on Cells 6 + 7
- **Bayo**: `05_E8_E9_Late_Fusion.ipynb`, `06_E10_Attention_Fusion.ipynb` — E2/E3 feature arrays are now available in `data/processed/features/`
- **Nada**: `07_E11_YOLO_Hybrid.ipynb` — depends on E1 + Cell 6

E1 (YOLO training) can start immediately — it only needs `dataset_split_70_15_15/data.yaml` and does not depend on Cells 6–7. As of the Codex update below, the current E1 Colab run had not yet been synced into the live repo.

## Codex Status Update - 2026-05-16

This section was added by **Codex** as a handoff for Claude/Claude Code. It records work completed after the original memory notes above.

### Current Experiment Status

| Experiment | Status | Key outputs |
|------------|--------|-------------|
| E1 YOLO baseline | **COMPLETE** on Modal H100. v1 archived. | `models/yolo/yolov8n_E1_best.pt`, `results/metrics/detection_results.csv`, `results/metrics/E1_per_class_metrics.csv`, `figures/yolo/`, YOLO features in `data/processed/features/` |
| E2 Classical + RF | **COMPLETE** on Modal, outputs downloaded locally. | `models/classifiers/random_forest_E2.pkl`, `results/metrics/E2_per_class_metrics.csv`, `results/predictions/E2_classical_predictions.csv`, E2 row in `results/metrics/classification_results.csv` |
| E3 EfficientNetB0 + SVM | **COMPLETE** (re-run with fixed script on Modal). | `models/classifiers/svm_E3.pkl`, `results/metrics/E3_per_class_metrics.csv`, `results/predictions/E3_deep_predictions.csv`, E3 row in `results/metrics/classification_results.csv` |

### E1 Results

Run: Modal H100 80GB, `E1_baseline_v2`, 100 epochs, imgsz=640, lr0=0.001, freeze=10, close_mosaic=15.

Metrics in `results/metrics/detection_results.csv`:

- `mAP50`: 0.4559
- `mAP50_95`: 0.3164
- `precision`: 0.6637
- `recall`: 0.5063
- `fps`: 259.9 (H100 eval — not comparable to Colab v1 117 fps)
- `model_size_mb`: 6.262
- `epochs_trained`: 100
- `image_size`: 640
- Best val mAP50 during training: 0.5451 at epoch 94

Per-class mAP50: BIODEGRADABLE=0.4663, CARDBOARD=0.4445, GLASS=0.4575, METAL=0.5161, PAPER=0.3939, PLASTIC=0.4572

v1 archived result (Colab, imgsz=416, lr0=0.01, no freeze): mAP50=0.4735, mAP50_95=0.3153. Use **v2** (this run) for the paper — it is fully reproducible with documented hyperparameters; mAP@0.5:0.95 is marginally better; freeze=10 is a deliberate methodological choice.

### E2 Results

E2 was run on Modal after killing a stale local `python scripts/run_e2_classical.py` process. Successful Modal app: `ap-NKl5Jw6qEuqKddhNZXcKuh`.

Metrics in `results/metrics/classification_results.csv`:

- `accuracy`: 0.7740
- `macro_f1`: 0.6476
- `weighted_f1`: 0.7551
- `balanced_accuracy`: 0.6024
- `auc_roc_macro`: 0.9085
- `top2_accuracy`: 0.8802
- `val_macro_f1`: 0.6426 — val/test gap: 0.0050 (no overfitting)
- Per-class AUC: BIODEGRADABLE=0.9317, CARDBOARD=0.9710, GLASS=0.8290, METAL=0.9246, PAPER=0.8721, PLASTIC=0.9225

E2 feature extraction details:

- Classical feature dim is correctly 252.
- `scripts/run_e2_classical_modal.py`: timeout=12h, memory=64GB, n_jobs=16, saves `features/classical_*.npy`
- Feature arrays downloaded into `data/processed/features/`

### E3 Results

E3 was **re-run on 2026-05-16** with a corrected script fixing two expert-audit issues:
1. **Val set was never scaled** — `X_val = scaler.transform(X_val)` was missing (data leakage in val eval). Fixed.
2. **Per-class AUC missing** — only macro AUC was computed. Per-class AUC added for all 6 classes.

E3 uses `timm` EfficientNetB0 with `num_classes=0` → 1280-dim global pooled vector (not 2048). Keep downstream code at 1280.

Metrics in `results/metrics/classification_results.csv`:

- `accuracy`: 0.8522
- `macro_f1`: 0.7848
- `weighted_f1`: 0.8471
- `balanced_accuracy`: 0.7672
- `mcc`: 0.7673
- `cohen_kappa`: 0.7642
- `auc_roc_macro`: 0.9566
- `top2_accuracy`: 0.9404
- `val_accuracy`: 0.8690
- `val_macro_f1`: 0.7826 — val/test gap: 0.0022 (no overfitting)
- `inference_ms_per_crop`: 21.27
- Per-class F1: BIODEGRADABLE=0.9249, CARDBOARD=0.8706, GLASS=0.7294, METAL=0.7654, PAPER=0.7419, PLASTIC=0.6767
- Per-class AUC: BIODEGRADABLE=0.9636, CARDBOARD=0.9931, GLASS=0.9134, METAL=0.9684, PAPER=0.9538, PLASTIC=0.9471

### Feature Handoff for Teammates

All feature arrays are in `data/processed/features/`. All three modalities are now complete.

**For E4-E10 fusion (row-aligned, no augmentation):**

| File | Shape | Source |
|------|-------|--------|
| `classical_train_clean_X.npy` | (45177, 252) | E2 classical |
| `classical_val_X.npy` | (9935, 252) | E2 classical |
| `classical_test_X.npy` | (10553, 252) | E2 classical |
| `deep_train_X.npy` | (45177, 1280) | E3 EfficientNetB0 |
| `deep_val_X.npy` | (9935, 1280) | E3 EfficientNetB0 |
| `deep_test_X.npy` | (10553, 1280) | E3 EfficientNetB0 |
| `yolo_train_X.npy` | (45177, 256) | E1 YOLO SPPF backbone |
| `yolo_val_X.npy` | (9935, 256) | E1 YOLO SPPF backbone |
| `yolo_test_X.npy` | (10553, 256) | E1 YOLO SPPF backbone |

Corresponding `_y.npy` files (integer class labels 0-5) exist for each split.

Important: `classical_train_X.npy` shape `(225885, 252)` is the 5x-augmented E2 training matrix. Not row-aligned with deep/YOLO features. Use `classical_train_clean_X.npy` for fusion.

Manifests: `classical_feature_manifest.json`, `deep_feature_manifest.json`, `yolo_feature_manifest.json`

### Next Work

E1/E2/E3 baselines are all complete. All feature arrays are ready.

- **Ahmed** (E4-E7 early fusion): use `classical_train_clean_X.npy` + `deep_train_X.npy`
- **Bayo** (E8-E9 late fusion, E10 attention): same arrays
- **Nada** (E11 YOLO hybrid): use `yolo_train_X.npy` + `classical_train_clean_X.npy`; YOLO weights at `models/yolo/yolov8n_E1_best.pt`
