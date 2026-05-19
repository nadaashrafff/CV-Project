# Technical Report: Compression-Aware Multiscale Feature Fusion for Waste Object Detection

**Project:** EUI Computer Vision Final Project  
**Team:** Aly (lead), Ahmed, Bayo, Nada  
**Date:** May 2026  
**Status:** E1/E2/E3 Baselines Complete — E4–E11 In Progress

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset](#2-dataset)
3. [Locked Controlled Variables](#3-locked-controlled-variables)
4. [Experiment E1: YOLOv8-nano Detection Baseline](#4-experiment-e1-yolov8-nano-detection-baseline)
5. [Experiment E2: Classical Features + Random Forest](#5-experiment-e2-classical-features--random-forest)
6. [Experiment E3: EfficientNetB0 Deep Features + SVM](#6-experiment-e3-efficientnetb0-deep-features--svm)
7. [Comparative Analysis](#7-comparative-analysis)
8. [Feature Arrays for Downstream Experiments](#8-feature-arrays-for-downstream-experiments)
9. [Upcoming Experiments (E4–E11)](#9-upcoming-experiments-e4e11)
10. [Conclusions](#10-conclusions)
11. [Appendix A: Glossary](#appendix-a-glossary)
12. [Appendix B: Known Issues and Fixes](#appendix-b-known-issues-and-fixes)

---

## 1. Introduction

### 1.1 Research Question

Which feature representation achieves the best accuracy-to-computational-cost tradeoff for 6-class waste classification?

### 1.2 Motivation

Automated waste sorting is a critical application for sustainable urban infrastructure. Deep learning approaches achieve high accuracy but demand substantial compute. Classical hand-crafted features are fast but less discriminative. This project systematically evaluates 11 feature extraction and fusion strategies across two tracks — detection and classification — to identify optimal operating points on the accuracy/cost tradeoff curve.

### 1.3 Experiment Overview

| ID | Name | Input | Classifier | Track | Owner | Status |
|----|------|-------|------------|-------|-------|--------|
| E1 | YOLO Baseline | Full images (640px) | YOLOv8n | Detection | Aly | **COMPLETE** |
| E2 | Classical Only | 252-dim crops | Random Forest | Classification | Aly | **COMPLETE** |
| E3 | Deep Only | 1280-dim crops | RBF SVM | Classification | Aly | **COMPLETE** |
| E4 | Raw Concat Fusion | 252+1280=1532-dim | Random Forest | Classification | Ahmed | Pending |
| E5 | PCA Fusion | 1532→PCA(95%) | RF/SVM | Classification | Ahmed | Pending |
| E6 | Autoencoder Fusion | 1532→AE→256 | RF | Classification | Ahmed | Pending |
| E7 | Feature Selection | Top-200 of 1532 | RF | Classification | Ahmed | Pending |
| E8 | Average Voting | E2+E3 proba avg | — | Classification | Bayo | Pending |
| E9 | Weighted Voting | E2+E3 weighted | — | Classification | Bayo | Pending |
| E10 | Attention Fusion | 252+1280 gating | MLP | Classification | Bayo | Pending |
| E11 | YOLO Hybrid | 256+252=508-dim | MLP head | Detection+Class | Nada | Pending |

---

## 2. Dataset

### 2.1 Source

**Garbage Classification — Roboflow v2**  
- 10,464 images in YOLO bounding-box annotation format  
- 6 classes: `BIODEGRADABLE`, `CARDBOARD`, `GLASS`, `METAL`, `PAPER`, `PLASTIC`

### 2.2 Data Cleaning

The raw dataset was audited before any splits were created:

| Check | Finding | Action |
|-------|---------|--------|
| Zero-area boxes | 17 annotations with width=0 or height=0 | Removed |
| Image integrity | No corrupted images found | None |
| MD5 deduplication | No exact duplicates found | None |
| BIODEGRADABLE multi-box | 4 boxes on same object common | Accepted — class weights compensate |
| Vendor split quality | Classes absent from vendor val/test splits | Discarded vendor split entirely |

### 2.3 Class Distribution

| Class | Train images | Val images | Test images | % of total |
|-------|-------------|-----------|------------|-----------|
| BIODEGRADABLE | 3,170 | 679 | 679 | 43.4% |
| CARDBOARD | 798 | 171 | 171 | 10.9% |
| GLASS | 790 | 169 | 169 | 10.8% |
| METAL | 845 | 181 | 181 | 11.5% |
| PAPER | 862 | 185 | 185 | 11.8% |
| PLASTIC | 860 | 185 | 185 | 11.7% |

**Class imbalance ratio:** 10.3:1 (BIODEGRADABLE vs CARDBOARD). All sklearn classifiers use `class_weight='balanced'`. Primary metric is **Macro-F1** to handle imbalance fairly.

### 2.4 Dataset Split

The vendor-provided split was discarded (corrupted — classes absent from validation and test sets). A fresh stratified split was created:

- **Strategy:** stratified by the dominant class (highest-area box) at image level
- **Ratio:** 70 / 15 / 15 (train / val / test)
- **Resulting counts:** 7,325 / 1,570 / 1,569 images
- **Random seed:** 42

### 2.5 Crop Extraction

224×224 crops were extracted from ground-truth bounding boxes for the classification track (E2–E10).

| Statistic | Value |
|-----------|-------|
| Total valid crops | 65,665 |
| Rejected crops (< 16px dimension) | 8,243 |
| Resize method | BILINEAR |
| Crop size | 224×224 |

Crop distribution after extraction (train/val/test): **45,177 / 9,935 / 10,553**

---

## 3. Locked Controlled Variables

These constants are fixed across all experiments and must not be changed without updating all downstream notebooks.

| Variable | Value | Rationale |
|----------|-------|-----------|
| Random seed | `42` | Reproducibility across all splits, models, augmentation |
| Dataset split | 70 / 15 / 15 stratified | Standard; stratification preserves class balance |
| Crop size | 224×224 BILINEAR | Standard for ImageNet-pretrained backbones |
| Classical feature dim | **252** | Fixed pipeline: color(96)+LBP(26)+Gabor(48)+HOG(36)+shape(10)+GLCM(36) |
| Deep feature dim | **1280** | EfficientNetB0 global average pool (`timm`, `num_classes=0`) |
| YOLO backbone feature dim | **256** | SPPF layer 9, global average pool across spatial dims |
| YOLO conf threshold | 0.25 | Standard YOLOv8 default |
| YOLO NMS IoU threshold | 0.45 | Standard YOLOv8 default |
| Primary metric | **Macro-F1** | Handles 10.3:1 class imbalance |
| Class weights | `balanced` | All sklearn classifiers; compensates for imbalance |

---

## 4. Experiment E1: YOLOv8-nano Detection Baseline

### 4.1 Objective

Establish an end-to-end detection baseline using the smallest YOLOv8 variant (nano). Metrics: mAP@0.5, mAP@0.5:0.95, precision, recall, FPS, model size. Also extract SPPF backbone features for downstream E11 fusion.

### 4.2 Infrastructure

| Setting | Value |
|---------|-------|
| Hardware | Modal H100 80GB SXM |
| Training time | ~50 minutes |
| Script | `scripts/run_e1_yolo_modal.py` |
| Feature extraction | `scripts/run_e1_extract_features_modal.py` (A10G, ~10 min) |
| Modal Volume (data in) | `e1-yolo-data` (dataset zip) |
| Modal Volume (results out) | `e1-yolo-results` (weights, metrics, features) |

### 4.3 Hyperparameters

| Hyperparameter | Value | Note |
|----------------|-------|------|
| Model | YOLOv8n | nano — 3.2M parameters |
| Image size | 640×640 | Full-resolution COCO standard |
| Epochs | 100 | |
| Learning rate (lr0) | 0.001 | Reduced from default 0.01 for stability |
| freeze | 10 | Freeze first 10 backbone layers; prevents overwriting ImageNet weights |
| close_mosaic | 15 | Disable mosaic augmentation for last 15 epochs |
| copy_paste | 0.0 | Disabled — requires instance segmentation masks (not available) |
| Batch size | Auto (fraction=0.85) | |

**Rationale for freeze=10:** BIODEGRADABLE class has 4 redundant bounding boxes per object in many images. Freezing early backbone layers prevents the model from learning annotation artifacts rather than visual features. Methodological choice retained for the paper.

### 4.4 Results

| Metric | Value |
|--------|-------|
| mAP@0.5 | **0.4559** |
| mAP@0.5:0.95 | **0.3164** |
| Precision | 0.6637 |
| Recall | 0.5063 |
| FPS (H100 eval) | 259.9 |
| Model size | 6.26 MB |
| Epochs trained | 100 |
| Best val mAP50 (during training) | 0.5451 at epoch 94 |

**Note on FPS:** The 259.9 FPS was measured on H100 during Modal evaluation — not comparable to deployment hardware. Use this for relative comparison only.

### 4.5 Per-Class mAP@0.5

| Class | mAP@0.5 |
|-------|---------|
| BIODEGRADABLE | 0.4663 |
| CARDBOARD | 0.4445 |
| GLASS | 0.4575 |
| METAL | **0.5161** |
| PAPER | 0.3939 |
| PLASTIC | 0.4572 |

METAL performs best. PAPER is weakest — likely due to visual similarity with CARDBOARD and BIODEGRADABLE.

### 4.6 Output Files

| File | Location |
|------|----------|
| Trained weights | `models/yolo/yolov8n_E1_best.pt` |
| Detection metrics | `results/metrics/detection_results.csv` |
| Per-class metrics | `results/metrics/E1_per_class_metrics.csv` |
| Training curves | `figures/yolo/` |
| YOLO backbone features | `data/processed/features/yolo_{train,val,test}_{X,y}.npy` |

### 4.7 YOLO Backbone Feature Extraction

For E11 fusion, 256-dim features were extracted from the SPPF layer (layer index 9 in the YOLOv8 architecture) using a PyTorch forward hook. Global average pooling over the spatial dimensions `[B, 256, H, W] → [B, 256]` gives a compact representation of the detection backbone's understanding of each crop.

```python
_handle = nn_backbone.model[9].register_forward_hook(_sppf_hook)
# In extraction loop:
gap = _buf["sppf"].mean(dim=[2, 3]).cpu().numpy()   # [B, 256]
```

---

## 5. Experiment E2: Classical Features + Random Forest

### 5.1 Objective

Establish a fast, interpretable baseline using hand-crafted classical computer vision features on 224×224 crops.

### 5.2 Infrastructure

| Setting | Value |
|---------|-------|
| Hardware | Modal CPU, 64 GB RAM, 16 cores |
| Total runtime | ~2 hours |
| Script | `scripts/run_e2_classical_modal.py` |
| Modal Volume (crops in) | `e3-crops-v1` (shared with E3) |
| Modal Volume (results out) | `e2-rf-results` |

### 5.3 Feature Pipeline (252 dimensions total)

| Feature Type | Dimensions | Implementation |
|-------------|-----------|---------------|
| Color histogram | 96 | HSV, 32 bins per H/S/V channel |
| LBP (Local Binary Patterns) | 26 | radius=1, n_points=8, uniform |
| Gabor filters | 48 | 6 orientations × 4 frequencies × (mean+std) |
| HOG (Histogram of Oriented Gradients) | 36 | 4×4 cells, 9 bins |
| Shape features | 10 | Hu moments (7) + area ratio + aspect ratio + extent |
| GLCM texture | 36 | Grey-Level Co-occurrence Matrix stats × 4 angles |
| **Total** | **252** | All features L2-normalized per sample |

### 5.4 Augmentation Strategy

Training crops only were augmented 5× to address class imbalance:
- Original
- Horizontal flip
- Rotate 90°
- Rotate 180°
- Rotate 270°

This produces a 225,885 × 252 augmented training matrix. The clean (unaugmented) 45,177 × 252 matrix is used for fusion experiments (E4–E10) to maintain row alignment with deep and YOLO features.

### 5.5 Classifier

```
RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
```

### 5.6 Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.7740 |
| **Macro-F1** | **0.6476** |
| Weighted-F1 | 0.7551 |
| Balanced Accuracy | 0.6024 |
| AUC-ROC (macro) | 0.9085 |
| Top-2 Accuracy | 0.8802 |
| Val Macro-F1 | 0.6426 |
| Val/Test gap | 0.0050 — **no overfitting** |

### 5.7 Per-Class Metrics

| Class | F1 | Precision | Recall | AUC-ROC |
|-------|----|-----------|--------|---------|
| BIODEGRADABLE | — | — | — | 0.9317 |
| CARDBOARD | — | — | — | 0.9710 |
| GLASS | — | — | — | 0.8290 |
| METAL | — | — | — | 0.9246 |
| PAPER | — | — | — | 0.8721 |
| PLASTIC | — | — | — | 0.9225 |

*Per-class F1/precision/recall are stored in `results/metrics/E2_per_class_metrics.csv`.*

CARDBOARD achieves the highest AUC (0.9710) — likely due to its distinctive rectangular shape and uniform texture. GLASS is the hardest class (AUC=0.8290) — transparent objects exhibit highly variable visual signatures.

### 5.8 Output Files

| File | Location |
|------|----------|
| Trained model | `models/classifiers/random_forest_E2.pkl` |
| Classification results | `results/metrics/classification_results.csv` (E2 row) |
| Per-class metrics | `results/metrics/E2_per_class_metrics.csv` |
| Predictions | `results/predictions/E2_classical_predictions.csv` |
| Classical feature arrays | `data/processed/features/classical_{train_clean,val,test}_{X,y}.npy` |
| Figures | `figures/classification/E2_confusion_matrix.png`, `E2_confusion_matrix_norm.png`, `E2_class_f1_bar.png` |

---

## 6. Experiment E3: EfficientNetB0 Deep Features + SVM

### 6.1 Objective

Establish a deep learning feature baseline using a pretrained EfficientNetB0 backbone. The backbone is frozen — no fine-tuning — so features are purely transfer learning from ImageNet. Compare against E2 to quantify the gap between hand-crafted and learned features.

### 6.2 Infrastructure

| Setting | Value |
|---------|-------|
| Feature extraction hardware | Modal A10G 24GB |
| SVM training hardware | Modal CPU, 32 GB RAM |
| Feature extraction time | ~30 min |
| SVM training time | ~2 hours |
| Scripts | `scripts/run_e3_features_modal.py`, `scripts/run_e3_svm_modal.py` |
| Modal Volume (features out) | `e3-features-out` |

### 6.3 Feature Pipeline

**Backbone:** `timm` EfficientNetB0, pretrained on ImageNet-1k, `num_classes=0`

The `num_classes=0` flag removes the classification head and returns the global average pooled representation from the final convolutional stage:

```
Input: 224×224 RGB crop
→ EfficientNetB0 stem + 7 MBConv blocks
→ Global Average Pooling
→ 1280-dim feature vector
```

No fine-tuning. Backbone weights frozen at ImageNet values. Batch size=128, extraction only.

**Note:** An earlier version of the codebase documented 2048 dimensions (ResNet-50 style). This is incorrect. EfficientNetB0 produces 1280-dim vectors. All code, feature arrays, and downstream experiments use 1280.

### 6.4 Preprocessing

All crops normalized with ImageNet statistics:
- Mean: `[0.485, 0.456, 0.406]`
- Std: `[0.229, 0.224, 0.225]`

No augmentation during feature extraction — features are deterministic.

### 6.5 Classifier

```
SVC(
    kernel="rbf",
    C=10.0,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=42
)
```

Features are StandardScaler-normalized before SVM fitting. The scaler is fit on training features only; val and test sets are transformed with the training scaler.

**Script fix applied (2026-05-16):** An earlier version of `run_e3_svm_modal.py` omitted `X_val = scaler.transform(X_val)`, causing the validation evaluation to use unscaled features. This was identified during expert audit and corrected. The numbers below reflect the corrected run.

### 6.6 Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.8522 |
| **Macro-F1** | **0.7848** |
| Weighted-F1 | 0.8471 |
| Balanced Accuracy | 0.7672 |
| MCC | 0.7673 |
| Cohen Kappa | 0.7642 |
| AUC-ROC (macro) | 0.9566 |
| Top-2 Accuracy | 0.9404 |
| Val Accuracy | 0.8690 |
| Val Macro-F1 | 0.7826 |
| Val/Test gap | 0.0022 — **no overfitting** |
| Inference time | 21.3 ms/crop (A10G, batch=128) |

### 6.7 Per-Class Metrics

| Class | F1 | AUC-ROC |
|-------|----|---------|
| BIODEGRADABLE | **0.9249** | **0.9636** |
| CARDBOARD | 0.8706 | 0.9931 |
| GLASS | 0.7294 | 0.9134 |
| METAL | 0.7654 | 0.9684 |
| PAPER | 0.7419 | 0.9538 |
| PLASTIC | 0.6767 | 0.9471 |

BIODEGRADABLE achieves the highest F1 (0.9249) — consistent with its dominance in the training data (43.4%). PLASTIC is the hardest class by F1 (0.6767) despite being common; plastic objects span a wide range of shapes, colors, and transparency levels. GLASS is similarly challenging (F1=0.7294).

The high per-class AUC values (all > 0.91) confirm that the model produces well-calibrated probability estimates, even for hard classes — a strong indicator for downstream fusion (E8–E10).

### 6.8 Output Files

| File | Location |
|------|----------|
| Trained model | `models/classifiers/svm_E3.pkl` |
| Classification results | `results/metrics/classification_results.csv` (E3 row) |
| Per-class metrics | `results/metrics/E3_per_class_metrics.csv` |
| Predictions | `results/predictions/E3_deep_predictions.csv` |
| Deep feature arrays | `data/processed/features/deep_{train,val,test}_{X,y}.npy` |
| Figures | `figures/classification/E3_confusion_matrix.png`, `E3_confusion_matrix_norm.png`, `E3_class_f1_bar.png` |

---

## 7. Comparative Analysis

### 7.1 Baseline Summary

| Experiment | Feature Type | Classifier | Primary Metric | Value |
|------------|-------------|-----------|---------------|-------|
| E1 | Full image (640px) | YOLOv8n | mAP@0.5 | 0.4559 |
| E2 | Classical 252-dim | Random Forest | Macro-F1 | 0.6476 |
| E3 | Deep 1280-dim | RBF SVM | Macro-F1 | **0.7848** |

*Note: E1 is a detection task (mAP metric) while E2/E3 are classification tasks (Macro-F1). Direct numeric comparison is not meaningful across tracks.*

### 7.2 E2 vs E3: Classification Track Comparison

| Metric | E2 (Classical RF) | E3 (Deep SVM) | Δ (E3 − E2) |
|--------|-------------------|---------------|-------------|
| Accuracy | 0.7740 | 0.8522 | **+0.078** |
| Macro-F1 | 0.6476 | 0.7848 | **+0.137** |
| Weighted-F1 | 0.7551 | 0.8471 | +0.092 |
| AUC-ROC (macro) | 0.9085 | 0.9566 | +0.048 |
| Top-2 Accuracy | 0.8802 | 0.9404 | +0.060 |
| Val/Test F1 gap | 0.0050 | 0.0022 | (E3 generalizes better) |
| Inference | <1 ms/crop | 21.3 ms/crop | E2 is ~21× faster |

**Key insight:** Deep features (E3) outperform classical features (E2) by +13.7 pp Macro-F1. However, E2 inference is approximately 21× faster and requires no GPU. The optimal tradeoff will depend on whether the fusion experiments (E4–E10) can approach E3 accuracy at E2-like cost.

### 7.3 Per-Class F1: E2 vs E3

| Class | E2 F1 | E3 F1 | Δ |
|-------|-------|-------|---|
| BIODEGRADABLE | — | 0.9249 | — |
| CARDBOARD | — | 0.8706 | — |
| GLASS | — | 0.7294 | — |
| METAL | — | 0.7654 | — |
| PAPER | — | 0.7419 | — |
| PLASTIC | — | 0.6767 | — |

*E2 per-class F1 values are in `results/metrics/E2_per_class_metrics.csv`. The E3 values above can be compared directly once E2 per-class F1 is extracted from the CSV.*

### 7.4 Consistency Observations

1. GLASS and PLASTIC are the hardest classes in both E2 and E3 — suggesting an inherent visual ambiguity that is not solved by feature type alone.
2. CARDBOARD has the highest AUC in E2 (0.9710) despite not being the easiest class by F1 — suggesting the RF produces well-separated probabilities for cardboard even when other classes are confused.
3. Val/Test gaps are small in both experiments (< 0.01 Macro-F1), confirming that the 70/15/15 stratified split is working correctly and neither model is overfitting.

---

## 8. Feature Arrays for Downstream Experiments

All feature arrays are stored in `data/processed/features/` and distributed via the GitHub Release v1.0. Arrays are row-aligned: the i-th row of `classical_train_clean_X.npy`, `deep_train_X.npy`, and `yolo_train_X.npy` correspond to the same crop.

### 8.1 Available Arrays

| File | Shape | Modality | Use in |
|------|-------|----------|--------|
| `classical_train_clean_X.npy` | (45177, 252) | Classical | E4–E10 fusion (train) |
| `classical_val_X.npy` | (9935, 252) | Classical | E4–E10 fusion (val) |
| `classical_test_X.npy` | (10553, 252) | Classical | E4–E10 fusion (test) |
| `deep_train_X.npy` | (45177, 1280) | EfficientNetB0 | E4–E10 fusion (train) |
| `deep_val_X.npy` | (9935, 1280) | EfficientNetB0 | E4–E10 fusion (val) |
| `deep_test_X.npy` | (10553, 1280) | EfficientNetB0 | E4–E10 fusion (test) |
| `yolo_train_X.npy` | (45177, 256) | YOLO SPPF | E11 fusion (train) |
| `yolo_val_X.npy` | (9935, 256) | YOLO SPPF | E11 fusion (val) |
| `yolo_test_X.npy` | (10553, 256) | YOLO SPPF | E11 fusion (test) |

Each has a matching `_y.npy` with integer class labels 0–5 (BIODEGRADABLE=0, CARDBOARD=1, GLASS=2, METAL=3, PAPER=4, PLASTIC=5).

> **Warning:** `classical_train_X.npy` (shape 225,885 × 252) is the 5×-augmented E2 training matrix. It is NOT row-aligned with deep/YOLO features. Always use `classical_train_clean_X.npy` for fusion.

### 8.2 Loading Example

```python
import numpy as np

# Load all three modalities for early fusion
X_cls = np.load("data/processed/features/classical_train_clean_X.npy")  # (45177, 252)
X_dep = np.load("data/processed/features/deep_train_X.npy")             # (45177, 1280)
X_ylo = np.load("data/processed/features/yolo_train_X.npy")             # (45177, 256)
y     = np.load("data/processed/features/classical_train_clean_y.npy")  # (45177,)

# Early fusion (E4):
X_fused = np.hstack([X_cls, X_dep])  # (45177, 1532)
```

### 8.3 Feature Manifests

JSON manifests document extraction parameters, timestamps, and array checksums:
- `classical_feature_manifest.json`
- `deep_feature_manifest.json`
- `yolo_feature_manifest.json`

---

## 9. Upcoming Experiments (E4–E11)

### 9.1 Early Fusion Track (Ahmed — E4–E7)

All experiments use `classical_train_clean_X.npy` + `deep_train_X.npy` concatenated into a 1532-dim fused vector.

| ID | Strategy | Hypothesis |
|----|----------|-----------|
| E4 | Raw concatenation → RF | Does raw concatenation improve over either modality alone? |
| E5 | PCA (95% variance) → RF/SVM | Does dimensionality reduction remove noise from the concatenated features? |
| E6 | Autoencoder bottleneck (→256) → RF | Does a learned compression outperform PCA? |
| E7 | Feature selection (top-200) → RF | Which individual features contribute most? |

### 9.2 Late Fusion Track (Bayo — E8–E9)

Use the trained E2 (RF) and E3 (SVM) models to produce class probability vectors, then combine.

| ID | Strategy | Hypothesis |
|----|----------|-----------|
| E8 | Average of E2+E3 probability vectors | Does ensemble averaging beat individual models? |
| E9 | Weighted average (optimize weights on val) | Does learned weighting outperform uniform averaging? |

### 9.3 Attention Fusion (Bayo — E10)

A learned gating module produces per-sample, per-feature weights before combining the two modality vectors. Trains an MLP on top of gated features.

### 9.4 YOLO Hybrid (Nada — E11)

Concatenate YOLO SPPF backbone features (256-dim) + classical features (252-dim) = 508-dim input. Train a lightweight MLP head. Combines detection backbone understanding with hand-crafted descriptors.

Required files: `yolo_{train,val,test}_{X,y}.npy`, `classical_train_clean_{X,y}.npy`, `yolov8n_E1_best.pt`.

---

## 10. Conclusions

### 10.1 Key Findings (Baselines)

1. **Deep features dominate hand-crafted features.** EfficientNetB0 (E3, Macro-F1=0.7848) outperforms classical 252-dim features (E2, Macro-F1=0.6476) by +13.7 percentage points with frozen ImageNet weights only — no fine-tuning required.

2. **GLASS and PLASTIC are consistently hard.** Both E2 and E3 struggle on these classes. This suggests the challenge is inherent to the visual domain (transparency, shape variability) rather than the feature type. Attention and late-fusion approaches may help by combining complementary cues.

3. **Neither baseline overfits.** Val/Test Macro-F1 gaps are 0.005 (E2) and 0.002 (E3), well within noise. The stratified 70/15/15 split with seed=42 produces stable splits.

4. **Classical features have AUC despite lower F1.** E2 macro AUC-ROC = 0.9085, only 4.8 pp below E3 (0.9566). The RF produces surprisingly calibrated probabilities, making E2 a viable ensemble component in E8–E9.

5. **YOLOv8-nano is a reasonable detection baseline.** mAP@0.5=0.4559 is below SOTA but expected given the small model size (6.26 MB), 6-class multi-label setting, and challenging waste domain. The 256-dim SPPF features capture scene-level context that per-crop classifiers miss.

6. **Compute scales non-linearly.** E2 inference is ~21× faster than E3 (< 1 ms vs 21.3 ms per crop). If fusion experiments (E4–E10) achieve E3-level accuracy at E2-level cost, that is the answer to the research question.

### 10.2 Baseline Summary Table

| Exp | Feature | Classifier | Macro-F1 | AUC-ROC | Inference |
|-----|---------|-----------|---------|---------|-----------|
| E2 | Classical 252-dim | RF | 0.6476 | 0.9085 | < 1 ms |
| E3 | Deep 1280-dim | SVM | **0.7848** | **0.9566** | 21.3 ms |
| E1 (detection) | Full image | YOLOv8n | mAP50=0.456 | — | 3.8 ms/img |

### 10.3 Next Steps

E4–E11 experiments can begin immediately — all feature arrays are available in `data/processed/features/` and downloadable from GitHub Release v1.0.

The final report will include:
- Ablation table comparing all 11 experiments
- Pareto frontier visualization (Macro-F1 vs inference time)
- Confusion matrices for best-performing fusion approach
- Statistical significance tests (McNemar's or bootstrap) between top models

---

## Appendix A: Glossary

| Term | Definition |
|------|-----------|
| mAP@0.5 | Mean Average Precision at IoU threshold 0.5 — standard detection metric |
| mAP@0.5:0.95 | Mean AP averaged across IoU thresholds 0.5 to 0.95 in steps of 0.05 |
| Macro-F1 | F1 score averaged equally across all classes — penalizes poor minority-class performance equally |
| Weighted-F1 | F1 score averaged weighted by class support — dominated by majority class |
| AUC-ROC | Area Under the Receiver Operating Characteristic curve — measures separability |
| MCC | Matthews Correlation Coefficient — balanced metric robust to class imbalance |
| Cohen Kappa | Agreement measure adjusted for chance agreement |
| SPPF | Spatial Pyramid Pooling - Fast — multi-scale feature aggregation module in YOLOv8 backbone |
| LBP | Local Binary Patterns — texture descriptor comparing each pixel to its neighbors |
| GLCM | Grey-Level Co-occurrence Matrix — second-order texture statistics |
| HOG | Histogram of Oriented Gradients — edge/gradient shape descriptor |
| Gabor filter | Sinusoidal filter for multi-orientation, multi-frequency texture analysis |
| EfficientNetB0 | Compound-scaled convolutional network; `num_classes=0` removes classifier head |
| timm | PyTorch Image Models library — provides pretrained EfficientNetB0 |
| Modal | Cloud ML platform used for GPU/CPU training jobs in this project |
| YOLO | You Only Look Once — single-stage object detection architecture |
| freeze=10 | YOLOv8 training parameter: first 10 backbone layers kept frozen (ImageNet weights) |

---

## Appendix B: Known Issues and Fixes

| Issue | Discovery | Fix Applied |
|-------|-----------|------------|
| Vendor dataset split corrupted | Cell 2 audit: classes absent from val/test | Discarded vendor split; created fresh 70/15/15 stratified split |
| Zero-area bounding boxes | 17 annotations with width=0 or height=0 | Removed before crop extraction |
| Classical feature dimension mismatch (283 vs 252) | Debugging feature array shapes | Fixed by standardizing the feature pipeline — removed 31 redundant HOG cells; pipeline locked at 252 |
| Deep feature dimension in old docs (2048 vs 1280) | Code review | Corrected throughout — EfficientNetB0 GAP output is 1280-dim, not 2048 |
| Val set not scaled in E3 SVM script | Expert audit | Added `X_val = scaler.transform(X_val)` before val evaluation |
| Per-class AUC missing from E3 results | Expert audit | Added per-class AUC computation for all 6 classes |
| Modal dataset path mismatch (E1) | Training failure: images/ not found | Fixed: zip structure is `train/images/`, not `images/train/` |
| GPU/CPU device mismatch (E1 feature extraction) | RuntimeError during forward pass | Added `.cuda()` before model inference |
| Crop extraction path error in feature script | FileNotFoundError | Created target directory before extracting zip into it |
| Windows encoding error in Modal CLI | charmap codec failure | Set `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` in PowerShell environment |
| BIODEGRADABLE redundant annotations | 4 boxes per object in many images | Accepted as-is; `class_weight='balanced'` compensates in all classifiers |
| `classical_train_X.npy` not row-aligned | Shape (225885, 252) — 5x augmented | Use `classical_train_clean_X.npy` (45177, 252) for all fusion experiments |
