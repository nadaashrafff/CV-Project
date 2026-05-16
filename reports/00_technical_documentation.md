# Technical Documentation: Data Preprocessing Pipeline (Cells 1–5)

**Project:** Compression-Aware Multiscale Feature Fusion for Waste Object Detection and Classification  
**Dataset:** Garbage Classification (Roboflow v2) — 10,464 images, 6 classes  
**Team Member:** Aly (Data Preparation & Baseline Phases)  
**Environment:** Google Colab GPU Runtime  
**Date:** 2026-05-14  
**Random Seed:** 42 (fixed for reproducibility)

---

## Table of Contents

1. [Project Scope & Dataset Expectations](#1-project-scope--dataset-expectations)
2. [Cell 1: Dataset Reconnaissance & Structure Analysis](#2-cell-1-dataset-reconnaissance--structure-analysis)
3. [Cell 2: Format Selection & Class Name Forensics](#3-cell-2-format-selection--class-name-forensics)
4. [Cell 3: Deep Cleaning & Quality Audit](#4-cell-3-deep-cleaning--quality-audit)
5. [Cell 4: Stratified 70/15/15 Split](#5-cell-4-stratified-701515-split)
6. [Cell 5: Object Crop Extraction](#6-cell-5-object-crop-extraction)
7. [Verification Cell: Visual Audit & Split Corruption Check](#7-verification-cell-visual-audit--split-corruption-check)
8. [Shared Artifacts & File Structure](#8-shared-artifacts--file-structure)
9. [Known Issues & Resolutions](#9-known-issues--resolutions)
10. [Next Steps](#10-next-steps)

---

## 1. Project Scope & Dataset Expectations

### Original Specification (from Project Description)

| Requirement | Original Spec | Actual Dataset | Status |
|-------------|---------------|----------------|--------|
| Total Images | ~4,142 | **10,464** | ✅ 2.5× more data |
| Classes | 7 (merge Biodegradable→Trash) | **6 native classes** | ✅ No merge needed |
| Format | YOLO bounding boxes (5 cols) | ✅ Standard YOLO bbox | Perfect |
| Annotations | Bounding boxes | ✅ Bounding boxes | Perfect |
| Task | Detection + Classification | ✅ Detection + Classification | Perfect |

### Final Class Taxonomy

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | **BIODEGRADABLE** | Organic waste: fruits, vegetables, compost, food scraps |
| 1 | **CARDBOARD** | Corrugated boxes, packaging, flat cardboard pieces |
| 2 | **GLASS** | Bottles, jars, transparent containers |
| 3 | **METAL** | Cans, foil trays, metallic containers |
| 4 | **PAPER** | Newspaper, shredded paper, books, printed materials |
| 5 | **PLASTIC** | Bottles, containers, plastic packaging |

**Note:** The original project description specified 7 classes with Biodegradable merged into Trash. This dataset contains 6 native classes with Biodegradable as a standalone class. The core research question (feature fusion & compression) remains fully intact.

---

## 2. Cell 1: Dataset Reconnaissance & Structure Analysis

### Purpose
Mount Google Drive, extract dataset ZIP, discover data root, and perform full structural audit.

### Input
- `ZIP_PATH = /content/drive/MyDrive/CV/Trash_Dataset.zip`
- Archive size: **986.5 MB** (first wrong dataset) → **190.7 MB** (correct dataset)

### Process
1. Mount Google Drive (`/content/drive`)
2. Extract ZIP to fast local storage (`/content/dataset`)
3. Auto-discover dataset root (handles nested folders)
4. Recursive inventory of all files and directories
5. Image-label pairing by filename stem (case-insensitive)
6. Annotation format forensics (5-column YOLO validation)
7. Preliminary class distribution

### Output

```
📊 Structural Audit:
   Total directories: 10
   Total files:       20,929

📂 Top-level contents:
   📂 GARBAGE CLASSIFICATION/  (20,938 items recursive)

📋 File extensions:
   .jpg : 10,464
   .txt : 10,464
   .yaml:     1
```

**Key Finding:** Perfect 1:1 image-to-label ratio. No orphan files.

### Critical Discovery: Wrong Dataset Initially Downloaded

First extraction revealed:
- **12,426 images** (triple dataset: COCO + YOLOv5 + YOLOv9 formats)
- **29 classes** (segmentation polygons, not bounding boxes)
- Dataset source: `viswaprakash1990/garbage-detection` (incorrect)

**Resolution:** Replaced with correct dataset from `ahnaftahmeed/trash-detection-image-dataset` (renamed locally to same path). Re-ran Cell 1 with forced extraction cleanup.

---

## 3. Cell 2: Format Selection & Class Name Forensics

### Purpose
Resolve class names from YAML metadata, verify annotation format, and perform GO/NO-GO compatibility check.

### Process
1. Parse `data.yaml` for class names
2. Infer class names from filenames (cross-validation)
3. Compare found classes vs. project requirements
4. Binary GO/NO-GO verdict

### YAML Metadata

```yaml
nc: 6
names: ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
roboflow:
  workspace: material-identification
  project: garbage-classification-3
  version: 2
  license: CC BY 4.0
```

### Filename-Based Inference (Cross-Validation)

| Class ID | Inferred from Filenames | YAML Name | Match |
|----------|------------------------|-----------|-------|
| 0 | biodegradable | BIODEGRADABLE | ✅ |
| 1 | cardboard | CARDBOARD | ✅ |
| 2 | glass | GLASS | ✅ |
| 3 | metal | METAL | ✅ |
| 4 | paper | PAPER | ✅ |
| 5 | plastic | PLASTIC | ✅ |

### Verdict

**GO:** Dataset is suitable for the project. Class names match waste taxonomy. Core research question (feature fusion & compression) remains intact.

**Required Report Update:** Section 5 (Dataset) must reflect 6 native classes instead of 7 with merge.

---

## 4. Cell 3: Deep Cleaning & Quality Audit

### Purpose
Remove corrupt annotations, detect duplicates, verify image integrity, and generate class distribution visualizations.

### Process
1. **YAML class name resolution** — lock down exact 6-class taxonomy
2. **Zero-area bounding box cleaning** — remove boxes with `w=0` or `h=0`
3. **Image integrity check** — verify all images readable by PIL
4. **Exact duplicate detection** — MD5 hash comparison
5. **Class distribution visualization** — bar chart + pie chart
6. **Create YOLO-ready unified pool** — flatten all splits for custom reshuffle

### Cleaning Results

| Metric | Value |
|--------|-------|
| Original matched pairs | 10,464 |
| Corrupt images excluded | **0** |
| Zero-area boxes removed | **17** |
| Images emptied by cleaning | **0** |
| Exact duplicate groups | **0** |
| Exact duplicate instances | **0** |
| Clean images in pool | **10,464** |
| Clean objects total | **74,073** |
| Classes | **6** |

### Zero-Area Boxes Removed (Log)

| Image | Line | Class | Issue |
|-------|------|-------|-------|
| paper1430_jpg.rf... | 9 | PAPER | w=0.000, h=0.006 |
| paper1805_jpeg.rf... | 1 | PAPER | w=0.000, h=0.005 |
| biodegradable1998_jpeg.rf... | 261 | BIODEGRADABLE | w=0.063, h=0.000 |
| ... | ... | ... | ... |

**Impact:** Negligible. 17 boxes out of 74,090 total = **0.023%**. No images were emptied.

### Class Distribution (After Cleaning)

| Class | Objects | % of Total | Images |
|-------|---------|------------|--------|
| BIODEGRADABLE | 45,395 | **61.3%** | 2,287 |
| GLASS | 7,809 | 10.5% | 2,684 |
| PLASTIC | 5,945 | 8.0% | 1,464 |
| METAL | 5,841 | 7.9% | 1,898 |
| CARDBOARD | 4,698 | 6.3% | 1,502 |
| PAPER | 4,390 | 5.9% | 1,768 |

**Class imbalance ratio:** 10.3:1 (BIODEGRADABLE vs. PAPER)

**Mitigation strategy:** Stratified sampling + `class_weight='balanced'` in downstream classifiers + Macro-F1 as primary metric.

### Image Properties

| Property | Value |
|----------|-------|
| Resolution | **Uniform 416×416** |
| Color mode | **RGB** (100%) |
| Unique resolutions | **1** (416×416) |

**Significance:** Uniform resolution eliminates variable-resize artifacts. All crops and features will have consistent input dimensions.

### Generated Artifacts

| File | Description |
|------|-------------|
| `02_removed_zero_area_boxes.csv` | Log of 17 removed invalid boxes |
| `04_master_inventory_cleaned.csv` | Full metadata for all 10,464 images |
| `Fig_01_object_count_per_class.png` | Bar chart: objects per class |
| `Fig_02_class_proportion.png` | Pie chart: class proportions |

---

## 5. Cell 4: Stratified 70/15/15 Split

### Purpose
Create reproducible, stratified train/validation/test split with data leakage protection.

### Why Original Split Was Discarded

| Split | Original (Roboflow) | Issue |
|-------|---------------------|-------|
| Train | 70.0% / 20.0% / 10.0% | Wrong ratios |
| Valid | PAPER: 0.1%, PLASTIC: 0.0% | **Class missing** |
| Test | GLASS: 0.0%, BIODEGRADABLE: 0.3% | **Class missing** |

The original vendor split was **severely corrupted** — entire classes were absent from validation and test sets.

### Stratification Strategy

**Problem:** Multi-object images (one image can contain multiple classes).

**Solution:** Assign each image a **dominant class** (the class with the most bounding boxes). Stratify the split on this dominant label.

**Why this is correct:**
- Project description explicitly mandates: *"The split must be done at the image level, not object level"*
- All objects in a multi-class image move together into one split
- Minority objects "hitchhike" with the dominant class without leakage

### Split Results

| Split | Images | % | Objects |
|-------|--------|---|---------|
| **Train** | 7,324 | **70.0%** | 50,957 |
| **Valid** | 1,570 | **15.0%** | 11,643 |
| **Test** | 1,570 | **15.0%** | 11,473 |
| **Total** | 10,464 | 100% | 74,073 |

### Per-Split Image Counts by Dominant Class

| Class | Train | Valid | Test | Valid% | Test% |
|-------|-------|-------|------|--------|-------|
| BIODEGRADABLE | 1,564 | 335 | 335 | 21.4% | 21.3% |
| CARDBOARD | 1,013 | 217 | 217 | 13.8% | 13.8% |
| GLASS | 1,816 | 389 | 389 | 24.8% | 24.8% |
| METAL | 885 | 190 | 190 | 12.1% | 12.1% |
| PAPER | 1,218 | 261 | 262 | 16.6% | 16.7% |
| PLASTIC | 828 | 178 | 177 | 11.3% | 11.3% |

**Verification:** All classes present in all splits. Proportional representation maintained.

### Leakage Protection

| Check | Result |
|-------|--------|
| Train ∩ Val (stem overlap) | **0 images** |
| Train ∩ Test (stem overlap) | **0 images** |
| Val ∩ Test (stem overlap) | **0 images** |
| **Verdict** | **PASS — splits are disjoint** |

### YOLO Directory Structure Created

```
/content/dataset_split_70_15_15/
├── data.yaml              # YOLO config
├── train/
│   ├── images/            # 7,324 JPGs
│   └── labels/            # 7,324 TXTs
├── valid/
│   ├── images/            # 1,570 JPGs
│   └── labels/            # 1,570 TXTs
└── test/
    ├── images/            # 1,570 JPGs
    └── labels/            # 1,570 TXTs
```

### `data.yaml` Content

```yaml
path: /content/dataset_split_70_15_15
train: train/images
val: valid/images
test: test/images

nc: 6
names: ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
```

### Generated Artifacts

| File | Description |
|------|-------------|
| `data.yaml` | YOLO dataset configuration |
| `05_split_statistics.csv` | Per-split object/image counts |
| `Fig_03_split_distribution_images.png` | Image count per class per split |
| `Fig_04_split_distribution_objects.png` | Object count per class per split |

---

## 6. Cell 5: Object Crop Extraction

### Purpose
Extract individual object crops from ground-truth bounding boxes for the classification track (E2–E10).

### Input
- Ground-truth bounding boxes from `dataset_split_70_15_15/*/labels/*.txt`
- Full images from `dataset_split_70_15_15/*/images/*.jpg`

### Process
1. For each image, read all valid YOLO annotations (5 columns)
2. Convert normalized coordinates to pixel coordinates
3. Clamp to image bounds
4. Validate minimum crop size (≥16 pixels)
5. Extract crop and resize to **224×224**
6. Save to class-organized directory structure

### Crop Size Rationale

| Target | Why 224×224 |
|--------|-------------|
| EfficientNetB0 | Native input size 224×224 |
| Classical features | Consistent spatial resolution for texture/shape |
| YOLO hybrid (E11) | Standard preprocessing dimension |

### Extraction Results

| Metric | Value |
|--------|-------|
| Total crops extracted | **65,830** |
| Rejected (too small) | **8,243** |
| Rejected (out of bounds) | **0** |
| Rejected (read error) | **0** |
| Crop size | **224×224** |

**Note:** 8,243 rejected crops were tiny annotation artifacts (width or height <16px). These were correctly filtered as they would be useless for feature extraction.

### Per-Split Crop Distribution

| Split | Total Crops | BIODEGRADABLE | GLASS | PLASTIC | METAL | CARDBOARD | PAPER |
|-------|-------------|---------------|-------|---------|-------|-----------|-------|
| Train | 45,258 | 26,787 (59.2%) | 4,944 | 3,790 | 3,691 | 3,143 | 2,903 |
| Valid | 10,495 | 5,764 (54.9%) | 1,397 | 1,066 | 871 | 722 | 675 |
| Test | 10,077 | 5,892 (58.5%) | 1,164 | 856 | 777 | 716 | 672 |

### Directory Structure

```
/content/dataset_crops/
├── train/
│   ├── BIODEGRADABLE/     # 26,787 crops
│   ├── CARDBOARD/         # 3,143 crops
│   ├── GLASS/             # 4,944 crops
│   ├── METAL/             # 3,691 crops
│   ├── PAPER/             # 2,903 crops
│   └── PLASTIC/           # 3,790 crops
├── valid/
│   └── [same 6 classes]
└── test/
    └── [same 6 classes]
```

### Generated Artifacts

| File | Description |
|------|-------------|
| `06_crop_metadata.csv` | Full metadata: path, split, class, source image, bbox coordinates |
| `Fig_09_10_crop_distributions.png` | Crop count bar charts |
| `Fig_11_sample_crops.png` | 6×6 grid: sample crops per class |

---

## 7. Verification Cell: Visual Audit & Split Corruption Check

### Purpose
Explicitly verify assumptions via visual inspection and numerical cross-checks.

### Tests Performed

| Test | Method | Finding |
|------|--------|---------|
| **Visual bbox verification** | Overlay all boxes on 5 random images | 1–4 boxes per image, all distinct objects |
| **Original split audit** | Count classes in vendor train/valid/test | **Confirmed corrupted**: PAPER 0.1% in valid, GLASS 0% in test |
| **Our reshuffle audit** | Count classes in our splits | **Fixed**: all classes 11–25% per split |
| **Redundancy check** | Pairwise IoU between boxes in same image | No overlap >0.5 — boxes are distinct |
| **Single-class rate** | Sample 1,500 images | **89.0% single-class**, 11.0% multi-class |

### Key Visual Findings

| Image | Objects | Annotation Lines | Conclusion |
|-------|---------|------------------|------------|
| paper1357 | 1 shredded paper | 1 | ✅ Single object, single box |
| glass2981 | 2 green bottles | 2 | ✅ **Multiple distinct objects** |
| biodegradable1011 | 1 rotting tomato | 4 | ⚠️ Same object, multiple overlapping annotations |
| metal841 | 1 foil tray | 1 | ✅ Single object, single box |
| metal636 | 1 crumpled foil | 1 | ✅ Single object, single box |

**BIODEGRADABLE class has noisy/redundant annotations** (4 boxes for 1 tomato). This is a known limitation but does not block the pipeline — all valid boxes are extracted as crops, and `class_weight='balanced'` will compensate for the inflated count.

---

## 8. Shared Artifacts & File Structure

### Complete Drive Output Tree

```
/content/drive/MyDrive/CV/
├── 01_dataset_analysis/
│   ├── 02_removed_zero_area_boxes.csv
│   ├── 04_master_inventory_cleaned.csv
│   ├── Fig_01_object_count_per_class.png
│   └── Fig_02_class_proportion.png
├── 02_dataset_split/
│   ├── data.yaml
│   ├── 05_split_statistics.csv
│   ├── Fig_03_split_distribution_images.png
│   └── Fig_04_split_distribution_objects.png
├── 04_crops/
│   ├── 06_crop_metadata.csv
│   ├── Fig_09_10_crop_distributions.png
│   └── Fig_11_sample_crops.png
└── [03_yolo_baseline/ — created by E1]
```

### Local (Colab) Working Directories

```
/content/
├── dataset/                    # Original extraction (temporary)
├── dataset_yolo_ready/         # Unified pool (temporary)
├── dataset_split_70_15_15/     # Final YOLO split (temporary)
└── dataset_crops/              # Object crops (temporary)
```

**Note:** Local directories are temporary (Colab runtime). All persistent outputs are saved to Drive.

---

## 9. Known Issues & Resolutions

| Issue | Severity | Resolution | Status |
|-------|----------|------------|--------|
| **Wrong dataset initially downloaded** | High | Replaced with correct Kaggle dataset | ✅ Fixed |
| **Original Roboflow split corrupted** | High | Discarded vendor split; created custom stratified 70/15/15 | ✅ Fixed |
| **Class imbalance (10.3:1)** | Medium | Stratified sampling + `class_weight='balanced'` + Macro-F1 metric | ✅ Mitigated |
| **BIODEGRADABLE redundant annotations** | Low | Multiple boxes per object extracted as multiple crops; class weights compensate | ⚠️ Known, acceptable |
| **8,243 tiny crops rejected** | Low | Filtered by minimum size check (≥16px) | ✅ Correctly handled |
| **Uniform 416×416 resolution** | Neutral | No variable-resize artifacts; consistent for all pipelines | ✅ Advantage |

---

## 10. Next Steps

### Immediate (Aly)

| Step | Notebook | Description |
|------|----------|-------------|
| **Cell 6** | `01_Preprocessing.ipynb` | Classical feature extraction (252-dim) — **BUG: dimension mismatch 283 vs 252. Fix in progress.** |
| **Cell 7** | `01_Preprocessing.ipynb` | Deep feature extraction (EfficientNetB0, 2,048-dim) — pending Cell 6 fix |
| **E1: YOLOv8-nano** | `02_E1_YOLO_Baseline.ipynb` | Object detection baseline — **can start immediately** (independent of Cells 6–7) |

### Parallel (Teammates)

| Teammate | Notebook | Experiment | Depends On |
|----------|----------|------------|------------|
| **Ahmed** | `04_E4_E7_Early_Fusion.ipynb` | E4–E7: Raw fusion, PCA, Autoencoder, Feature selection | Cell 6 + Cell 7 |
| **Bayo** | `05_E8_E9_Late_Fusion.ipynb` | E8–E9: Average voting, Weighted voting | E2 + E3 |
| **Bayo** | `06_E10_Attention_Fusion.ipynb` | E10: Attention/gating network | Cell 6 + Cell 7 |
| **Nada** | `07_E11_YOLO_Hybrid.ipynb` | E11: YOLO FPN + classical features | E1 + Cell 6 |

### YOLO Training Can Start NOW

YOLOv8-nano (E1) uses:
- ✅ `dataset_split_70_15_15/data.yaml`
- ✅ `dataset_split_70_15_15/train/images/`
- ✅ `dataset_split_70_15_15/train/labels/`

**Does NOT require:** Cells 6 or 7 (those are for the classification track).

---

## Appendix A: Controlled Variables (Locked)

| Variable | Value | Locked By |
|----------|-------|-----------|
| Dataset split | 70/15/15 stratified | Cell 4 |
| Class mapping | 6 classes (0–5) | Cell 3 YAML |
| Crop size | 224×224 | Cell 5 |
| Random seed | 42 | All cells |
| Image preprocessing | BILINEAR resize | Cell 5 |
| YOLO confidence threshold | 0.25 | E1 config |
| NMS IoU threshold | 0.45 | E1 config |

## Appendix B: Reproducibility Checklist

To reproduce this preprocessing pipeline:

1. Download dataset: `ahnaftahmeed/trash-detection-image-dataset` from Kaggle
2. Upload to `/content/drive/MyDrive/CV/Trash_Dataset.zip`
3. Run Cell 1 (with forced extraction cleanup)
4. Run Cell 2 (verify GO verdict)
5. Run Cell 3 (cleaning + audit)
6. Run Cell 4 (stratified split)
7. Run Cell 5 (crop extraction)
8. Verify outputs in `/content/drive/MyDrive/CV/`

**All random operations use `seed=42`.**

---

*Documentation generated: 2026-05-14 18:27*  
*Author: Aly (Data Preparation Lead)*  
*Reviewed by: Senior Data Scientist (Kimi)*
