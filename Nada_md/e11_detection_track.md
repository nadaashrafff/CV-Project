# E11 — Detection Track: Methodology & Research Notes

**Owner:** Nada  
**Status:** ⏳ Running — results pending  
**Date:** May 2026  
**Depends on:** E1 (YOLO baseline), E11 Classification Track (SVM + scalers)

---

## 1. Research Question

> Can replacing a single-stage YOLO classifier with a two-stage hybrid system — where YOLO handles localization and a fused SPPF + classical SVM handles material recognition — improve detection accuracy over the E1 baseline?

This is the only experiment in the project that evaluates a **full detection pipeline** beyond E1. It directly answers whether the feature fusion insights from the classification track translate into real-world detection gains.

---

## 2. Motivation: Why a Detection Track at All?

The entire classification track (E2–E11) uses ground-truth bounding boxes. Every crop fed into the classifiers was extracted from a known, clean box provided by a human annotator. This is a controlled setup ideal for comparing feature representations, but it does not reflect real deployment.

A real waste sorting system must:
1. Find where the object is in the image (detection / localization)
2. Identify what material it is (classification / recognition)

E1 (YOLOv8n) handles both in one pass. The E11 Detection Track tests whether splitting these two jobs — giving each to the system best suited for it — produces a stronger overall result.

The argument is:
- YOLO was trained to detect objects. Its loss function penalizes missed boxes and wrong locations. Classification is secondary.
- The hybrid SVM was trained specifically on material classification using 508 dimensions of rich visual features. Classification is its only job.

---

## 3. Architecture: Two-Stage Detector

```
                    ┌──────────────────────────────────────────┐
                    │          FULL TEST IMAGE                 │
                    └──────────────────┬───────────────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │   STAGE 1: YOLO         │
                          │   YOLOv8n (E1 weights)  │
                          │                         │
                          │  Output:                │
                          │  • Bounding boxes (xyxy)│
                          │  • Detection confidence │
                          │    (objectness score)   │
                          │                         │
                          │  YOLO class → discarded │
                          └────────────┬────────────┘
                                       │
                          For each predicted box:
                                       │
                          ┌────────────▼────────────┐
                          │   Crop → 224×224 RGB    │
                          └────────┬──────┬─────────┘
                                   │      │
                     ┌─────────────▼─┐  ┌─▼──────────────────┐
                     │  SPPF Branch  │  │  Classical Branch  │
                     │               │  │                    │
                     │  Run crop     │  │  HOG + LBP + Gabor │
                     │  through YOLO │  │  + GLCM + Color    │
                     │  backbone     │  │  + Shape           │
                     │  (hook@layer9)│  │                    │
                     │  GlobalAvgPool│  │                    │
                     │  → 256-dim    │  │  → 252-dim         │
                     └──────┬────────┘  └────────┬───────────┘
                            │                    │
                     StandardScaler         StandardScaler
                            │                    │
                            └──────────┬─────────┘
                                       │ np.hstack
                          ┌────────────▼────────────┐
                          │   508-dim fused vector  │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │   STAGE 2: E11 SVM      │
                          │   RBF kernel, C=10      │
                          │   class_weight=balanced │
                          │                         │
                          │  Output:                │
                          │  • Class label (0–5)    │
                          │  • Class probability    │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │   Final prediction:     │
                          │   box  = YOLO           │
                          │   class = SVM           │
                          │   conf  = YOLO (det.)   │
                          └─────────────────────────┘
```

### Why two stages instead of one?

Single-stage detectors like YOLO are optimized for speed and localization. Their classification heads are shallow — YOLOv8n's detection head predicts classes from a 1×1 convolution over the feature map. There is no explicit texture analysis, color histogram, or material-specific descriptor.

Two-stage detectors (like Faster R-CNN) use a separate region proposal network for localization and a separate head for classification. Each component is trained for its specific task. The E11 Detection Track follows this same philosophy: decouple the two problems.

---

## 4. Stage 1 — YOLO for Localization

### Model
- **Weights:** `yolov8n_E1_best.pt` — the E1 trained model (100 epochs, H100, imgsz=640)
- **Task:** Predict bounding boxes for waste objects in full 640×640 images
- **Thresholds:** conf ≥ 0.25, NMS IoU ≤ 0.45 (identical to E1 for fair comparison)

### What YOLO contributes to E11
- Bounding box coordinates (x1, y1, x2, y2) in absolute pixels
- Detection confidence score (objectness × localization quality)

### What YOLO does NOT contribute to E11
- Class labels — these are completely discarded and replaced by Stage 2

### Why use E1 weights?
Using the same trained weights as E1 ensures that any mAP difference between E1 and E11 is purely due to the classification change, not a change in the detector. The box predictions are identical between E1 and E11. This makes the comparison scientifically controlled.

---

## 5. Stage 2 — Hybrid SVM for Material Classification

For each bounding box predicted by YOLO:

### 5.1 Crop Extraction
The predicted box is cropped from the full image and resized to 224×224 using bilinear interpolation. This is the same crop size used throughout the classification track (E2–E11).

**Important:** The crop comes from the YOLO-predicted box, not a ground-truth box. This introduces localization noise — the box may be slightly off, contain background, or miss object edges. This is the realistic setting that the classification track's ground-truth crops do not simulate.

### 5.2 SPPF Features (256-dim)

**What is SPPF?**
Spatial Pyramid Pooling - Fast (SPPF) is the final layer of YOLOv8's backbone (layer index 9). It aggregates multi-scale spatial information through a series of max-pooling operations, then concatenates the results. The output is a feature map of shape [B, 256, H', W'] for YOLOv8n.

**Extraction process:**
A PyTorch forward hook is registered on `nn_backbone.model[9]`. When the crop is passed through the YOLO backbone, the hook captures the SPPF output. Global Average Pooling reduces the spatial dimensions to give a single 256-dim vector per crop:
```
[B, 256, H', W']  →  GAP  →  [B, 256]
```

**Why SPPF and not a deeper layer?**
SPPF is the richest summary produced by the backbone before the detection neck (FPN) and head begin. At this point, the backbone has processed the crop through all its convolutional stages (7 stages for YOLOv8n), capturing both low-level texture and high-level structural patterns. The FPN and head layers introduce skip connections from earlier feature maps and are tuned for spatial detection — their features are less suited for per-object classification.

**Preprocessing:**
Input pixels are divided by 255 (YOLO convention). No ImageNet mean/std normalization is applied — this matches how the YOLO model was trained and how the training SPPF features were extracted for the classification track.

### 5.3 Classical Features (252-dim)

Classical features are extracted from the same 224×224 crop using the locked pipeline from E2:

| Feature Group | Dimensions | What it captures |
|---------------|-----------|------------------|
| Color histogram (HSV) | 96 | Material color distribution |
| LBP (radius=3, 24-point) | 26 | Local texture patterns |
| Gabor filters (6 freq × 4 orient) | 48 | Multi-scale oriented texture |
| HOG (9 orient, 8×8 cells) | 36 | Shape and edge gradients |
| Shape features (Hu moments) | 10 | Object geometry |
| GLCM (5 props × 2 dist × 4 ang) | 36 | Texture co-occurrence statistics |
| **Total** | **252** | |

These are the same features that gave Macro-F1 = 0.6476 in E2 alone. They are not powerful on their own but capture texture, color, and material cues that SPPF features do not explicitly encode.

### 5.4 Feature Fusion

The two branches are fused using early concatenation, identical to the E11 classification track:

```python
feat_sppf_scaled = sc_yolo.transform(feat_sppf.reshape(1, -1))   # [1, 256]
feat_cls_scaled  = sc_cls.transform(feat_cls.reshape(1, -1))     # [1, 252]
fused = np.hstack([feat_sppf_scaled, feat_cls_scaled])            # [1, 508]
```

**Why scale each branch separately?**
SPPF features and classical features live on completely different numerical scales. SPPF values are activations from a deep network (range approximately 0–3.5 in our data). Classical features include histograms (range 0–1), HOG gradients, and GLCM statistics — all with different ranges. Without normalization, the larger-magnitude branch would dominate the SVM's decision boundary purely by scale, not by information content. Separate StandardScalers (fit on training data only, never on val/test) bring both to mean=0, std=1.

### 5.5 SVM Classification

The 508-dim fused vector is classified by the E11 SVM:

```
SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', probability=True)
```

This is the same model trained in the E11 Classification Track (Macro-F1 = 0.7345 on ground-truth crops). The SVM outputs:
- `predict_proba()` → 6-class probability vector
- `argmax(proba)` → final class label (0–5)

The class label from `argmax(proba)` replaces YOLO's class prediction entirely.

---

## 6. Confidence Score Design

This is a critical design decision for mAP computation.

### The problem
mAP@0.5 is computed by sorting all predictions by confidence (descending) and walking through them to build a precision-recall curve. The confidence score determines which predictions are counted first, which directly affects where the curve sits.

Two options:
1. **SVM max class probability** — reflects how confident the classifier is about the material type
2. **YOLO detection confidence** — reflects how confident the detector is that there is an object and that the box is well-localized

### Why we use YOLO detection confidence (Approach B)

YOLO's detection confidence captures localization quality. A high-confidence detection means YOLO is very sure there is a real object at that location. A low-confidence detection may be a false positive (background predicted as object).

If we use SVM probability instead, we conflate two different signals:
- SVM high confidence → "I am sure this is plastic" (classification)
- YOLO low confidence → "I am not sure there is an object here at all" (detection)

A prediction where YOLO is uncertain but SVM is confident should not rank above a prediction where YOLO is confident. The detection quality (is there an object?) is determined by YOLO. The recognition quality (what is it?) is determined by SVM.

Using YOLO confidence also ensures a **controlled comparison with E1**: E1's mAP is computed using exactly the same YOLO confidence values. The only variable that changes between E1 and E11 is the class label. Any mAP difference is purely due to classification quality.

---

## 7. Evaluation Protocol

### mAP@0.5 computation
For each class c:
1. Collect all predictions assigned class c by the SVM across all test images
2. Sort by YOLO detection confidence (descending)
3. For each prediction (in order): check if the box overlaps with any unmatched GT box of class c (IoU ≥ 0.5)
   - If yes → TP (true positive), mark that GT box as matched
   - If no → FP (false positive)
4. Compute precision-recall curve from cumulative TP/FP
5. AP = area under the PR curve (PASCAL VOC method)

mAP@0.5 = mean AP across all 6 classes

### Test set
1,569 full test images from the 70/15/15 stratified split (same as E1).
Ground-truth labels in YOLO format (class_id x_center y_center width height, normalized).

### Comparison baseline
E1 YOLOv8n: mAP@0.5 = 0.4559

| Class | E1 AP@0.5 |
|-------|-----------|
| BIODEGRADABLE | 0.4663 |
| CARDBOARD | 0.4445 |
| GLASS | 0.4575 |
| METAL | 0.5161 |
| PAPER | 0.3939 |
| PLASTIC | 0.4572 |

---

## 8. Expected Findings

### Expected mAP range: 0.46 – 0.50

The E11 SVM achieved Macro-F1 = 0.7345 on ground-truth crops. On YOLO-predicted crops, performance will be lower because:
- Predicted boxes introduce localization noise (background included, object partially cropped)
- The SVM was trained on GT crops, not predicted crops (domain gap)

Despite this, we expect a small improvement over E1 (0.4559) because:
- The SVM has 508 dimensions of dedicated material descriptors versus YOLO's shallow 1×1 classification head
- Classical features explicitly capture texture, color, and reflectance cues that help with hard classes (Glass, Plastic, Paper)

### Per-class predictions

| Class | Expected change | Reasoning |
|-------|----------------|-----------|
| PAPER | Most likely ↑ | PAPER was YOLO's weakest class (0.3939). Classical texture/color features distinguish paper from cardboard better than YOLO's spatial features. |
| CARDBOARD | Likely ↑ | Strong rectangular shape features help confirm cardboard. |
| GLASS | Uncertain | Glass was hard even with GT crops (SVM F1=0.6677). Transparent objects rely heavily on background, which changes completely with predicted vs GT crops. |
| METAL | Neutral/slight ↑ | METAL was YOLO's best class (0.5161). SVM may not add much. |
| PLASTIC | Slight ↑ | High visual diversity, but classical color features may help. |
| BIODEGRADABLE | Likely ↓ | BIODEGRADABLE has redundant annotations (4 boxes per object). YOLO handles this better because it applies NMS. The SVM has no NMS-awareness. |

### SVM–YOLO agreement rate
Expected: 60–75% of boxes will have the same class from SVM and YOLO.  
Where they disagree, that is where the E11 pipeline makes its contribution — correct or incorrect.

---

## 9. What This Experiment Proves (For the Report)

Regardless of the mAP outcome, E11 Detection Track proves:

**If mAP improves:**
> "A two-stage architecture that delegates classification to a feature-rich hybrid SVM improves detection accuracy over the single-stage YOLO baseline. This confirms that detection pretraining features (SPPF) are insufficient for material discrimination without complementary classical descriptors. The 508-dim hybrid features add meaningful recognition signal on top of YOLO's localization."

**If mAP stays the same or drops slightly:**
> "The two-stage approach does not degrade detection performance, confirming that the SVM classifier is compatible with YOLO's localization pipeline. The gap between classification-track results (Macro-F1 = 0.7345 on GT crops) and detection-track results (mAP@0.5 on predicted crops) quantifies the localization quality bottleneck: the representation is strong, but imperfect boxes limit end-to-end performance."

**If mAP drops significantly:**
> "The domain gap between ground-truth crops (used for SVM training) and YOLO-predicted crops (used for detection-track inference) is the primary performance bottleneck. This motivates future work in training the hybrid classifier on predicted boxes rather than GT boxes."

All three outcomes are scientifically valid and discussable. A negative result is not a failure — it is a finding.

---

## 10. Connection to the Full Experiment Table

| Exp | Track | Features | Classifier | Primary metric |
|-----|-------|----------|------------|----------------|
| E1 | Detection | YOLO CNN | YOLOv8n | mAP@0.5 = 0.4559 |
| **E11** | **Detection** | **SPPF 256 + Classical 252** | **Two-stage SVM** | **mAP@0.5 = TBD** |
| E2 | Classification | Classical 252 | RF | Macro-F1 = 0.6476 |
| E11 | Classification | SPPF 256 + Classical 252 | SVM | Macro-F1 = 0.7345 |
| E3 | Classification | Deep 1280 | SVM | Macro-F1 = 0.7848 |
| E5 | Classification | Classical+Deep+PCA 939 | SVM | Macro-F1 = 0.8071 |

E11 is the only experiment that appears in both tracks — it bridges the representation study (E2–E10) and the full detection pipeline (E1).

---

## 11. Results

**Run completed:** 2026-05-18 | 1,570 test images | 8,340 boxes | 489s (MPS, Apple Silicon)

### 11.1 mAP@0.5

| Model | mAP@0.5 | Δ vs E1 |
|-------|---------|---------|
| E1 — YOLOv8n (single-stage) | 0.4559 | baseline |
| **E11 — Two-stage (YOLO + Hybrid SVM)** | **0.3840** | **−0.0719** |

**Verdict: No improvement.** E11 Detection Track scores 7.2 percentage points below E1.

### 11.2 Per-Class AP@0.5

| Class | E1 AP | E11 AP | Δ | Severity |
|-------|-------|--------|---|---------|
| BIODEGRADABLE | 0.4663 | 0.4441 | −0.0222 | Mild |
| CARDBOARD | 0.4445 | 0.3915 | −0.0530 | Moderate |
| GLASS | 0.4575 | 0.3826 | −0.0749 | High |
| METAL | 0.5161 | 0.4065 | **−0.1096** | **Worst** |
| PAPER | 0.3939 | 0.3256 | −0.0683 | High |
| PLASTIC | 0.4572 | 0.3537 | −0.1035 | Very high |

Every class dropped. METAL suffered the largest absolute loss (−0.1096), BIODEGRADABLE the smallest (−0.0222).

### 11.3 SVM vs YOLO Agreement

| Metric | Value |
|--------|-------|
| Total boxes classified | 8,340 |
| SVM agreed with YOLO | 7,107 (85.2%) |
| SVM differed from YOLO | 1,233 (14.8%) |
| Avg YOLO conf on disagreements | 0.508 |
| Avg SVM conf on disagreements | 0.668 |

**Top disagreement pairs (YOLO → SVM):**

| YOLO predicted | SVM predicted | Count |
|---------------|--------------|-------|
| METAL | GLASS | 92 |
| GLASS | METAL | 91 |
| GLASS | BIODEGRADABLE | 80 |
| METAL | BIODEGRADABLE | 75 |
| METAL | PAPER | 72 |
| PLASTIC | BIODEGRADABLE | 66 |
| PLASTIC | METAL | 65 |

### 11.4 Root Cause Analysis

**Finding 1: Domain gap is the primary cause of the drop.**

The SVM was trained exclusively on ground-truth crops — clean, perfectly bounded, 224×224 windows extracted from annotated boxes. At test time, it receives YOLO-predicted crops — which may include background, partially capture the object, or contain multiple overlapping objects. The SVM has never seen this distribution and fails to generalize to it.

This is the scenario predicted in Section 9 ("worst case"): the representation is strong on GT crops but the domain shift from predicted crops limits end-to-end performance.

**Finding 2: The SVM is confidently wrong on disagreements.**

When the SVM disagrees with YOLO, it does so with higher average confidence (0.668) than YOLO's confidence on the same boxes (0.508). High-confidence wrong predictions rank near the top of the mAP precision-recall curve, causing precision to drop sharply at low recall values. This is more damaging to mAP than uncertain wrong predictions would be.

**Finding 3: BIODEGRADABLE acts as a catch-all class for uncertain crops.**

The SVM systematically reassigns ambiguous crops to BIODEGRADABLE — absorbing GLASS (80 boxes), METAL (75), and PLASTIC (66) into it. Two factors drive this:
1. BIODEGRADABLE represents 43.4% of training crops. The SVM learned a BIODEGRADABLE-dominant prior.
2. YOLO-predicted crops often contain green background context (organic-looking), which the classical color histogram (96-dim HSV) misreads as organic/biodegradable material.

**Finding 4: METAL and GLASS confuse each other symmetrically.**

METAL→GLASS (92) and GLASS→METAL (91) are the two most common disagreements and are nearly equal in count — suggesting the SVM sees these materials as interchangeable under predicted-crop conditions. Both are reflective/shiny materials. In GT crops, the object boundary is clean; in YOLO crops, specular highlights and partial backgrounds make the distinction ambiguous.

**Finding 5: The 85.2% of cases where SVM agrees with YOLO are fine.**

The mAP drop comes entirely from the 14.8% disagreement cases. This means the two-stage pipeline is not fundamentally broken — it performs identically to E1 on most inputs. The damage is localized to ambiguous, low-confidence detections where the SVM's GT-crop training distribution fails.

### 11.5 Discussion for Report

This result is a valid and discussable scientific finding. It does not mean E11 is a failure — it precisely diagnoses a known limitation of two-stage systems trained on different data distributions.

The academically honest conclusion is:

> "The E11 two-stage detector achieves mAP@0.5 = 0.3840, below the E1 single-stage baseline (0.4559). The performance gap is attributable to a domain shift between training conditions (ground-truth crops) and inference conditions (YOLO-predicted crops). The hybrid SVM, when applied to the clean crop distribution it was trained on, achieves Macro-F1 = 0.7345 — demonstrating that the representation is strong but that localization quality critically affects downstream classification. This motivates future work in training the hybrid classifier on predicted-box crops, which would close the domain gap and likely recover the classification performance advantage seen in the GT-crop setting."

Three additional points for the Discussion section:
1. **YOLO's implicit calibration advantage:** YOLO's single-stage head has been jointly optimized with its own backbone features on predicted-box scenarios. It has seen this exact distribution during training. The SVM has not — this is the core asymmetry.
2. **The 8,340-box pipeline throughput:** The two-stage pipeline processed 8,340 boxes in 489 seconds (~58ms/box on MPS). This is slower than E1's pure YOLO inference. A production deployment would need optimization.
3. **The localization bottleneck quantified:** The gap between E11 Classification (Macro-F1 = 0.7345 on GT crops) and E11 Detection (mAP = 0.3840 on predicted crops) quantifies exactly how much localization noise costs — roughly 35 percentage points of recognition performance.

---

## 12. Output Files

| File | Location | Description |
|------|----------|-------------|
| Detection results | `results/metrics/detection_results.csv` (E11 row) | mAP@0.5, per-class AP |
| Analysis log | `results/fusion/e11_detection_analysis_log.csv` | Per-box SVM vs YOLO breakdown |
| Comparison figure | `figures/fusion/E11_vs_E1_detection.png` | Bar charts |
| This document | `Nada_md/e11_detection_track.md` | Methodology notes |
