# Nada's Experiment Tasks — E11 & Extensions

**Project:** Compression-Aware Multiscale Feature Fusion for Waste Object Detection and Classification  
**Member:** Nada  
**Date:** May 2026  
**Primary Focus:** E11 (YOLO Hybrid) + High-Value Extensions

---

## TABLE OF CONTENTS
1. [Priority 1: E11 — YOLO Hybrid](#1-e11-yolo-hybrid)
2. [Priority 2: E5 Causality Ablation](#2-e5-causality-ablation)
3. [Priority 3: Predicted-Box vs Ground-Truth-Crop](#3-predicted-box-vs-ground-truth-crop)
4. [Priority 4: Localization Perturbation Robustness](#4-localization-perturbation-robustness)
5. [Priority 5: Visual Feature Attribution / Grad-CAM](#5-visual-feature-attribution-grad-cam)
6. [Priority 6: Foreground Masking / GrabCut Ablation](#6-foreground-masking-grabcut-ablation)
7. [Priority 7: Hard-Class Specialist](#7-hard-class-specialist)
8. [Priority 8: Calibration and Confidence Reliability](#8-calibration-and-confidence-reliability)
9. [Priority 9: External Mini-Test Set](#9-external-mini-test-set)
10. [Experiments to Avoid](#10-experiments-to-avoid)
11. [Final Recommended Set](#11-final-recommended-set)

---

## 1. E11 — YOLO HYBRID (Priority 1)

### Status
**Implementation complete.** Notebook and module created.

### What It Tests
Instead of using ground-truth crops, use YOLO detections:

```
Full image
↓
YOLO predicts bounding box
↓
Crop detected object
↓
Extract YOLO features + classical features
↓
Hybrid classifier refines class
↓
Final output: box + class + confidence
```

### Architecture
- **Input:** YOLO SPPF features (256-dim) + Classical features (252-dim) = **508-dim hybrid**
- **Hidden:** Linear(508→256) → ReLU → Dropout(0.3)
- **Hidden:** Linear(256→128) → ReLU → Dropout(0.2)
- **Output:** Linear(128→6) → Softmax

### Why It Is Valuable
This directly answers the doctor's concern. It turns your project from:

> "We classify object crops."

Into:

> "We detect objects in full images and refine recognition using fused visual features."

### What to Compare

| Comparison | Why |
|-----------|-----|
| E1 YOLO baseline vs E11 YOLO Hybrid | Does hybrid fusion improve full detection? |
| Per-class AP before/after hybrid | Which materials benefit? |
| Wrong YOLO class corrected by hybrid | Shows real value |
| Correct YOLO class damaged by hybrid | Shows limitations honestly |

### What Would Prove Viability

**Best possible result:** E11 improves per-class AP for visually ambiguous classes like Glass, Plastic, or Paper.

**Acceptable result:** E11 does not improve mAP overall, but improves classification confidence or corrects selected class confusions.

### Files Created
- `notebooks/08_yolo_hybrid_E11.ipynb` — Complete 13-cell notebook
- `src/models/train_yolo_hybrid.py` — Reusable training module

### Next Step
Run the notebook and compare results with E1–E10 baselines.

---

## 2. E5 CAUSALITY ABLATION (Priority 2)

### Status
**Not yet implemented.** Ahmed's experiment — Nada can assist or verify.

### Why It Is Critical
E5 is currently your best result (0.8071 Macro-F1), but a good doctor may ask:

> "Did E5 win because classical + deep fusion helped, or because PCA + SVM is simply better?"

### Required Ablations

| Experiment | Input | Purpose |
|-----------|-------|---------|
| **E5A** | Deep features only + PCA + SVM | Tests whether PCA alone improves E3 |
| **E5B** | Classical features only + PCA + SVM | Tests classical branch strength under same classifier |
| **E5C** | Classical + deep + PCA + SVM | Your current E5 |
| **E5D** | Classical + deep + SVM without PCA | Tests if PCA is necessary |
| **E5E** | PCA 90%, 95%, 99% variance | Tests compression sensitivity |

### Why It Is Valuable
This proves the actual reason E5 works. Right now, E4 raw concatenation fails at 0.6960, while E5 PCA fusion reaches 0.8071, so compression is clearly important. But you still need to isolate whether the gain comes from:
- fusion,
- PCA,
- SVM,
- or all of them together.

### Best Possible Conclusion
E5 outperforms deep-only PCA, classical-only PCA, and raw fusion, proving that the gain comes from structured classical-deep feature complementarity, not PCA alone.

### Note
This is probably the **most academically important extension after E11**.

---

## 3. PREDICTED-BOX VS GROUND-TRUTH-CROP (Priority 3)

### Status
**Not yet implemented.**

### What It Tests
Currently E2–E10 use ground-truth crops. That is clean for representation analysis, but real systems use predicted boxes.

Test your best classifiers on two crop sources:

| Setting | Crop Source |
|---------|-------------|
| Clean setting | Ground-truth bounding boxes |
| Realistic setting | YOLO-predicted bounding boxes |

### Models to Test
- E3 Deep SVM
- E5 PCA Fusion
- E9 Weighted Voting
- E10 Attention Fusion

### Why It Is Valuable
This tells you whether your crop classifiers are robust to imperfect localization.

A real detector may crop:
- too much background,
- only part of the object,
- multiple objects together,
- shifted boxes.

### What It Proves
**If E5 remains strong on YOLO-predicted crops:** your project becomes much more viable.

**If it drops badly:** that is also useful because it identifies the real bottleneck:

> The representation is strong, but localization quality limits end-to-end performance.

This is a very mature insight.

---

## 4. LOCALIZATION PERTURBATION ROBUSTNESS (Priority 4)

### Status
**Not yet implemented.**

### What It Tests
Take ground-truth boxes and artificially disturb them:

| Perturbation | Meaning |
|-------------|---------|
| Shift box ±5%, ±10%, ±20% | Simulates localization error |
| Expand box by 10–30% | Adds background noise |
| Shrink box by 10–30% | Removes object edges |
| Random jitter | Simulates real detector instability |

Then test E3, E5, E9, and E10.

### Why It Is Valuable
It proves whether your best model is fragile or robust.

### What You May Discover

| Result | Meaning |
|--------|---------|
| E5 drops less than E3 | Fusion is more robust to localization noise |
| E5 drops more than E3 | PCA fusion depends on clean crops |
| Classical features drop badly | Shape/color descriptors need accurate boxes |
| Deep features drop less | CNN features tolerate crop noise better |

This is a genuinely vision-related experiment because bounding-box quality is part of object detection.

---

## 5. VISUAL FEATURE ATTRIBUTION / GRAD-CAM (Priority 5)

### Status
**Not yet implemented.**

### What It Tests
Use Grad-CAM or activation maps for EfficientNet examples.

Show one or two images per class:
- original image,
- bounding box,
- crop,
- Grad-CAM heatmap,
- prediction,
- confidence.

### Why It Is Valuable
It makes the deep visual representation visible.

E3 uses frozen EfficientNetB0 and produces a 1280-dimensional feature vector from the global average pooled backbone output. Without visualization, the doctor may see this as just "feature extraction."

With Grad-CAM, you can show:
- The model focuses on the bottle body, reflective surface, edges, label, or sometimes background.

### Important Warning
Grad-CAM is easier if you run it on a CNN classifier or fine-tuned EfficientNet head. Since E3 uses EfficientNet features + SVM, Grad-CAM is less straightforward.

**So either:**
- Use activation-map visualization instead of strict Grad-CAM, or
- Train a small EfficientNet classification head only for visualization, not as the main result.

**Do not spend too much time here.**

---

## 6. FOREGROUND MASKING / GRABCUT ABLATION (Priority 6)

### Status
**Not yet implemented.**

### What It Tests
Do not train a full segmentation model unless you have masks.

Instead, use bounding boxes to initialize GrabCut or foreground masking.

| Experiment | Input |
|-----------|-------|
| E2 normal | Classical features from normal crop |
| E2-mask | Classical features from foreground-masked crop |
| E5 normal | PCA fusion with normal classical features |
| E5-mask | PCA fusion with masked classical features |

### Why It Is Valuable
It asks a real CV question:

> Does removing background improve material recognition?

### Possible Results

**If performance improves:**
Background noise was hurting feature quality.

**If performance drops:**
Context/background or object boundary information was useful, or the mask removed important material cues.

Both results are discussable.

### Warning
Call it:
> "Foreground masking ablation"

Not:
> "Segmentation system"

Because you do not have segmentation ground truth.

---

## 7. HARD-CLASS SPECIALIST (Priority 7)

### Status
**Not yet implemented.**

### What It Tests
Glass and Plastic are still weak. Your reports repeatedly show Glass/Plastic difficulty.

- E3 says Plastic is the hardest by F1.
- Bayo's report says Glass has low recall across experiments and is visually ambiguous because of transparency/reflectivity.

Add a specialist analysis for hard classes.

### Specialist Classifiers

| Specialist | Purpose |
|-----------|---------|
| Glass vs Plastic | Most visually ambiguous |
| Paper vs Cardboard | Similar texture/color |
| Metal vs Plastic | Reflective materials |

### How It Works
When the main model predicts one of the confusing classes, send it to a small specialist classifier.

```
Main model predicts Glass or Plastic
↓
Specialist Glass-vs-Plastic classifier decides final class
```

### Why It Is Valuable
It shows you are not blindly chasing global accuracy. You are targeting the actual visual failure modes.

### Self-Critical Warning
This can become too much if you implement all pairs. **Start with only: Glass vs Plastic.**

---

## 8. CALIBRATION AND CONFIDENCE RELIABILITY (Priority 8)

### Status
**Not yet implemented.**

### What It Tests
Waste sorting systems need confidence. If the model is unsure, it should say:

> "uncertain / needs manual review."

### Add These Analyses
- reliability diagram,
- expected calibration error,
- confidence threshold analysis,
- selective classification.

### Questions to Test

| Question | Why |
|---------|-----|
| When E5 is 90% confident, is it actually correct 90% of the time? | Calibration |
| What happens if we reject predictions below 0.6 confidence? | Practical safety |
| Which classes are overconfident? | Error diagnosis |

### Why It Is Valuable
It makes the project more practical.

Especially for Glass/Plastic, the model may be confident but wrong.

This is not flashy, but it is research-quality.

---

## 9. EXTERNAL MINI-TEST SET (Priority 9)

### Status
**Not yet implemented.**

### What It Tests
Collect 100–200 new waste images using phone camera or public images.

Test:
- E1 YOLO,
- E3,
- E5,
- E9,
- E10,
- E11 if done.

### Why It Is Valuable
It tests generalization outside the dataset.

### What to Report

| Metric | Meaning |
|--------|---------|
| Accuracy/Macro-F1 on external crops | Recognition generalization |
| YOLO detection examples | Full image viability |
| Failure examples | Domain shift analysis |

### Self-Critical Note
External results may be lower. That is okay.

A mature conclusion would be:

> Performance drops on external images due to lighting/background/domain shift, showing that dataset generalization is the next major limitation.

That is honest and strong.

---

## 10. EXPERIMENTS TO AVOID

### Do Not Add Random YOLOv8s / YOLOv8m Comparisons Unless Needed
A bigger YOLO may improve detection, but it may shift the project into:

> "Which YOLO size is better?"

That is not your research question.

Only add YOLOv8s if you use it as a stronger detection baseline, not as the main contribution.

### Do Not Add Full Segmentation Unless You Have Masks
Without ground-truth masks, segmentation becomes hard to evaluate.

You can add GrabCut/foreground masking, but do not claim a segmentation contribution.

### Do Not Add Too Many Classifiers
Do not add:
- XGBoost,
- KNN,
- logistic regression,
- Naive Bayes,
- many random ML models.

This will make the project look like model shopping.

Your story is already strong:

> representation fusion and compression.

Protect that story.

### Do Not Add Transformer/ViT Unless You Have a Reason
A ViT baseline could be interesting, but it may distract.

If you add it, frame it as:

> "modern deep visual representation baseline"

But it is not necessary.

---

## 11. FINAL RECOMMENDED SET

### If You Want the Project to Look Very Strong

| Priority | Experiment | Value |
|----------|-----------|-------|
| 1 | **E11 YOLO Hybrid** | Makes it end-to-end vision |
| 2 | **E5 ablation** | Proves why best result works |
| 3 | **Predicted-box vs GT-crop** | Tests real-world localization effect |
| 4 | **Localization perturbation** | Tests robustness to box noise |
| 5 | **Visual feature/Grad-CAM analysis** | Makes the vision part visible |
| 6 | **Foreground masking / GrabCut** | Adds light segmentation-style CV |
| 7 | **Glass-vs-Plastic specialist** | Targets hardest visual confusion |
| 8 | **External mini-test set** | Proves generalization |
| 9 | **Calibration analysis** | Proves practical reliability |

### If Time Is Short
**Do only the first four.**

---

## SUMMARY

| Task | Status | Owner | Files |
|------|--------|-------|-------|
| E11 YOLO Hybrid | ✅ Complete | Nada | `08_yolo_hybrid_E11.ipynb`, `train_yolo_hybrid.py` |
| E5 Ablation | ⏳ Pending | Ahmed/Nada | — |
| Predicted-Box vs GT | ⏳ Pending | Nada | — |
| Localization Perturbation | ⏳ Pending | Nada | — |
| Grad-CAM | ⏳ Pending | Nada | — |
| GrabCut Ablation | ⏳ Pending | Nada | — |
| Hard-Class Specialist | ⏳ Pending | Nada | — |
| Calibration Analysis | ⏳ Pending | Nada | — |
| External Test Set | ⏳ Pending | Nada | — |

---

*Generated from project recommendations and team codebase analysis.*
*Date: May 2026*
