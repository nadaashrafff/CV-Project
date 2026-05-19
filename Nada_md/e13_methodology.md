# E13: Three-Way Hybrid Detection with Domain Adaptation and Late Fusion

**Owner:** Nada
**Status:** ⏳ Planned — implementation pending
**Date:** May 2026
**Builds on:** E1 (YOLO baseline), E11-DA (domain-adapted two-stage), E12 (three-way classification fusion)

---

## 1. Problem Statement

The detection track produced two prior results, both below the end-to-end YOLO baseline:

| Experiment | Architecture | mAP@0.5 | Δ vs E1 |
|---|---|---|---|
| E1 | End-to-end YOLOv8n | 0.4559 | — |
| E11 | Two-stage, GT-trained SVM on 508-dim YOLO+Classical | 0.3840 | −0.0719 |
| E11-DA | Two-stage, DA-trained SVM on 508-dim YOLO+Classical | 0.3897 | −0.0662 |

E11-DA closed only 8% of E11's gap to E1. The diagnostic conclusion was that the dominant cause of two-stage underperformance is *architectural decoupling* — the end-to-end detector benefits from joint gradient optimization between localization and classification, which a decoupled SVM cannot replicate.

However, this conclusion was reached under two confounding design choices that E11-DA inherited from E11 unchanged:

1. **Weak classifier representation.** E11 and E11-DA used a 508-dimensional fused vector (YOLO SPPF 256 + Classical 252). On clean GT crops this representation achieved Macro-F1 = 0.7345 — the weakest fusion variant in the project. The strongest classification result, E12's three-way PCA fusion (SPPF + EfficientNetB0 + Classical), reached Macro-F1 = 0.8130 on the same GT crops. The detection track has never tested this stronger representation.

2. **YOLO's classification probabilities were discarded.** E11 and E11-DA used YOLO only for box regression and objectness. YOLO's own six-class probability vector was thrown away and entirely replaced by the SVM's class prediction. This severed the joint-optimization signal that E11-DA's diagnostic identified as critical, without testing whether *partial* preservation through probability-level late fusion would help.

E13 controls for both confounds simultaneously.

---

## 2. Research Question

> If the strongest available classification representation (E12) is deployed in a domain-adapted two-stage pipeline, and YOLO's classification probabilities are partially preserved through late fusion rather than discarded, can the resulting system close — or eliminate — the gap to end-to-end YOLOv8n?

Two sub-questions:

1. **Representation effect**: Does upgrading the second-stage classifier from the 508-dim E11 representation to the 1788-dim → PCA E12 representation produce a meaningful mAP improvement under matched training conditions?
2. **Late fusion effect**: Does combining YOLO's class probabilities with the SVM's class probabilities (rather than overriding YOLO's class) further close the gap, and what value of the fusion weight α is optimal?

---

## 3. Method: Three-Way Representation + Domain Adaptation + Late Fusion

### 3.1 High-Level Pipeline

```
TRAINING PHASE (one-time, on 70% train split)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  7,324 training images
       │
       ▼
  YOLOv8n predict()                  [conf=0.25, iou=0.45 — same as E1]
       │
       ├──────────────────────────────────────────────────────┐
       ▼                                                      ▼
  Predicted boxes                                   GT boxes YOLO MISSED
  IoU ≥ 0.50 with any GT?                          (~50% of all GT objects)
    YES → matched_sample (source="predicted")                 │
    NO  → discard                                             ▼
       │                                            Jitter Fill
       │                                            translate ±20% box size
       │                                            scale × U[0.80, 1.25]
       │                                            aspect × U[0.90, 1.10]
       │                                            keep if IoU ∈ [0.50, 0.80]
       │                                                      │
       │                                            jitter_sample (source="jitter")
       │                                                      │
       └────────────────────┬─────────────────────────────────┘
                            ▼
               merged_samples (~52K noisy crops)
                            │
  Each sample (predicted OR jitter — identical code path):
    ├── Crop bbox from full image
    ├── Resize 224×224 (BILINEAR)
    ├── Extract SPPF features      → 256-dim
    ├── Extract EfficientNetB0     → 1280-dim     ← NEW vs E11-DA
    └── Extract Classical          → 252-dim
                            │
                            ▼
       Per-modality StandardScaler (fit on noisy crop distribution)
                            │
                            ▼
              hstack → 1788-dim raw fused vector
                            │
                            ▼
                  PCA(95% variance)               ← NEW vs E11-DA
                            │
                            ▼
              ~1088-dim PCA-projected vector
                            │
                            ▼
        SVM(RBF, C=10, class_weight='balanced')
                            │
                            ▼
                 Domain-Adapted E13 SVM   ←── save


INFERENCE PHASE (test set)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Test image
    │
    ▼ YOLO → predicted boxes + YOLO_class_probabilities + YOLO_detection_conf
    │
  For each box:
    ├── Crop + resize 224×224
    ├── Extract SPPF (256) + EfficientNet (1280) + Classical (252)
    ├── Per-modality scale with DA scalers
    ├── hstack → 1788-dim
    ├── Project through DA PCA → ~1088-dim
    └── E13 SVM predicts class probabilities → p_SVM
    │
    ▼  Late fusion:
        p_fused = α · p_YOLO + (1−α) · p_SVM
        final_class = argmax(p_fused)
        final_conf  = YOLO_detection_conf
    │
  Compute mAP@0.5 (PASCAL VOC)
```

### 3.2 Why Three Modalities Instead of Two

E11-DA used 508 dimensions: SPPF (256) + Classical (252). The Macro-F1 ceiling for this representation on GT crops is 0.7345.

E13 uses 1788 dimensions: SPPF (256) + EfficientNetB0 (1280) + Classical (252). On GT crops, this representation (after PCA) reaches Macro-F1 = 0.8130 — a +0.0785 improvement over the E11 representation.

The EfficientNetB0 branch adds two specific properties the E11 representation lacked:

1. **Hierarchical semantic abstraction.** EfficientNetB0 was pretrained on ImageNet and encodes object-level structure (shape, part composition, material patterns) that SPPF's detection-trained features do not fully capture.
2. **Robustness to crop noise.** EfficientNetB0's pretraining distribution includes many off-center, partial, and contextually noisy images. Its representations are empirically more stable under the kind of imperfect framing that YOLO-predicted crops produce, compared to SPPF features which are tuned to YOLO's own framing distribution.

The expectation is that the EfficientNetB0 branch will degrade *less* under predicted-crop conditions than either SPPF or classical features, providing a more stable backbone for the fused representation.

### 3.3 Why Re-fit PCA on the Predicted-Crop Distribution

E12's PCA was fit on GT-crop features. Using that same PCA on predicted-crop features at inference would introduce a subtle distribution shift even after the StandardScalers are re-fit: the principal axes themselves were chosen to maximize variance in the GT-crop space, not the predicted-crop space.

E13 re-fits both the per-modality StandardScalers and the PCA on the domain-adapted training set (predicted + jittered crops). This ensures the entire feature pipeline — scaling, projection, and classification — is calibrated to the inference distribution. The number of retained components may differ from E12's 1088 because the variance structure of predicted-crop features may differ from GT-crop features.

### 3.4 Why Late Fusion of Probabilities

E11-DA's diagnostic showed that the SVM was *confidently wrong* on its disagreements with YOLO (mean SVM confidence 0.668 vs mean YOLO confidence 0.508 on the same boxes). High-confidence wrong predictions rank near the top of the precision-recall curve and damage mAP disproportionately.

Late fusion addresses this by combining the two probability vectors element-wise before taking the argmax:

$$\mathbf{p}_{\text{fused}} = \alpha \cdot \mathbf{p}_{\text{YOLO}} + (1 - \alpha) \cdot \mathbf{p}_{\text{SVM}}$$

with $\alpha \in [0, 1]$. The final predicted class is $\arg\max_c \mathbf{p}_{\text{fused}}[c]$. The detection confidence used for mAP ranking remains YOLO's objectness score, exactly as in E11 and E11-DA. Only the class label assignment is changed.

Three properties of this formulation matter:

1. **Boundary cases recover prior experiments exactly.** At α = 1, E13 reduces to E1 (YOLO classification only) with mAP = 0.4559. At α = 0, E13 reduces to a pure E13-SVM-override pipeline (analogous to E11-DA but with the stronger representation). The optimal α is therefore guaranteed to produce mAP ≥ max(E1, E13_α=0).
2. **YOLO's class probabilities carry calibration the SVM lacks.** Because YOLO's classification head was trained jointly with its box regressor and objectness head on the predicted-crop distribution itself, its class probabilities are implicitly calibrated against localization quality. When YOLO is uncertain about the box, it tends to spread class probability more uniformly. The SVM has no equivalent signal.
3. **α is selected on validation, applied once on test.** Standard project protocol — same as E9. Grid search α ∈ {0.0, 0.05, 0.10, ..., 1.0}, choose argmax of validation mAP, apply once.

### 3.5 Crop Preprocessing Must Match E11-DA Exactly

The feature extraction pipeline at training and inference must be byte-for-byte identical to E11-DA's preprocessing, with the addition of the EfficientNetB0 branch:

1. OpenCV `cvtColor(BGR → RGB)` on the full image
2. Pixel clamp to image bounds: `x1 = max(0, int(x1))`, etc.
3. Resize with `cv2.INTER_LINEAR` to 224×224
4. **SPPF branch**: `T.ToTensor()` (÷255 only, no ImageNet normalization) → YOLO backbone forward → GAP on layer-9 output → 256-dim
5. **EfficientNetB0 branch**: `T.ToTensor()` → ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) → EfficientNetB0 forward → global pooled output → 1280-dim
6. **Classical branch**: identical to `src/features/classical_features.py` → 252-dim
7. Per-modality StandardScaler (fit only on training split)
8. hstack to 1788-dim
9. PCA(95%) transform (fit only on training split)

The two backbones (YOLO and EfficientNetB0) use different normalization conventions and must be applied to separate copies of the crop. Both crops start from the same resized 224×224 array.

### 3.6 Why the Confidence Score Remains YOLO's

mAP requires a scalar confidence per detection for precision-recall ranking. E13 uses YOLO's detection confidence (objectness × box quality) as this scalar, *not* the max of the fused probability vector. This is the same choice made in E11 and E11-DA and is deliberate for three reasons:

1. **Isolates the variable.** The only difference between E1, E11, E11-DA, and E13 should be the class label assignment. Changing the confidence score introduces a second degree of freedom that would confound interpretation.
2. **YOLO's detection confidence is calibrated against the predicted-crop distribution.** YOLO's objectness was trained on exactly the crops it produces at inference; its calibration is reliable for ranking.
3. **SVM probabilities are known to be miscalibrated under domain shift.** E11-DA's diagnostic showed the SVM tends to be overconfident on hard cases. Using max(p_fused) for ranking would push these overconfident wrong predictions higher in the precision-recall curve.

---

## 4. Experimental Controls

| Variable | Value | Reason |
|---|---|---|
| Random seed | 42 | Locked project-wide |
| Dataset split | Same 70/15/15 as all prior experiments | Row alignment with E11-DA's matched crops |
| YOLO detector | E1 weights (`yolov8n_E1_best.pt`) | Identical to E11, E11-DA — box predictions reused |
| YOLO inference thresholds | conf ≥ 0.25, NMS IoU ≤ 0.45 | Identical to E1, E11, E11-DA |
| Match threshold for training | IoU ≥ 0.50 between predicted and GT box | Same as E11-DA; consistent with mAP@0.5 evaluation |
| Jitter parameters | translate ±20%, scale U[0.80, 1.25], aspect U[0.90, 1.10] | Identical to E11-DA |
| Crop preprocessing | 224×224 BILINEAR, BGR→RGB, per-backbone normalization | Identical to all prior experiments |
| Classifier | SVM(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', probability=True, random_state=42) | Identical to E3, E5, E11, E11-DA, E12 |
| PCA threshold | 95% variance | Identical to E5 and E12 |
| α grid | {0.0, 0.05, 0.10, ..., 1.0} | 21 values, fine enough to detect inflection |
| α selection | Validation mAP@0.5 | Test set used exactly once |
| Primary metric | mAP@0.5 (PASCAL VOC) | Locked project-wide for detection track |

---

## 5. Expected Outcomes

### Outcome A — E13 substantially exceeds E1 (mAP > 0.48)

**Interpretation**: The architectural decoupling diagnosis from E11-DA was incomplete. Two-stage systems can outperform end-to-end detectors when (a) the classifier uses a sufficiently strong representation and (b) the joint-optimization signal is preserved through probability-level fusion rather than discarded. The optimal α is expected to lie in (0, 1) — neither extreme matches the joint configuration.

**Report framing**: "E13 establishes that the two-stage architectural penalty observed in E11 and E11-DA was a consequence of under-resourced design choices rather than a fundamental limitation. When the second stage uses the project's strongest representation and YOLO's classification signal is preserved through late fusion, the two-stage system exceeds end-to-end YOLOv8n by +X mAP. This validates the broader hypothesis that classical, deep semantic, and detection-trained spatial features are partially orthogonal and that their combination, properly fused with the detector's own predictions, produces the strongest deployment result."

### Outcome B — E13 reaches or comes within ±0.01 of E1 (0.445 ≤ mAP ≤ 0.466)

**Interpretation**: The gap is closeable but only by effectively reintroducing YOLO's classification head through the back door. Optimal α likely lies in the range [0.4, 0.7] — substantial reliance on YOLO's classification probabilities. The two-stage system matches end-to-end performance without exceeding it.

**Report framing**: "E13 achieves parity with the end-to-end YOLO baseline, but only when YOLO's class probabilities are weighted at α = X in the final prediction. The result establishes that the two-stage system, given the strongest available representation and probability-level fusion with the detector's own classification head, can match end-to-end joint optimization but does not surpass it. The contribution is the demonstration that decoupled architectures are competitive when properly configured, with the additional advantage of representational interpretability that monolithic detectors lack."

### Outcome C — E13 improves over E11-DA but remains below E1 (0.39 < mAP < 0.45)

**Interpretation**: The architectural decoupling penalty is confirmed as fundamental and not attributable to the confounds E13 controlled for. Even with the strongest possible representation and probability-level fusion, the two-stage system cannot match end-to-end joint optimization. This is the strongest possible negative result the project can produce.

**Report framing**: "E13 represents the strongest two-stage system constructible from the project's components: the E12 three-way fusion representation, trained on the domain-adapted predicted-crop distribution, with YOLO's classification probabilities preserved through validation-tuned late fusion. Despite addressing both confounding factors identified in the E11-DA diagnostic, the system achieves mAP@0.5 = X, still below the end-to-end YOLO baseline of 0.4559. This exhausts the two-stage approach within the project's feature taxonomy and confirms that joint gradient optimization between localization and classification confers an architectural advantage that no combination of feature richness, domain alignment, or probability fusion within a decoupled framework can fully recover."

---

## 6. Connection to Prior Experiments

E13 is the synthesis experiment of the detection track. Every methodological choice originates in a prior result:

| E13 design choice | Origin | What it borrows |
|---|---|---|
| Three-way feature fusion | E12 | The strongest classification representation in the project |
| Per-modality StandardScaler | E12 | Independent scaling prevents one modality from dominating |
| PCA(95%) compression | E5, E12 | Suppresses cross-modal redundancy; enables fusion to outperform single modalities |
| RBF SVM, C=10, balanced | E3, E5, E11, E12 | Held constant for fair representation comparison |
| Training on predicted + jittered crops | E11-DA | Domain alignment between training and inference distributions |
| IoU ≥ 0.50 match threshold | E11-DA | Consistent with mAP@0.5 evaluation protocol |
| Validation-tuned late fusion weight | E9 | Probability-level fusion outperformed fixed-weight ensembling for classification |
| YOLO detection confidence as ranking score | E11, E11-DA | Isolates class assignment as the only variable across detection experiments |

E13 introduces no new methodology of its own. Its contribution is *combinatorial*: it tests whether the project's known-best components, assembled into a single pipeline, produce a deployment result that no individual component achieved.

---

## 7. What E13 Will Prove (Regardless of Outcome)

| Hypothesis | How E13 adjudicates it |
|---|---|
| H1: Two-stage decoupling helps detection | Already falsified by E11. E13 does not retest. |
| H2: The drop in E11 was caused by domain mismatch | Partially falsified by E11-DA (+0.0057). E13 holds domain adaptation constant. |
| H3: The remaining gap is architectural | E13 directly tests this. If E13 ≥ E1, H3 is falsified. If E13 < E1, H3 is strengthened. |
| H4: Stronger features + late fusion can overcome the architectural penalty | E13 is the experiment that tests H4. |

The detection track's narrative arc therefore becomes:

```
E1 (baseline)
  ↓
E11 falsifies H1 (decoupling hurts: −0.0719 mAP)
  ↓
E11-DA mostly falsifies H2 (domain alignment recovers only 8% of the gap)
  ↓
E13 adjudicates H3 vs H4:
  ├── E13 ≥ E1 → H4 wins, H3 falsified, positive result
  ├── E13 ≈ E1 → H4 partially wins, two-stage matches but does not exceed
  └── E13 < E1 → H3 confirmed, two-stage definitively bounded below end-to-end
```

All three terminal branches are scientifically valid and publishable.

---

## 8. Outputs

| File | Description |
|---|---|
| `data/processed/da_checkpoints/da_efficientnet_train_X.npy` | EfficientNetB0 features on DA training crops (new) |
| `data/processed/da_checkpoints/da_efficientnet_test_X.npy` | EfficientNetB0 features on YOLO-predicted test crops |
| `data/processed/da_checkpoints/da_yolo_test_X.npy` | SPPF features on YOLO-predicted test crops |
| `data/processed/da_checkpoints/da_classical_test_X.npy` | Classical features on YOLO-predicted test crops |
| `data/processed/da_checkpoints/yolo_class_probabilities_test.npy` | YOLO 6-class probability vectors per predicted box |
| `models/e13_scalers.pkl` | Three per-modality StandardScalers fit on DA training distribution |
| `models/e13_pca.pkl` | PCA fit on DA training distribution (1788 → ~1088) |
| `models/e13_svm.pkl` | Domain-adapted SVM on PCA-projected three-way features |
| `results/metrics/detection_results.csv` | E13 row(s) appended for α=0, α*, α=1 |
| `results/metrics/E13_alpha_sweep.csv` | mAP@0.5 for each α value on validation and test |
| `results/metrics/E13_per_class_AP.csv` | Per-class AP@0.5 for E13 at optimal α |
| `figures/detection/E13_alpha_sweep.png` | Validation mAP vs α curve, with optimal α marked |
| `figures/detection/E13_vs_all_detection.png` | Bar chart: E1, E11, E11-DA, E13 mAP comparison |
| `figures/detection/E13_per_class_comparison.png` | Per-class AP across all four detection experiments |
| `figures/detection/E13_pca_variance_curve.png` | PCA cumulative explained variance on DA distribution |

---

## 9. Implementation Order

1. **Reuse existing checkpoints where possible.** E11-DA already produced matched + jittered training boxes and the SPPF + Classical features for them. Verify these are still on disk before regenerating.
2. **Extract EfficientNetB0 features on the DA training crops.** This is the only new feature extraction step at training time. Use the same EfficientNetB0 checkpoint and preprocessing as E3/E5/E12.
3. **Fit per-modality StandardScalers and PCA on the DA training set.** Save both.
4. **Train the E13 SVM.** Same hyperparameters as E12. Expect ~15–25 min on CPU given ~52K samples × 1088 PCA components.
5. **Re-run YOLO inference on the test set, saving the full class probability tensor per box.** This is the one change to the inference protocol that E11 and E11-DA did not require.
6. **Extract EfficientNetB0, SPPF, and Classical features on YOLO-predicted test crops.** Project through saved scalers + PCA. Apply E13 SVM. Save `p_SVM` per box.
7. **Run the α grid search on validation.** For each α, compute fused probabilities, run mAP evaluation. Plot the curve.
8. **Apply best α once on test.** Record mAP, per-class AP, agreement rate, and SVM-vs-YOLO disagreement statistics in the same format as E11 and E11-DA's analysis logs.
9. **Generate comparison figures and update `detection_results.csv`.**

---

## 10. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| EfficientNetB0 features on noisy crops do not generalize as expected | Medium | If E13 underperforms E11-DA, this is itself a finding: deep features were not the missing piece. Report and discuss. |
| PCA fit on DA distribution retains too few or too many components | Low | The 95% threshold is self-calibrating. Report whatever number is retained. |
| Optimal α is exactly 1.0 (E1 reproduced) | Low-Medium | This itself is publishable: "the SVM adds no signal even with the strongest representation." It confirms H3 maximally. |
| Class imbalance in DA training set biases SVM (61.3% BIODEGRADABLE) | Known issue from E11-DA | Same mitigation: `class_weight='balanced'` partially compensates. Report imbalance honestly. |
| Inference compute exceeds available resources | Low | EfficientNetB0 inference on ~8K test crops is well within tractable compute. Batch size 64 on CPU is sufficient. |
| The DA-trained E13 SVM has lower Macro-F1 on GT crops than E12 | Expected | Same cross-distribution penalty as E11-DA exhibited. Report alongside detection mAP; do not treat as failure. |

---

## 11. What E13 Is Not

To avoid scope creep, the following are explicitly outside E13's design:

- **Not a retraining of YOLO.** E1's weights are used unchanged. Box predictions are byte-for-byte identical to E1, E11, and E11-DA.
- **Not an end-to-end joint training of the hybrid system.** The SVM remains non-differentiable and decoupled from YOLO. The only "coupling" is at the probability output level via late fusion.
- **Not a per-class learned α.** A scalar global α is used. Per-class α is mentioned in Section 12 as future work.
- **Not a new feature extractor.** All three feature extractors (SPPF, EfficientNetB0, Classical) are reused from prior experiments without modification.
- **Not a re-evaluation of the classification track.** E13 is a detection experiment. The classification track concludes at E12.

---

## 12. Future Work (Out of Scope for This Project)

If E13 produces a positive result (Outcome A or B), the natural extensions for future work are:

1. **Per-class learned α**: train a logistic regression on validation predictions to assign a different α to each of the six classes, exploiting the E10-attention finding that PLASTIC benefits disproportionately from classical features.
2. **Confidence-aware α**: parameterize α as a function of YOLO's detection confidence — trust YOLO more when its box is high-confidence, trust the SVM more when YOLO is uncertain about the box.
3. **End-to-end differentiable hybrid**: replace the SVM with a small neural classifier head trained jointly with YOLO's classification loss, sharing gradients through the EfficientNetB0 and classical branches. This would test whether the architectural penalty diagnosed in E11-DA can be eliminated entirely through joint training of the hybrid system.

These extensions are noted for completeness but are not part of the present project.

---

## 13. Results

**Status:** COMPLETE — run on 2026-05-19, Colab T4

---

### 13.1 Training Configuration

| Parameter | Value |
|---|---|
| DA training samples | ~52,000 (predicted + jitter, same as E11-DA) |
| EfficientNetB0 features extracted on DA crops | ✓ (new vs E11-DA) |
| Fused input dim | 1788 (SPPF 256 + EfficientNet 1280 + Classical 252) |
| PCA components retained (95% variance, DA distribution) | **1098** (from 1788) |
| PCA variance retained | ≥ 95.00% |
| SVM hyperparameters | RBF, C=10, gamma='scale', class_weight='balanced' |
| α grid | {0.00, 0.05, …, 1.00} — 21 values, tuned on val |

---

### 13.2 Alpha Sweep (Validation)

| α | Val mAP@0.5 |
|---|---|
| 0.00 (SVM only) | 0.4599 |
| 0.10 | 0.4647 |
| 0.20 | 0.4643 |
| 0.25 | 0.4656 |
| 0.35 | 0.4682 |
| 0.40 | 0.4690 |
| **0.45 ← best** | **0.4706** |
| 0.50 | 0.4705 |
| 0.55 | 0.4674 |
| 0.60 | 0.4638 |
| 0.70 | 0.4625 |
| 0.80 | 0.4609 |
| 0.90 | 0.4589 |
| 1.00 (YOLO only) | 0.4587 |

The curve peaks at α = 0.45. Both boundary cases are meaningful: at α = 0.0, the pure SVM already outperforms pure YOLO (0.4599 vs 0.4587 on val), which is a direct reversal of the E11-DA dynamic where the SVM-only pipeline sat well below YOLO. At the optimal α = 0.45, the near-equal weighting means neither source dominates — the two classifiers are adding genuinely complementary signal, not one correcting the other.

---

### 13.3 Test Results

#### Overall detection mAP@0.5

| Experiment | Architecture | mAP@0.5 | Δ vs E1 | Δ vs E11-DA |
|---|---|---|---|---|
| E1 | End-to-end YOLOv8n | 0.4559 | — | — |
| E11 | Two-stage, 508-dim, GT-trained | 0.3840 | −0.0719 | — |
| E11-DA | Two-stage, 508-dim, DA-trained | 0.3897 | −0.0662 | — |
| E13 (α=0.0, SVM only) | Two-stage, 1788→1098-dim, DA-trained | 0.4465 | −0.0094 | +0.0568 |
| **E13 (α=0.45)** | **Three-way DA + late fusion** | **0.4590** | **+0.0031** | **+0.0693** |

E13 at α = 0.45 records mAP@0.5 = **0.4590**, placing it +0.0031 above the E1 end-to-end baseline. This is **Outcome B**: the two-stage system achieves parity with end-to-end YOLO, but does not substantially exceed it. The gap is closed, not reversed.

#### Decomposing the gain over E11-DA (+0.0693 total)

The total improvement over E11-DA breaks cleanly into two independent contributions:

- **Representation effect** (α=0.0 vs E11-DA): +0.0568 — upgrading from 508-dim (SPPF+Classical) to 1788-dim (SPPF+EfficientNet+Classical) with DA training, no late fusion
- **Late fusion effect** (α=0.45 vs α=0.0): +0.0125 — adding YOLO's classification probabilities at optimal weighting

The representation upgrade accounts for 82% of the total gain. Late fusion adds the remaining 18%.

#### Per-class AP@0.5

| Class | E1 | E11 | E11-DA | E13 (α=0.45) | Δ vs E1 |
|---|---|---|---|---|---|
| BIODEGRADABLE | 0.4663 | 0.4441 | 0.4426 | 0.4568 | −0.0095 |
| CARDBOARD | 0.4445 | 0.3915 | 0.3956 | 0.4408 | −0.0037 |
| GLASS | 0.4575 | 0.3826 | 0.3837 | 0.4475 | −0.0100 |
| METAL | 0.5161 | 0.4065 | 0.4175 | **0.5354** | **+0.0193** |
| PAPER | 0.3939 | 0.3256 | 0.3414 | **0.4283** | **+0.0344** |
| PLASTIC | 0.4572 | 0.3537 | 0.3577 | 0.4451 | −0.0121 |
| **Mean (mAP)** | **0.4559** | **0.3840** | **0.3897** | **0.4590** | **+0.0031** |

The per-class pattern is not uniform. E13 beats E1 on exactly two classes: **METAL (+0.0193) and PAPER (+0.0344)**. It falls marginally short on the other four, with the largest deficit on PLASTIC (−0.0121) and GLASS (−0.0100). The overall +0.0031 mean is therefore a balance between real gains on two material categories and small losses on four others — not a uniform improvement across the board.

#### YOLO/SVM agreement rate

| Experiment | Agreement Rate |
|---|---|
| E11 (GT-trained) | 85.2% |
| E11-DA (DA-trained, 508-dim) | 85.3% |
| E13 (DA-trained, 1788-dim) | **91.3%** |

The jump from ~85% to 91.3% is the single clearest signal of how much the stronger representation changed the SVM's behaviour. In E11 and E11-DA, 1 in 7 boxes had the SVM override YOLO's class. In E13, the SVM overrides YOLO on fewer than 1 in 11 boxes. The SVM is not more passive — it is simply more often right, and when it is right, it agrees with YOLO. The remaining 8.7% disagreements are where the late fusion earns its keep.

---

### 13.4 Outcome B: What It Means

E13 falls squarely in **Outcome B** as defined in Section 5 of this document: it reaches parity with E1 (within ±0.01) but does not substantially exceed it. The optimal α = 0.45 confirms the second part of Outcome B's interpretation — substantial reliance on YOLO's classification signal (α = 0.45 is meaningfully different from both α = 0 and α = 1, but sits closer to equal weighting than to either extreme). Neither source can be removed without cost.

The claim the paper can now make is: **a carefully configured two-stage system — using the strongest available representation, trained on the domain-adapted distribution, with YOLO's classification signal preserved through late fusion — is competitive with end-to-end joint optimization for waste object detection.** The architectural advantage of end-to-end training is real but not insurmountable: it is closeable through a combination of representation quality (the dominant term, accounting for 82% of the gain) and probability-level coupling (the secondary term, accounting for 18%).

This outcome simultaneously resolves and updates the H3/H4 hypotheses from Section 7:

- **H3 (architectural gap is fundamental) — partially confirmed, partially falsified.** The gap exists (E13 does not substantially exceed E1), but it is not large enough to constitute a fundamental barrier. A +0.0031 difference is within measurement noise and practical equivalence. A stronger claim than "confirmed" cannot be supported.
- **H4 (stronger features + late fusion can overcome the penalty) — substantially confirmed for the "close the gap" version, not confirmed for the "exceed E1" version.** Upgrading from 508-dim to 1788-dim representation recovered 82% of the E11-original gap to E1 on its own. Adding late fusion recovered the remaining 9% and pushed E13 marginally above E1.

---

### 13.5 Per-Class Analysis

The class-level pattern warrants separate discussion because it reveals which categories benefit from the richer representation and which do not.

**METAL and PAPER: meaningful gains above E1**

METAL improves by +0.0193 and PAPER by +0.0344 — the two largest positive deltas in the detection track. Both categories have high-discriminability visual signatures that EfficientNetB0's ImageNet pretraining captures well: metal surfaces have reflectance and edge structure that deep features encode reliably, and paper has strong HOG and LBP signatures reinforced by EfficientNet's texture-encoding layers. In both cases, the three-way representation gives the SVM confident, correct predictions that occasionally override YOLO's incorrect classification — and those overrides pay off in AP.

**PLASTIC: largest deficit below E1 (−0.0121)**

PLASTIC was already the hardest class in the classification track (E3 per-class F1 = 0.6767, the lowest of any class). Its difficulty comes from visual ambiguity: transparent and coloured plastics are frequently misclassified as glass or cardboard in poor lighting. On YOLO-predicted crops, this ambiguity is compounded by the imperfect framing. YOLO's classification head, trained jointly on the full detection distribution, handles this ambiguity better than the SVM's offline decision boundary. The −0.0121 deficit suggests that for PLASTIC, YOLO's joint training confers a per-class advantage the SVM cannot replicate even with the strongest representation.

**BIODEGRADABLE (−0.0095) and GLASS (−0.0100): small symmetric deficits**

Both classes fall just under 0.01 below E1. BIODEGRADABLE's deficit is partially explained by its known annotation issue (redundant multi-box annotations per object), which systematically penalises any two-stage system that generates one detection per annotation rather than one per object. GLASS and BIODEGRADABLE are also the most morphologically variable classes in the dataset — the diverse shapes mean the SVM's decision boundary is harder to calibrate on DA training crops than YOLO's gradient-trained head.

**Implication for per-class α**

The per-class pattern is exactly what motivates per-class α as future work (Section 12). A learned α per class would give METAL and PAPER higher SVM weight (where the SVM adds signal) and PLASTIC and GLASS higher YOLO weight (where YOLO's joint training dominates). Even the coarse scalar α = 0.45 that E13 uses globally is a compromise between these competing pressures — which is partly why it arrives at 0.45 rather than at either extreme.

---

### 13.6 Interpretation of the α = 0.45 Optimum

The optimal α lying near 0.45 rather than near 0 or 1 is the most important single number in the E13 results. It tells a specific story.

At α = 0, the SVM classifies every box alone. Its val mAP is 0.4599 — already marginally above YOLO-alone (0.4587). This means the E13 SVM, on its own, has already closed essentially the full architectural gap that motivated E13 in the first place. The three-way DA representation is strong enough that the pure-SVM two-stage system matches E1.

But the best result is not at α = 0. It is at α = 0.45, adding +0.0107 val mAP over the SVM-alone baseline. This increment comes from the 8.7% of boxes where the two classifiers disagree — specifically, from YOLO being correct on those disagreements more often than the SVM. At α = 0.45, fused probabilities on disagreement cases are pulled toward YOLO's prediction just enough to correct the SVM's errors without losing the cases where the SVM is right and YOLO is wrong.

The α = 1.0 boundary (YOLO only) gives val mAP = 0.4587 — lower than α = 0 — confirming that the SVM adds net positive information even without fusion. The full curve (monotonically increasing from α = 0.5 to α = 1.0 on val, with peak at α = 0.45) has the characteristic shape of two complementary sources: neither is uniformly dominant, and mixing is strictly better than either extreme.

---

### 13.7 What E13 Proves

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1: Two-stage decoupling helps detection | Falsified (by E11) | E13 does not retest. |
| H2: The E11 drop was caused primarily by domain mismatch | Partially falsified (by E11-DA, −8% recovery) | E13 holds domain adaptation constant; not retested. |
| H3: The remaining gap is architectural (fundamental) | **Not confirmed as fundamental** | E13 closes the gap to +0.0031 — too small to constitute a "fundamental" architectural barrier. Rephrased: the gap exists but is within practical equivalence. |
| H4: Stronger features + late fusion can close the gap | **Confirmed for "close the gap"** | E13 closes the gap via 82% representation effect + 18% late fusion effect. Not confirmed for "exceed E1 substantially." |
| Representation effect dominates late fusion effect | **Confirmed** | 0.0568 (representation) vs 0.0125 (fusion) — 82% vs 18% of total gain. |
| EfficientNetB0 adds signal on noisy crops | **Confirmed** | α=0 E13 (0.4465) vs E11-DA (0.3897) — +0.0568 with only representation change. |
| Near-equal α weighting is optimal | **Confirmed** | Best α = 0.45; both extreme α = 0 and α = 1 are suboptimal by >0.01 val mAP. |

---

### 13.8 Limitations

**1. Soft YOLO probability approximation.** Because the ultralytics `predict()` API does not expose the full per-box class distribution after NMS, the YOLO probability vector used in late fusion was constructed as: `p_yolo[yolo_cls] = yolo_conf; p_yolo[others] = (1 - yolo_conf) / 5`. This is a valid approximation (it places the correct mass at the predicted class and distributes residual probability uniformly), but the true YOLO class distribution may be sharpened or peaked differently. A raw-output forward pass to recover true class logits before NMS could produce a slightly different (and possibly better) late fusion result.

**2. Single scalar α.** The global α = 0.45 is a compromise across all six classes. The per-class analysis in §13.5 shows that METAL and PAPER want more SVM weight while PLASTIC and GLASS want more YOLO weight. A scalar α captures the average case but leaves per-class improvement on the table.

**3. YOLO's detection confidence as the mAP ranking score.** E13 inherits E11/E11-DA's design choice of using YOLO's objectness score for ranking, not the max of the fused probability. This isolates classification as the only variable and keeps the comparison fair, but it means that the SVM's confidence — even when the SVM is right and YOLO is wrong — never influences where a detection appears in the precision-recall curve. A hybrid confidence score (e.g., max(p_fused)) might produce higher AP on classes where the SVM is systematically more confident than YOLO.

**4. DA training crop coverage.** As in E11-DA, ~50% of GT objects are missed by YOLO at training time and replaced with jitter-filled synthetic crops. The jitter distribution (IoU 0.50–0.80, translate ±20%, scale 0.80–1.25) is calibrated to resemble real YOLO localization noise but is not identical to it. The EfficientNetB0 features extracted on these jittered crops may have higher noise than features extracted on actual YOLO-predicted crops, which could slightly depress the SVM's performance on the hardest classes.

---

### 13.9 Detection Track Summary

The complete detection track narrative, as established by E1 → E11 → E11-DA → E13:

```
E1 (mAP = 0.4559)
  ↓
E11 (mAP = 0.3840)  — falsifies H1: two-stage decoupling hurts, Δ = −0.0719
  ↓
E11-DA (mAP = 0.3897)  — mostly falsifies H2: domain alignment recovers only 8% of the gap
  ↓
E13, α=0 (mAP = 0.4465)  — representation effect: upgrading to 1788-dim recovers 82% of original gap
  ↓
E13, α=0.45 (mAP = 0.4590)  — late fusion effect: adds final 18%, places E13 above E1 by +0.0031
```

The detection track concludes that the two-stage architectural penalty observed in E11 was driven primarily by the weakness of the second-stage classifier representation, with a secondary contribution from domain mismatch, and a smaller additional contribution from the discarding of YOLO's classification signal. When all three factors are addressed simultaneously — stronger representation (E12's three-way fusion), domain-adapted training (E11-DA's protocol), and probability-level coupling (E9-style late fusion) — the two-stage system reaches practical equivalence with end-to-end YOLOv8n at mAP@0.5 on this dataset.
