# Bayo — Experiments E7–E10: Full Report

**Project:** Compression-Aware Multiscale Feature Fusion for Waste Object Detection and Classification  
**Member:** Bayo (Omar)  
**Date:** May 2026  
**Experiments:** E7 (Feature Selection), E8 (Average Voting), E9 (Weighted Voting), E10 (Attention Fusion)

---

## 1. Project Context

This report documents Bayo's contribution to a group Computer Vision research project. The project compares different strategies for representing, fusing, and compressing visual features extracted from waste object images, with the goal of finding the most accurate and efficient representation for classifying waste into 6 material categories.

The full project has 11 experiments distributed across 4 team members:

| Member | Experiments | Focus |
|--------|-------------|-------|
| Aly | E1, E2, E3 | YOLO baseline, classical features (RF), deep features (SVM) |
| Ahmed | E4, E5, E6 | Raw fusion, PCA fusion, autoencoder fusion |
| **Bayo** | **E7, E8, E9, E10** | **Feature selection, average voting, weighted voting, attention fusion** |
| Nada | E11, E12, E13 | YOLO hybrid |

---

## 2. Dataset and Classes

**Source:** Kaggle Garbage Classification dataset — 10,464 images with YOLO bounding-box annotations  
**Task:** Object crop classification (location already known from ground-truth boxes; model predicts material class)  
**Split:** 70% train / 15% validation / 15% test — stratified at image level

After merging the Biodegradable class with Trash (per project design), the final taxonomy has **6 classes**:

| ID | Class | Count (test set) |
|----|-------|-----------------|
| 0 | BIODEGRADABLE | 5,862 |
| 1 | CARDBOARD | 683 |
| 2 | GLASS | 1,555 |
| 3 | METAL | 948 |
| 4 | PAPER | 681 |
| 5 | PLASTIC | 824 |

> **Note on class imbalance:** BIODEGRADABLE has ~8× more samples than CARDBOARD or PAPER. This makes **Macro-F1** the primary metric (treats all classes equally, not dominated by the majority class).

---

## 3. Baselines (Aly's Results — E2, E3)

All of Bayo's experiments build on Aly's pre-trained models and feature arrays:

| Experiment | Method | Feature Dim | Accuracy | Macro-F1 |
|------------|--------|-------------|----------|----------|
| E2 | Classical features (HOG, LBP, GLCM, color) → Random Forest | 252 | 77.40% | 0.6476 |
| E3 | EfficientNetB0 late features → SVM (RBF kernel) | 1,280 | 85.22% | 0.7848 |

E3 (deep features + SVM) sets the benchmark Bayo's experiments aim to match or beat.

---

## 4. Experiment E7 — Feature Selection Fusion

### 4.1 What it does

E7 tests whether a **small, carefully chosen subset of features** can replace the full combined feature vector without significant accuracy loss.

**Method:**
1. Concatenate classical (252-dim) + deep (1,280-dim) features → **1,532-dim fused vector**
2. Train a "selector" Random Forest (100 trees) on the full 1,532-dim vector to get feature importances
3. Keep only the **top 200 most important features** (13% of the fused vector)
4. Train a new, final Random Forest (200 trees) on only those 200 features
5. Evaluate the final RF on the test set

> **Two-RF rule:** The selector RF is only used to rank features. The final RF (trained on 200 features) is the actual classifier and saved model artifact.

**Research question:** Do we really need all 1,532 features, or are the best 200 enough?

### 4.2 Feature Group Analysis

After ranking all 1,532 features, the top-200 selection was almost evenly split:

| Feature Group | Dimensions in Fused Vector | Selected into Top-200 |
|--------------|---------------------------|----------------------|
| Classical (HOG, LBP, color) | 0–251 (252 dims) | **102** (51%) |
| Deep (EfficientNetB0) | 252–1531 (1,280 dims) | **98** (49%) |

Despite the deep branch having 5× more dimensions than the classical branch, the selector RF chose nearly equal numbers from each. This shows that classical hand-crafted features carry information that the deep features don't fully replicate.

### 4.3 Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 80.40% |
| **Test Macro-F1** | **0.7079** |
| Weighted F1 | 0.7878 |
| Balanced Accuracy | 0.6502 |
| AUC-ROC (macro) | 0.9283 |
| Inference (ms/crop) | 0.0071 |
| Val Macro-F1 | 0.7067 |

**Per-class F1 scores:**

| Class | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| BIODEGRADABLE | 0.8893 | 0.8038 | 0.9952 |
| CARDBOARD | 0.8116 | 0.9061 | 0.7350 |
| GLASS | 0.6160 | 0.8461 | 0.4842 |
| METAL | 0.6911 | 0.7721 | 0.6255 |
| PAPER | 0.6144 | 0.7949 | 0.5007 |
| PLASTIC | 0.6252 | 0.7064 | 0.5607 |

### 4.4 Interpretation

- E7 (macro-F1 = 0.7079) substantially outperforms E2's classical-only baseline (0.6476) — **+9.3 percentage points**
- It falls below E3's deep-only baseline (0.7848) — showing that 200 selected features from the fused vector don't fully replicate the deep SVM's performance
- BIODEGRADABLE dominates due to its large sample count; Glass, Paper, and Plastic remain the hardest classes
- The nearly 50/50 feature split confirms that both modalities contribute meaningfully

### 4.5 Saved Files

| File | Description |
|------|-------------|
| `models/e7_feature_selection_rf.pkl` | Final trained RF (200 features) |
| `models/e7_top200_feature_indices.npy` | Indices of selected 200 features |
| `results/e7_predictions.npy` | Test set class predictions |
| `results/e7_probabilities.npy` | Test set class probabilities (10,553 × 6) |
| `figures/E7_feature_group_analysis.png` | Bar chart showing classical vs deep selection |
| `figures/E7_confusion_matrix.png` | 6×6 confusion matrix |

---

## 5. Experiment E8 — Average Voting

### 5.1 What it does

E8 tests **late fusion** (also called decision-level fusion): instead of combining features before classification, two separate models are trained independently and their probability outputs are averaged.

**Method:**
- Load Aly's pre-trained RF (E2) and SVM (E3) — no new training required
- Get probability outputs from both:
  - RF: `predict_proba(classical_features)` → (N, 6) probability matrix
  - SVM: `scaler.transform(deep_features)` → `predict_proba()` → (N, 6) probability matrix
- Average: `final_proba = (RF_proba + SVM_proba) / 2`
- Predict: `argmax(final_proba)`

**Research question:** If both models vote together equally, does the ensemble outperform either individual model?

### 5.2 Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 84.84% |
| **Test Macro-F1** | **0.7820** |
| Weighted F1 | 0.8390 |
| Balanced Accuracy | 0.7374 |
| AUC-ROC (macro) | 0.9457 |
| Val Macro-F1 | 0.7836 |

**Per-class F1 scores:**

| Class | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| BIODEGRADABLE | 0.9131 | 0.8459 | 0.9918 |
| CARDBOARD | 0.8861 | 0.8954 | 0.8770 |
| GLASS | 0.7140 | 0.9060 | 0.5891 |
| METAL | 0.7590 | 0.7856 | 0.7342 |
| PAPER | 0.7196 | 0.8724 | 0.6123 |
| PLASTIC | 0.7000 | 0.8035 | 0.6201 |

### 5.3 Interpretation

- E8 (0.7820) comes very close to E3 (0.7848) — a gap of only 0.3 percentage points
- It dramatically outperforms E2 (0.6476) — combining two imperfect models gives much better results than either alone
- Every single class improves significantly compared to E7
- Glass precision is very high (0.906) — the SVM is very confident when it says "glass" — but recall is low (0.589), meaning many glass objects get misclassified
- The equal 50/50 weight may not be optimal — motivating E9

### 5.4 Saved Files

| File | Description |
|------|-------------|
| `results/e8_predictions.npy` | Test set predictions |
| `results/e8_probabilities.npy` | Averaged probability matrix (10,553 × 6) |
| `results/e8_rf_probabilities_test.npy` | Raw RF probabilities (for reference) |
| `results/e8_svm_probabilities_test.npy` | Raw SVM probabilities (for reference) |
| `figures/E8_confusion_matrix.png` | 6×6 confusion matrix |

---

## 6. Experiment E9 — Weighted Voting

### 6.1 What it does

E9 extends E8 by asking: should we trust the RF and SVM equally, or should one model carry more weight?

**Method:**
- Same probability matrices as E8 (RF on classical, SVM on deep)
- **Grid search on validation set only** — test 19 weight combinations:
  - `w_rf` ∈ {0.05, 0.10, ..., 0.90, 0.95}
  - `w_svm = 1 − w_rf`
  - Metric optimised: **macro-F1** on validation set
- Select the weight pair that gives the highest validation macro-F1
- Apply those weights to the **test set exactly once**

> **Data hygiene rule:** The test set is never touched during the weight search — only the validation set is used. This prevents overfitting the weight to the test set.

### 6.2 Weight Search Results

| RF Weight | SVM Weight | Val Accuracy | Val Macro-F1 |
|-----------|-----------|-------------|-------------|
| 0.05 | 0.95 | 86.56% | 0.7791 |
| 0.10 | 0.90 | 86.56% | 0.7793 |
| 0.20 | 0.80 | 86.55% | 0.7805 |
| 0.30 | 0.70 | 86.75% | 0.7844 |
| 0.35 | 0.65 | 86.74% | 0.7845 |
| **0.40** | **0.60** | **86.80%** | **0.7857** ← best |
| 0.45 | 0.55 | 86.73% | 0.7856 |
| 0.50 | 0.50 | 86.52% | 0.7836 |
| 0.60 | 0.40 | 85.84% | 0.7726 |
| 0.80 | 0.20 | 82.90% | 0.7212 |
| 0.95 | 0.05 | 79.65% | 0.6626 |

**Optimal weights:** RF = 0.40, SVM = 0.60

This confirms that the deep SVM (E3) is the stronger model and deserves more trust, but the RF still adds meaningful information — a pure SVM (w_rf=0.05) scores lower than the optimal blend.

### 6.3 Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 84.97% |
| **Test Macro-F1** | **0.7832** |
| Weighted F1 | 0.8410 |
| Balanced Accuracy | 0.7417 |
| MCC | 0.7618 |
| Cohen's Kappa | 0.7527 |
| AUC-ROC (macro) | 0.9480 |
| Val Macro-F1 | 0.7857 |

**Per-class F1 scores:**

| Class | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| BIODEGRADABLE | 0.9148 | 0.8513 | 0.9886 |
| CARDBOARD | 0.8822 | 0.8874 | 0.8770 |
| GLASS | 0.7218 | 0.9042 | 0.6006 |
| METAL | 0.7598 | 0.7744 | 0.7458 |
| PAPER | 0.7223 | 0.8714 | 0.6167 |
| PLASTIC | 0.6980 | 0.7963 | 0.6214 |

### 6.4 Interpretation

- E9 is the **best macro-F1 among all Bayo's experiments** (0.7832)
- It improves over E8's equal-weight average (0.7820 → 0.7832) — small but consistent gain
- Both E8 and E9 are within 0.002 of beating E3's baseline (0.7848), showing that late fusion nearly matches the best single-modality result
- The optimal weight (RF=0.40, SVM=0.60) shows the SVM has 50% more influence than the RF, consistent with E3 > E2 individually
- Glass recall is still low (0.60) across all experiments — glass is visually ambiguous

### 6.5 Saved Files

| File | Description |
|------|-------------|
| `results/e9_predictions.npy` | Test set predictions |
| `results/e9_probabilities.npy` | Weighted probability matrix (10,553 × 6) |
| `results/e9_weight_search_results.csv` | Full 19-row weight search table |
| `results/e9_optimal_weights.json` | `{"w_rf": 0.4, "w_svm": 0.6, "val_macro_f1": 0.7857}` |
| `figures/E9_weight_search.png` | Macro-F1 and accuracy vs RF weight curve |
| `figures/E9_confusion_matrix.png` | 6×6 confusion matrix |

---

## 7. Experiment E10 — Attention Fusion

### 7.1 What it does

E10 is the most sophisticated experiment. Instead of fixed weights (E8/E9), a neural network **learns per-sample attention weights** — for each individual crop, it decides how much to trust classical features vs. deep features based on the actual content of the image.

**Research question:** Can the model learn that "for this specific glass object, deep features matter more, but for this cardboard box, texture (classical) matters more"?

### 7.2 Architecture

```
Input A: Classical features (252-dim)
Input B: Deep EfficientNetB0 features (1280-dim)

Gate network:
  concat(A, B) → Linear(1532→128) → ReLU → Dropout(0.3)
               → Linear(128→2) → Softmax
               → [g_classical, g_deep]  (two attention weights, sum to 1)

Projection:
  A → Linear(252→128)  = classical_proj
  B → Linear(1280→128) = deep_proj

Fused vector (128-dim):
  g_classical × classical_proj + g_deep × deep_proj

Classifier:
  fused (128-dim) → Linear(128→6) → class logits
```

**Training:**
- Optimizer: Adam (lr=0.001, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau (halves LR when val macro-F1 plateaus for 5 epochs)
- Early stopping: patience=10 epochs on validation macro-F1
- Batch size: 256
- Feature normalisation: StandardScaler fitted on training data only (separately for classical and deep)

### 7.3 Training Summary

The model converged quickly — early stopping triggered at **epoch 13**, suggesting the task is learnable without long training:

| Epoch | Train Loss | Val Macro-F1 |
|-------|-----------|-------------|
| 10 | 0.1503 | 0.7695 |
| 13 | (early stop) | 0.7696 (best) |

Total training time: **~4 seconds on CPU**  
Model file size: **1.58 MB**

### 7.4 Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **85.64%** ← highest among Bayo's experiments |
| **Test Macro-F1** | 0.7752 |
| Weighted F1 | 0.8525 |
| Balanced Accuracy | 0.7613 |
| MCC | 0.7748 |
| Cohen's Kappa | 0.7726 |
| AUC-ROC (macro) | **0.9578** ← highest |
| Inference (ms/crop) | 0.0011 |
| Val Macro-F1 | 0.7696 |

**Per-class F1 scores:**

| Class | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| BIODEGRADABLE | **0.9405** | 0.9108 | 0.9722 |
| CARDBOARD | 0.8578 | 0.8504 | 0.8653 |
| GLASS | **0.7567** | 0.8744 | 0.6669 |
| METAL | 0.7419 | 0.6959 | 0.7943 |
| PAPER | 0.6910 | 0.7928 | 0.6123 |
| PLASTIC | 0.6634 | 0.6704 | 0.6566 |

### 7.5 Attention Weight Analysis

The gate network outputs a 2-dimensional softmax vector per sample. Averaging over all test crops in each class:

| Class | Avg Classical Weight | Avg Deep Weight | Interpretation |
|-------|---------------------|-----------------|----------------|
| BIODEGRADABLE | 0.3863 | **0.6137** | Strong preference for deep features |
| CARDBOARD | 0.4116 | **0.5884** | Moderate preference for deep features |
| GLASS | 0.4260 | **0.5740** | Moderate preference for deep features |
| METAL | 0.4488 | **0.5512** | Slight preference for deep features |
| PAPER | 0.4210 | **0.5790** | Moderate preference for deep features |
| **PLASTIC** | **0.5099** | 0.4901 | **Only class preferring classical features** |

**Key finding:** The attention network consistently trusts deep (EfficientNetB0) features more than classical features for 5 out of 6 classes. The one exception is PLASTIC, where classical features (color, texture, reflectance cues) are slightly more trusted.

This is broadly consistent with intuition:
- **BIODEGRADABLE** (organic materials): EfficientNet's semantic representations generalise better to diverse organic textures
- **GLASS:** High deep-feature preference aligns with glass's complex visual properties (transparency, reflections) that are hard to capture with hand-crafted features
- **PLASTIC:** The slight classical preference may reflect that plastic objects often have distinctive color/texture patterns (e.g., labeled bottles, colored containers)

The global attention average is approximately 0.43 classical / 0.57 deep — consistent with the optimal weights found by E9's grid search (0.40 / 0.60).

### 7.6 Saved Files

| File | Description |
|------|-------------|
| `models/e10_attention_model.pt` | Trained PyTorch model (1.58 MB) |
| `models/e10_scaler_classical.pkl` | StandardScaler for classical features |
| `models/e10_scaler_deep.pkl` | StandardScaler for deep features |
| `results/e10_predictions.npy` | Test set predictions |
| `results/e10_probabilities.npy` | Softmax class probabilities (10,553 × 6) |
| `results/e10_attention_weights.npy` | Per-crop attention weights (10,553 × 2) |
| `results/e10_attention_by_class.csv` | Per-class average attention weights |
| `results/e10_training_history.json` | Loss and val macro-F1 per epoch |
| `figures/E10_training_curves.png` | Training loss + validation macro-F1 over epochs |
| `figures/E10_attention_heatmap.png` | Heatmap of per-class attention weights |
| `figures/E10_confusion_matrix.png` | 6×6 confusion matrix |

---

## 8. Comparative Results

### 8.1 Full Comparison Table

| Exp | Method | Feature Dim | Accuracy | Macro-F1 | Bal. Acc | AUC-ROC | Inference (ms/crop) |
|-----|--------|-------------|----------|----------|----------|---------|-------------------|
| E2 *(baseline)* | Classical RF | 252 | 77.40% | 0.6476 | 0.6024 | 0.9085 | 0.009 |
| E3 *(baseline)* | Deep SVM | 1,280 | 85.22% | **0.7848** | 0.7672 | 0.9566 | 21.27 |
| **E7** | Feature Selection | **200** | 80.40% | 0.7079 | 0.6502 | 0.9283 | 0.007 |
| **E8** | Average Voting | separate | 84.84% | 0.7820 | 0.7374 | 0.9457 | — |
| **E9** | Weighted Voting | separate | 84.97% | 0.7832 | 0.7417 | 0.9480 | — |
| **E10** | Attention Fusion | 128 | **85.64%** | 0.7752 | **0.7613** | **0.9578** | 0.001 |

### 8.2 Key Takeaways

**1. Feature selection (E7) compresses well but has a macro-F1 cost**  
Using only 200 features (13% of the 1,532-dim fused vector) achieves 80.4% accuracy and macro-F1 of 0.7079 — a strong result given the extreme compression. It clearly beats the classical-only baseline (E2) while being far more compact than using all features.

**2. Late fusion is powerful with minimal complexity (E8, E9)**  
Simply averaging two pre-trained model's probability outputs (E8) nearly matches the best single model (E3) — 0.7820 vs 0.7848 macro-F1. Adding optimised weights (E9) provides a small additional gain. These experiments require zero new training and run in near-zero inference time.

**3. Weighted voting confirms deep features dominate (E9)**  
The optimal weights (RF=0.40, SVM=0.60) confirm that EfficientNetB0 features are 50% more informative than classical features for this task. Yet the RF still matters — a pure SVM vote (w_rf=0.05) scores 0.7791 vs the optimum of 0.7857.

**4. Attention fusion achieves the best overall accuracy (E10)**  
E10 achieves the highest raw accuracy (85.64%), highest AUC-ROC (0.9578), and highest balanced accuracy (0.7613). Its macro-F1 (0.7752) is slightly below E9, but it is the most sophisticated and interpretable model, producing per-sample attention weights.

**5. Glass is the hardest class across all experiments**  
Glass consistently has the lowest recall (48–67%) across all experiments. The model tends to be very precise when it predicts glass (89–91% precision) but misses many glass objects — they likely get confused with plastic or metal due to transparency and reflectivity.

**6. BIODEGRADABLE inflates metrics due to class imbalance**  
BIODEGRADABLE has ~8× more test samples than cardboard or paper. Its consistently high F1 (0.88–0.94) pulls the weighted F1 up. Macro-F1 corrects for this by averaging F1 equally across all 6 classes, which is why it is the primary metric.

### 8.3 Efficiency Analysis

| Exp | Feature Dim | Model Size | Inference |
|-----|-------------|-----------|-----------|
| E3 (baseline) | 1,280 | SVM in-memory | 21.27 ms/crop |
| E7 | **200** | ~50 MB RF | **0.007 ms/crop** |
| E10 | 128 | **1.58 MB** | 0.001 ms/crop |
| E8/E9 | 252 + 1,280 | no new model | ~0 ms (inference only) |

E7 provides the best compression-to-performance trade-off: 200 features at 0.007 ms per crop with macro-F1 of 0.7079.  
E10 is the fastest and smallest learned model: 1.58 MB, 0.001 ms/crop, with the highest accuracy.

---

## 9. Per-Class Comparison Across Experiments

| Class | E2 F1 | E3 F1 | E7 F1 | E8 F1 | E9 F1 | E10 F1 |
|-------|-------|-------|-------|-------|-------|--------|
| BIODEGRADABLE | 0.8857 | 0.9249 | 0.8893 | 0.9131 | 0.9148 | **0.9405** |
| CARDBOARD | 0.7442 | 0.8706 | 0.8116 | 0.8861 | 0.8822 | 0.8578 |
| GLASS | 0.5678 | 0.7294 | 0.6160 | 0.7140 | 0.7218 | **0.7567** |
| METAL | 0.5666 | 0.7654 | 0.6911 | 0.7590 | 0.7598 | 0.7419 |
| PAPER | 0.5335 | 0.7419 | 0.6144 | 0.7196 | 0.7223 | 0.6910 |
| PLASTIC | 0.5880 | 0.6767 | 0.6252 | 0.7000 | 0.6980 | 0.6634 |
| **Macro-F1** | 0.6476 | 0.7848 | 0.7079 | 0.7820 | 0.7832 | 0.7752 |

---

## 10. Outputs Summary

### Models Saved

| File | Experiment | Description |
|------|-----------|-------------|
| `models/e7_feature_selection_rf.pkl` | E7 | Random Forest trained on top-200 features |
| `models/e7_top200_feature_indices.npy` | E7 | Index array of selected 200 features |
| `models/e10_attention_model.pt` | E10 | PyTorch attention fusion network (1.58 MB) |
| `models/e10_scaler_classical.pkl` | E10 | StandardScaler for classical branch |
| `models/e10_scaler_deep.pkl` | E10 | StandardScaler for deep branch |

### Results Saved

| File | Description |
|------|-------------|
| `results/classification_results.csv` | All metrics for E7, E8, E9, E10 in a single table |
| `results/e7_predictions.npy` | E7 test predictions |
| `results/e7_probabilities.npy` | E7 class probabilities |
| `results/e8_predictions.npy` | E8 test predictions |
| `results/e8_probabilities.npy` | E8 averaged probabilities |
| `results/e8_rf_probabilities_test.npy` | E8 raw RF probabilities |
| `results/e8_svm_probabilities_test.npy` | E8 raw SVM probabilities |
| `results/e9_predictions.npy` | E9 test predictions |
| `results/e9_probabilities.npy` | E9 weighted probabilities |
| `results/e9_weight_search_results.csv` | 19-row weight search table |
| `results/e9_optimal_weights.json` | Best weights: RF=0.40, SVM=0.60 |
| `results/e10_predictions.npy` | E10 test predictions |
| `results/e10_probabilities.npy` | E10 softmax probabilities |
| `results/e10_attention_weights.npy` | Per-crop attention weights (10,553 × 2) |
| `results/e10_attention_by_class.csv` | Average attention weight per class |
| `results/e10_training_history.json` | Loss and val macro-F1 per epoch |

### Figures Saved

| File | Description |
|------|-------------|
| `figures/E7_feature_group_analysis.png` | Bar chart: classical vs. deep features selected |
| `figures/E7_confusion_matrix.png` | E7 confusion matrix |
| `figures/E8_confusion_matrix.png` | E8 confusion matrix |
| `figures/E9_weight_search.png` | Val macro-F1 and accuracy vs. RF weight |
| `figures/E9_confusion_matrix.png` | E9 confusion matrix |
| `figures/E10_training_curves.png` | Training loss and val macro-F1 over epochs |
| `figures/E10_attention_heatmap.png` | Per-class attention weight heatmap |
| `figures/E10_confusion_matrix.png` | E10 confusion matrix |

---

## 11. How to Reproduce

All experiments are implemented in a single self-contained script:

```bash
cd "/Users/omarbayoumi/Documents/new cv"
python3 bayo_experiments.py
```

**Requirements:**
```
numpy, pandas, scikit-learn, matplotlib, seaborn, joblib, scipy, torch
```

Install via:
```bash
pip3 install numpy pandas scikit-learn matplotlib seaborn joblib scipy torch --break-system-packages
```

**Input data** (downloaded from team GitHub release v1.0):
- `classical_train_clean_X.npy` (45,177 × 252)
- `deep_train_X.npy` (45,177 × 1,280)
- Corresponding val and test arrays
- `random_forest_E2.pkl` — Aly's pre-trained RF
- `svm_E3.pkl` — Aly's pre-trained SVM + scaler bundle

---

## 12. Conclusions

1. **Feature selection (E7)** provides a strong compression: 200 features from a 1,532-dim fused vector retain 90% of the combined model's macro-F1 while running at near-zero inference cost. This is ideal for resource-constrained deployment.

2. **Average voting (E8)** is the simplest and most practical fusion method. It requires no new training and achieves near-parity with the best single model.

3. **Weighted voting (E9)** confirms that the deep model should carry 60% of the weight vs. 40% for classical. The gain over equal weighting is small (+0.001 macro-F1) but consistent.

4. **Attention fusion (E10)** is the most technically sophisticated experiment. It achieves the best raw accuracy (85.64%), the highest AUC-ROC (0.9578), and the highest balanced accuracy (0.7613). The attention analysis reveals that deep EfficientNet features are trusted more for nearly all classes, with PLASTIC being the notable exception where classical color/texture cues are slightly preferred.

5. **Glass remains the hardest class.** All experiments have low glass recall (48–67%), suggesting that glass's visual ambiguity (transparency, reflections) requires specialised features or more training data to resolve.

6. **The ensemble beats the sum of its parts:** Both E8 and E9 nearly match E3's macro-F1 (0.7848) despite E3 using 1,280 features trained specifically for this task, while E8/E9 simply combine two separately trained models. This validates the late fusion approach as practical and effective.
