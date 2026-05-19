# Early Fusion Experiments (E4–E7) Results Summary

## Overview
This document summarizes the execution and results of the Early Fusion pipeline for the Compression-Aware Multiscale Feature Fusion for Waste Object Detection project. The goal of these experiments was to fuse **Classical features (252-dim)** with **Deep features (1280-dim)**, creating a **1532-dim vector**, and apply various compression techniques to beat the established baselines.

The experiments were run successfully on local hardware after downloading the required feature arrays from the v1.0 GitHub release. Due to the absence of TensorFlow in the local environment, the Autoencoder (E6) was rebuilt using `sklearn.neural_network.MLPRegressor` while preserving the exact mathematical architecture.

---

## Baseline Reference
| Experiment | Architecture | Dimension | Macro-F1 | Accuracy |
|------------|--------------|-----------|----------|----------|
| **E2** | Classical RF | 252 | 0.6476 | 0.7740 |
| **E3** | Deep SVM | 1280 | **0.7848** | 0.8522 |

*The target for E4–E7 was to beat the E3 baseline of **0.7848** Macro-F1.*

---

## E4–E7 Early Fusion Results

| Experiment | Strategy | Classifier | Dimension | Macro-F1 | vs E3 |
|------------|----------|------------|-----------|----------|-------|
| **E4** | Raw Concatenation | Random Forest | 1532 | 0.6960 | -0.0888 |
| **E5** | PCA (95% variance) | SVM (RBF) | 939 | **0.8071** | **+0.0223** |
| **E6** | Autoencoder | SVM (RBF) | 256 | 0.7705 | -0.0143 |
| **E7** | Feature Selection | Random Forest | 200 | 0.7081 | -0.0767 |

### Detailed Analysis

#### 1. E4: Raw Concatenation (Macro-F1: 0.6960)
Directly concatenating classical and deep features and feeding them into a Random Forest performed worse than the deep-only baseline (0.7848). This suggests that the Random Forest struggled with the high dimensionality (1532) and the differing scales/distributions of the two feature sets, introducing noise rather than complementary signal.

#### 2. E5: PCA Fusion (Macro-F1: 0.8071) 🏆
Applying PCA (retaining 95% variance) reduced the dimensionality to 939 while de-correlating the features. Feeding this into an SVM with an RBF kernel **successfully beat the E3 baseline**. The SVM effectively mapped the dense, continuous latent space to highly accurate class boundaries, proving that the classical features *do* contain useful complementary information when properly compressed.

#### 3. E6: Autoencoder Fusion (Macro-F1: 0.7705)
Using an `MLPRegressor` to compress the 1532-dim space into a highly restrictive 256-dim bottleneck resulted in a strong performance (0.7705) but fell just short of the E3 baseline. While highly efficient (compressing by ~6x), some critical variance was lost compared to PCA. 

#### 4. E7: Feature Selection (Macro-F1: 0.7081)
Selecting the top 200 most important features (as ranked by E4's Random Forest) improved upon E4 (0.6960 → 0.7081) while dropping 87% of the data. Interestingly, the selection algorithm chose exactly 100 Classical and 100 Deep features, showing that both domains hold predictive value. However, it still underperformed against SVM-based methods.

---

## Conclusion
The **E5 PCA Fusion strategy (0.8071 Macro-F1)** is the clear winner of the Early Fusion track, successfully proving the hypothesis that combining classical domain knowledge with deep learning representations improves classification performance over deep learning alone.
