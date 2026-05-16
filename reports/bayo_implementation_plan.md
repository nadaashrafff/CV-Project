# CLAUDE CODE IMPLEMENTATION PLAN
## Member: Bayo | Experiments: E7, E8, E9, E10
## Project: Compression-Aware Multiscale Feature Fusion for Waste Object Detection and Classification

---

# SECTION 1: PROJECT CONTEXT & BAYO'S ROLE

## 1.1 Project Overview

This project investigates how different visual feature representations can be combined and compressed to improve waste object classification. The project runs 11 experiments (E1-E11) with 4 team members:

| Member | Experiments | Description |
|--------|------------|-------------|
| **Aly** | E1, E2, E3 | YOLO Baseline, Classical Only (252-dim), Deep Only (1280-dim) |
| **Ahmed** | E4, E5, E6 | Raw Early Fusion (2300-dim), PCA Fusion (~512-dim), Autoencoder Fusion (256-dim) |
| **Bayo (YOU)** | **E7, E8, E9, E10** | **Feature Selection (200-dim), Average Vote, Weighted Vote, Attention Fusion (256-dim)** |
| **Nada** | E11, E12, E13 | YOLO Baseline detection, YOLO Neck Fusion, Best Compressed |

## 1.2 What Each of Bayo's Experiments Does

### E7 - Feature Selection Fusion ("Smart Selection")
- **Input**: The full 2,300-dimension fused feature vector (classical 252 + deep early 256 + deep mid 512 + deep late 1280)
- **What it does**: Uses a Random Forest to rank all 2,300 features by importance, keeps only the top 200 most important features
- **Output**: Classification using only 200 selected features
- **Classifier**: Random Forest
- **Key Question**: "Do we really need all 2,300 features, or are the best 200 enough?"

### E8 - Average Voting ("Average Vote")
- **Input**: Probability outputs from the Classical classifier (E2) and Deep classifier (E3)
- **What it does**: Trains classical and deep classifiers separately, then averages their prediction probabilities equally (50/50)
- **Output**: Final prediction = (classical_probs + deep_probs) / 2
- **Key Question**: "If both models vote together equally, does the result improve?"

### E9 - Weighted Voting ("Weighted Vote")
- **Input**: Same probability outputs from Classical (E2) and Deep (E3) classifiers
- **What it does**: Same as E8 but finds the optimal weight combination on the validation set (e.g., 40% classical + 60% deep)
- **Output**: Final prediction = w1 * classical_probs + w2 * deep_probs, where w1 + w2 = 1
- **Key Question**: "Should we trust one model more than the other?"

### E10 - Attention Fusion ("Attention Fusion")
- **Input**: Classical features (252-dim) + EfficientNet late features (1,280-dim)
- **What it does**: Builds a lightweight neural network that learns how much to trust each feature branch for each sample
- **Architecture**:
  1. Project classical 252 -> 256 dimensions (dense layer)
  2. Project deep 1,280 -> 256 dimensions (dense layer)
  3. Concatenate both -> 512 dimensions
  4. Learn attention weights for classical vs deep branches
  5. Produce fused 256-dimensional vector
  6. Classify with final dense layer
- **Output**: Classification + attention weight analysis per class
- **Key Question**: "Can the model learn which feature type to trust for each waste type?"

---

# SECTION 2: CRITICAL DEPENDENCIES - WHAT BAYO MUST WAIT FOR

## 2.1 Files Bayo Needs from Teammates

### From Aly (Experiments E2, E3):
| File | Description | Purpose |
|------|-------------|---------|
| `classical_features_train.pkl` | 252-dim classical features for training crops | E10 input |
| `classical_features_val.pkl` | 252-dim classical features for validation crops | E10 input |
| `classical_features_test.pkl` | 252-dim classical features for test crops | E10 input |
| `deep_features_train.pkl` | 1,280-dim EfficientNet late features for training | E10 input |
| `deep_features_val.pkl` | 1,280-dim EfficientNet late features for validation | E10 input |
| `deep_features_test.pkl` | 1,280-dim EfficientNet late features for test | E10 input |
| `rf_classifier.pkl` | Trained Random Forest model (E2) | E8, E9 probability generation |
| `svm_classifier.pkl` | Trained SVM model (E3) | E8, E9 probability generation |
| `crop_labels_train.pkl` | Class labels for training crops | All experiments |
| `crop_labels_val.pkl` | Class labels for validation crops | All experiments |
| `crop_labels_test.pkl` | Class labels for test crops | All experiments |

### From Ahmed (Experiment E4):
| File | Description | Purpose |
|------|-------------|---------|
| `fused_features_train.pkl` | 2,300-dim fused vector (252+256+512+1280) for training | E7 input |
| `fused_features_val.pkl` | 2,300-dim fused vector for validation | E7 input |
| `fused_features_test.pkl` | 2,300-dim fused vector for test | E7 input |

### From Shared/Team Repository:
| File | Description |
|------|-------------|
| `dataset_split.json` | Which images are in train/val/test split |
| `class_mapping.json` | Original class ID -> Final 6-class mapping |
| `crop_metadata.csv` | List of all crops with their image source and bbox coordinates |

## 2.2 Dependency Graph

```
Aly: E2 (Classical + RF) ──┬──> Bayo E8 (Average Voting) ──┐
        │                    ├──> Bayo E9 (Weighted Voting)  │
        │                    └──> Bayo E10 (Attention Fusion)│
        │                                                    │
Aly: E3 (Deep + SVM) ──────┬───────────────────────────────┘
        │                    │
        └──> classical_features.pkl                         │
             deep_features.pkl                              │
                                                            │
Ahmed: E4 (Raw Fusion) ──> fused_features.pkl ──> Bayo E7 (Feature Selection)
```

## 2.3 What Bayo Can Prepare Now (Before Teammates Finish)

While waiting for teammates, Bayo can:
1. Set up the project structure and GitHub integration
2. Write skeleton code for all 4 experiments
3. Create helper functions for data loading, evaluation, and reporting
4. Write the evaluation/metrics computation code
5. Set up the attention network architecture (E10)

---

# SECTION 3: TECHNICAL IMPLEMENTATION DETAILS

## 3.1 EXACT FEATURE DIMENSIONS

### Fused Feature Vector Breakdown (for E7):
| Feature Source | Dimensions | Indices in Fused Vector |
|---------------|-----------|------------------------|
| Classical features | 252 | 0 - 251 |
| EfficientNet early | 256 | 252 - 507 |
| EfficientNet mid | 512 | 508 - 1019 |
| EfficientNet late | 1,280 | 1020 - 2299 |
| **TOTAL** | **2,300** | 0 - 2299 |

### E10 Input Features:
- Classical: 252 dimensions (indices 0-251 from the fused vector)
- Deep (late only): 1,280 dimensions ( EfficientNet late-layer features)
- Both need separate projection to 256-dim before fusion

## 3.2 EXACT CLASS LABELS (6 classes after merging)

| Class ID | Class Name | Original Class |
|----------|-----------|---------------|
| 0 | cardboard | cardboard |
| 1 | glass | glass |
| 2 | metal | metal |
| 3 | paper | paper |
| 4 | plastic | plastic |
| 5 | trash | trash + biodegradable (merged) |

## 3.3 EVALUATION METRICS BAYO MUST COMPUTE

For ALL Bayo experiments (E7-E10), compute these classification metrics:

| Metric | How to Compute | sklearn Function |
|--------|---------------|-----------------|
| **Accuracy** | Overall correct predictions / total | `accuracy_score(y_true, y_pred)` |
| **Macro-F1** | Average F1 across all 6 classes (treats all classes equally, important for imbalanced data) | `f1_score(y_true, y_pred, average='macro')` |
| **Per-class F1** | F1 score for each individual class | `f1_score(y_true, y_pred, average=None)` |
| **Per-class Precision** | Precision for each class | `precision_score(y_true, y_pred, average=None)` |
| **Per-class Recall** | Recall for each class | `recall_score(y_true, y_pred, average=None)` |
| **Confusion Matrix** | 6x6 matrix showing predictions vs actual | `confusion_matrix(y_true, y_pred)` |
| **Feature Dimension** | Number of features used | Just record the number |
| **Feature Extraction Time** | Time to extract features | `time.time()` before/after |
| **Classification Time** | Time to make predictions | `time.time()` before/after |

---

# SECTION 4: STEP-BY-STEP IMPLEMENTATION

## EXPERIMENT E7: FEATURE SELECTION FUSION

### Goal
Reduce the 2,300-dimension fused vector to only the top 200 most important features using Random Forest feature importance ranking.

### Exact Steps for Claude Code:

```
STEP E7-1: LOAD THE FUSED FEATURES
- Load fused_features_train.pkl (shape: N_train x 2300)
- Load fused_features_val.pkl (shape: N_val x 2300)
- Load fused_features_test.pkl (shape: N_test x 2300)
- Load corresponding labels: crop_labels_train/val/test.pkl

STEP E7-2: TRAIN RANDOM FOREST ON FULL FUSED VECTOR
- Create RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
- Fit on training data: rf.fit(X_fused_train, y_train)
- This trains a forest that also computes feature_importances_

STEP E7-3: RANK FEATURES BY IMPORTANCE
- Extract importances: importances = rf.feature_importances_ (length 2300)
- Get top 200 indices: top_200_indices = np.argsort(importances)[-200:]
  (this gives indices of 200 most important features, sorted by importance)
- Save these indices: np.save('e7_top200_feature_indices.npy', top_200_indices)

STEP E7-4: SELECT TOP 200 FEATURES
- X_train_selected = X_fused_train[:, top_200_indices]  (shape: N_train x 200)
- X_val_selected = X_fused_val[:, top_200_indices]      (shape: N_val x 200)
- X_test_selected = X_fused_test[:, top_200_indices]    (shape: N_test x 200)

STEP E7-5: TRAIN FINAL CLASSIFIER ON SELECTED FEATURES
- Create new RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
- Fit on selected training data: rf_selected.fit(X_train_selected, y_train)
- Save model: joblib.dump(rf_selected, 'e7_feature_selection_rf.pkl')

STEP E7-6: EVALUATE
- Predict on test: y_pred = rf_selected.predict(X_test_selected)
- Predict probabilities: y_proba = rf_selected.predict_proba(X_test_selected)
- Compute all metrics (accuracy, macro-f1, per-class f1, confusion matrix)
- Record feature dimension: 200
- Record training time and inference time

STEP E7-7: SAVE RESULTS
- Save predictions: np.save('e7_predictions.npy', y_pred)
- Save probabilities: np.save('e7_probabilities.npy', y_proba)
- Save metrics to JSON: e7_metrics.json
- Save feature importance plot showing which original feature groups contributed
```

### Key Code Pattern:
```python
# Feature selection using Random Forest importance
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Step 1: Train RF to get importances
rf_importance = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_importance.fit(X_fused_train, y_train)

# Step 2: Get top 200 features
importances = rf_importance.feature_importances_
top_200_idx = np.argsort(importances)[-200:]  # Indices of top 200

# Step 3: Select features
X_train_200 = X_fused_train[:, top_200_idx]
X_test_200 = X_fused_test[:, top_200_idx]

# Step 4: Train final classifier
rf_final = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_final.fit(X_train_200, y_train)

# Step 5: Evaluate
y_pred = rf_final.predict(X_test_200)
```

### E7 Expected Output Files:
| File | Description |
|------|-------------|
| `e7_top200_feature_indices.npy` | Indices of selected 200 features |
| `e7_feature_selection_rf.pkl` | Trained Random Forest on 200 features |
| `e7_predictions.npy` | Predictions on test set |
| `e7_probabilities.npy` | Class probabilities on test set |
| `e7_metrics.json` | All evaluation metrics |
| `e7_feature_group_analysis.png` | Bar chart showing which feature groups were selected |

### E7 Feature Group Analysis:
After selecting top 200, analyze how many came from each group:
- Classical (indices 0-251): count how many selected
- Deep Early (indices 252-507): count how many selected
- Deep Mid (indices 508-1019): count how many selected
- Deep Late (indices 1020-2299): count how many selected
- Plot this as a bar chart to show which feature groups are most informative

---

## EXPERIMENT E8: AVERAGE VOTING

### Goal
Combine predictions from the classical classifier (E2) and deep classifier (E3) by averaging their probability outputs equally.

### Exact Steps for Claude Code:

```
STEP E8-1: LOAD TRAINED CLASSIFIERS
- Load rf_classifier.pkl (Aly's trained Random Forest on 252 classical features)
- Load svm_classifier.pkl (Aly's trained SVM on 1280 deep late features)
- Note: SVM must have probability=True to output probabilities

STEP E8-2: LOAD FEATURES
- Load classical_features_train/val/test.pkl (252-dim)
- Load deep_features_train/val/test.pkl (1280-dim - late features only)
- Load labels: crop_labels_train/val/test.pkl

STEP E8-3: GENERATE PROBABILITY PREDICTIONS FROM BOTH CLASSIFIERS
- On TRAINING set:
  - rf_proba_train = rf.predict_proba(classical_features_train)  (shape: N_train x 6)
  - svm_proba_train = svm.predict_proba(deep_features_train)     (shape: N_train x 6)
- On VALIDATION set:
  - rf_proba_val = rf.predict_proba(classical_features_val)
  - svm_proba_val = svm.predict_proba(deep_features_val)
- On TEST set:
  - rf_proba_test = rf.predict_proba(classical_features_test)
  - svm_proba_test = svm.predict_proba(deep_features_test)

STEP E8-4: AVERAGE PROBABILITIES (EQUAL WEIGHTS)
- avg_proba_train = (rf_proba_train + svm_proba_train) / 2
- avg_proba_val = (rf_proba_val + svm_proba_val) / 2
- avg_proba_test = (rf_proba_test + svm_proba_test) / 2

STEP E8-5: GET FINAL PREDICTIONS
- y_pred_train = np.argmax(avg_proba_train, axis=1)
- y_pred_val = np.argmax(avg_proba_val, axis=1)
- y_pred_test = np.argmax(avg_proba_test, axis=1)

STEP E8-6: EVALUATE
- Compute all metrics on test set
- Also evaluate on validation set to compare

STEP E8-7: SAVE RESULTS
```

### Key Code Pattern:
```python
# Average Voting - Equal weights
# Get probabilities from both classifiers
rf_probs = rf_classifier.predict_proba(X_classical_test)   # (N, 6)
svm_probs = svm_classifier.predict_proba(X_deep_test)       # (N, 6)

# Average equally (50/50)
fused_probs = (rf_probs + svm_probs) / 2

# Final prediction
y_pred = np.argmax(fused_probs, axis=1)
```

### E8 Expected Output Files:
| File | Description |
|------|-------------|
| `e8_avg_voting_predictions.npy` | Final predictions |
| `e8_avg_voting_probabilities.npy` | Averaged probabilities |
| `e8_rf_probabilities_test.npy` | Individual RF probabilities (for record) |
| `e8_svm_probabilities_test.npy` | Individual SVM probabilities (for record) |
| `e8_metrics.json` | All evaluation metrics |

### E8 Important Notes:
- NO TRAINING needed for E8 - it only combines pre-trained models
- Both classifiers must output probability distributions, not just class labels
- If SVM was trained without probability=True, it needs to be retrained or use decision_function + softmax
- Record dimension as "Separate" (two separate models)

---

## EXPERIMENT E9: WEIGHTED VOTING

### Goal
Find the optimal weight combination for classical and deep classifier probabilities by searching on the validation set.

### Exact Steps for Claude Code:

```
STEP E9-1: LOAD SAME CLASSIFIERS AND FEATURES AS E8
- Same as E8 Step 1 and 2

STEP E9-2: GENERATE PROBABILITY PREDICTIONS
- Same as E8 Step 3
- rf_proba_val, svm_proba_val, rf_proba_test, svm_proba_test

STEP E9-3: SEARCH FOR OPTIMAL WEIGHTS ON VALIDATION SET
- Try weight pairs (w_rf, w_svm) where w_rf + w_svm = 1
- Search grid: w_rf in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
- For each weight pair:
  - fused_proba = w_rf * rf_proba_val + w_svm * svm_proba_val
  - y_pred = np.argmax(fused_proba, axis=1)
  - Compute accuracy on validation set
- Select weights that give HIGHEST validation accuracy
- Common expected result: w_rf ~ 0.3-0.4, w_svm ~ 0.6-0.7 (deep features usually stronger)

STEP E9-4: APPLY OPTIMAL WEIGHTS TO TEST SET
- fused_proba_test = best_w_rf * rf_proba_test + best_w_svm * svm_proba_test
- y_pred_test = np.argmax(fused_proba_test, axis=1)

STEP E9-5: EVALUATE AND SAVE
- Compute all metrics
- Save the optimal weights found
- Save comparison: E8 (equal) vs E9 (weighted) performance
```

### Key Code Pattern:
```python
# Weighted Voting - Optimized weights
from sklearn.metrics import accuracy_score

# Search for best weights on validation set
best_acc = 0
best_weights = (0.5, 0.5)

for w_rf in np.arange(0.0, 1.01, 0.1):
    w_svm = 1.0 - w_rf
    fused_val = w_rf * rf_proba_val + w_svm * svm_proba_val
    preds_val = np.argmax(fused_val, axis=1)
    acc = accuracy_score(y_val, preds_val)
    if acc > best_acc:
        best_acc = acc
        best_weights = (w_rf, w_svm)

# Apply best weights to test set
w_rf, w_svm = best_weights
fused_test = w_rf * rf_proba_test + w_svm * svm_proba_test
y_pred = np.argmax(fused_test, axis=1)

print(f"Optimal weights: RF={w_rf:.2f}, SVM={w_svm:.2f}")
```

### E9 Expected Output Files:
| File | Description |
|------|-------------|
| `e9_weighted_voting_predictions.npy` | Final predictions |
| `e9_weighted_voting_probabilities.npy` | Weighted probabilities |
| `e9_optimal_weights.json` | Best weights found (e.g., {"rf": 0.3, "svm": 0.7}) |
| `e9_weight_search_results.csv` | Accuracy for each weight combination tested |
| `e9_metrics.json` | All evaluation metrics |
| `e9_weight_search_plot.png` | Line plot showing accuracy vs. RF weight |

### E9 Important Notes:
- Grid search is done ONLY on validation set, NOT on test set
- Test set is used only ONCE with the best weights
- Save the weight search results to show the optimization process
- Typical expected result: deep features get higher weight (0.6-0.7)

---

## EXPERIMENT E10: ATTENTION FUSION

### Goal
Build a lightweight neural network that learns to dynamically weight classical and deep features for each sample, producing a 256-dim fused representation.

### Architecture Details:
```
Input Branch 1: Classical features (252-dim)
  -> Dense(256, activation='relu')     # Project to 256
  -> BatchNormalization()
  -> Dense(128, activation='relu')     # Further processing
  -> Output: 128-dim classical embedding

Input Branch 2: Deep features (1,280-dim)
  -> Dense(512, activation='relu')     # Reduce dimensionality
  -> BatchNormalization()
  -> Dense(256, activation='relu')     # Project to 256
  -> BatchNormalization()
  -> Dense(128, activation='relu')     # Further processing
  -> Output: 128-dim deep embedding

Fusion Layer:
  -> Concatenate classical(128) + deep(128) = 256-dim
  -> Dense(128, activation='relu')
  -> Dropout(0.3)
  -> Dense(64, activation='relu')

Attention Mechanism:
  -> Dense(2, activation='softmax')    # 2 attention weights (classical, deep)
  -> Multiply embeddings by attention weights
  -> Sum to get fused 128-dim vector

Classification Head:
  -> Dense(64, activation='relu')
  -> Dropout(0.3)
  -> Dense(6, activation='softmax')    # 6 waste classes
```

### Alternative Simpler Architecture (if above is too complex):
```
Input 1: Classical (252) -> Dense(256, relu) -> classical_embed (256)
Input 2: Deep (1280) -> Dense(256, relu) -> deep_embed (256)

Concatenate -> Dense(256, relu) -> Dense(128, relu) -> Dense(6, softmax)
```
This simpler version still projects both to 256 and learns their combination.

### Exact Steps for Claude Code:

```
STEP E10-1: LOAD FEATURES
- Load classical_features_train/val/test.pkl (252-dim)
- Load deep_features_train/val/test.pkl (1280-dim late features)
- Load labels and convert to one-hot encoding for training

STEP E10-2: PREPROCESS
- Normalize features using StandardScaler fitted on training data
- Apply to train, val, test
- Convert labels to categorical one-hot: tf.keras.utils.to_categorical(y, 6)

STEP E10-3: BUILD MODEL
- Use TensorFlow/Keras or PyTorch
- Build dual-input model architecture as described above
- Compile with categorical_crossentropy loss, Adam optimizer, metrics=['accuracy']

STEP E10-4: TRAIN
- Use early stopping (patience=10) monitoring validation loss
- Use ModelCheckpoint to save best model
- Train for up to 100 epochs, batch_size=32
- Callbacks: EarlyStopping + ReduceLROnPlateau

STEP E10-5: EVALUATE
- Load best saved model
- Predict on test set
- Convert predictions back to class labels
- Compute all metrics

STEP E10-6: ANALYZE ATTENTION WEIGHTS
- Create a function to extract attention weights for each sample
- Run on test set
- Average attention weights per class
- Create analysis showing which classes rely more on classical vs deep features
- Expected patterns:
  - Cardboard/Paper: higher classical weight (color/texture important)
  - Glass/Metal: higher deep weight (complex visual patterns)
  - Plastic/Trash: mixed weights

STEP E10-7: SAVE RESULTS
```

### Key Code Pattern (TensorFlow/Keras):
```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model

# Input layers
input_classical = Input(shape=(252,), name='classical_input')
input_deep = Input(shape=(1280,), name='deep_input')

# Classical branch
x1 = Dense(256, activation='relu')(input_classical)
x1 = BatchNormalization()(x1)

# Deep branch
x2 = Dense(256, activation='relu')(input_deep)
x2 = BatchNormalization()(x2)

# Concatenate
fused = Concatenate()([x1, x2])

# Fusion layers
fused = Dense(256, activation='relu')(fused)
fused = Dropout(0.3)(fused)
fused = Dense(128, activation='relu')(fused)
fused = Dropout(0.3)(fused)

# Output
output = Dense(6, activation='softmax')(fused)

model = Model(inputs=[input_classical, input_deep], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### E10 Expected Output Files:
| File | Description |
|------|-------------|
| `e10_attention_model.h5` | Saved trained model |
| `e10_predictions.npy` | Predictions on test set |
| `e10_probabilities.npy` | Class probabilities |
| `e10_attention_weights.npy` | Attention weights per sample |
| `e10_attention_by_class.csv` | Average attention weights per class |
| `e10_metrics.json` | All evaluation metrics |
| `e10_training_history.json` | Loss and accuracy curves during training |
| `e10_attention_heatmap.png` | Heatmap showing attention weights per class |
| `e10_training_curves.png` | Training/validation loss and accuracy plots |

### E10 Attention Analysis Code:
```python
# Extract attention weights and analyze by class
def extract_attention_weights(model, X_classical, X_deep):
    """Get attention weights for each sample"""
    # Create a sub-model that outputs attention layer
    attention_layer = model.get_layer('attention_softmax')
    attention_model = Model(inputs=model.input, outputs=attention_layer.output)
    weights = attention_model.predict([X_classical, X_deep])
    return weights  # shape: (N, 2) - [classical_weight, deep_weight]

# Average by class
for class_id in range(6):
    mask = (y_test == class_id)
    avg_weights = attention_weights[mask].mean(axis=0)
    print(f"Class {class_names[class_id]}: Classical={avg_weights[0]:.3f}, Deep={avg_weights[1]:.3f}")
```

---

# SECTION 5: GITHUB INTEGRATION REQUIREMENTS

## 5.1 GitHub Repository Setup

Claude Code MUST set up GitHub integration to pull teammate contributions:

```
REPOSITORY STRUCTURE (Expected):
waste-classification-project/
├── data/
│   ├── raw/                    # Original dataset (don't modify)
│   ├── processed/
│   │   ├── crops/              # Cropped object images
│   │   ├── classical_features/ # 252-dim features (from Aly)
│   │   ├── deep_features/      # 2048-dim features (from Aly)
│   │   ├── fused_features/     # 2300-dim features (from Ahmed)
│   │   ├── labels/             # Class labels
│   │   └── dataset_split.json  # Train/val/test split
├── models/
│   ├── rf_classifier.pkl       # Aly's classical model
│   ├── svm_classifier.pkl      # Aly's deep model
│   ├── e4_raw_fusion/          # Ahmed's models
│   ├── e5_pca_fusion/          # Ahmed's models
│   ├── e6_autoencoder/         # Ahmed's models
│   ├── e7_feature_selection/   # BAYO'S OUTPUT
│   ├── e8_avg_voting/          # BAYO'S OUTPUT
│   ├── e9_weighted_voting/     # BAYO'S OUTPUT
│   └── e10_attention_fusion/   # BAYO'S OUTPUT
├── results/
│   ├── e7_metrics.json
│   ├── e8_metrics.json
│   ├── e9_metrics.json
│   ├── e10_metrics.json
│   └── comparison_table.csv    # Combined results
├── src/
│   ├── feature_extraction/     # Aly's code
│   ├── fusion/                 # Ahmed's + Bayo's code
│   ├── classifiers/            # All classifier code
│   └── evaluation/             # Metrics computation
└── notebooks/
    ├── bayo_experiments.ipynb  # BAYO'S MAIN NOTEBOOK
    └── results_visualization.ipynb
```

## 5.2 Git Commands Claude Code Must Run:

```bash
# Initial setup
git clone <repository_url>
cd waste-classification-project
git checkout -b bayo-experiments

# Before starting work, pull latest from teammates
git pull origin main

# Check what files teammates have uploaded
git log --oneline --all
git ls-tree -r main --name-only

# After completing work
git add models/e7_feature_selection/ models/e8_avg_voting/ models/e9_weighted_voting/ models/e10_attention_fusion/
git add results/e7_* results/e8_* results/e9_* results/e10_*
git add notebooks/bayo_experiments.ipynb
git commit -m "Add Bayo experiments E7-E10: Feature Selection, Average Voting, Weighted Voting, Attention Fusion"
git push origin bayo-experiments
```

## 5.3 Sync Strategy:
- Bayo should check GitHub daily for updates from Aly and Ahmed
- If teammate files are not yet uploaded, create PLACEHOLDER functions that load dummy data
- Once files are available, switch to real data
- NEVER modify teammate's files - only read from them

---

# SECTION 6: EVALUATION & REPORTING

## 6.1 Unified Results Format

Bayo must produce a results file that matches the team's unified format:

```json
{
  "experiment_id": "E7",
  "experiment_name": "Feature Selection Fusion",
  "member": "Bayo",
  "feature_dimension": 200,
  "metrics": {
    "accuracy": 0.0,
    "macro_f1": 0.0,
    "per_class_f1": {
      "cardboard": 0.0,
      "glass": 0.0,
      "metal": 0.0,
      "paper": 0.0,
      "plastic": 0.0,
      "trash": 0.0
    },
    "confusion_matrix": [[...]],
    "feature_extraction_time_ms": 0.0,
    "classification_time_ms": 0.0
  },
  "model_size_mb": 0.0,
  "timestamp": "2026-05-15T00:00:00"
}
```

## 6.2 Comparison Table Bayo Must Generate

| Exp. | Method | Feature Dim. | Accuracy | Macro-F1 | Time (ms) |
|------|--------|-------------|----------|----------|-----------|
| E7 | Feature Selection | 200 | - | - | - |
| E8 | Average Voting | Separate | - | - | - |
| E9 | Weighted Voting | Separate | - | - | - |
| E10 | Attention Fusion | 256 | - | - | - |

## 6.3 Expected Results (from project description)

| Exp. | Expected Accuracy Range | Notes |
|------|------------------------|-------|
| E7 | ~75-85% | Top 200 features should preserve most performance |
| E8 | ~78-88% | Should improve over individual classifiers |
| E9 | ~80-90% | Should be slightly better than E8 with optimal weights |
| E10 | ~82-92% | Most advanced method, should perform best among Bayo's experiments |

---

# SECTION 7: IMPLEMENTATION ORDER FOR BAYO

## Phase 1: Setup (Can do immediately)
1. Create GitHub branch `bayo-experiments`
2. Set up project directory structure
3. Install dependencies: scikit-learn, tensorflow (or pytorch), numpy, pandas, matplotlib, seaborn, joblib
4. Create evaluation helper functions
5. Create data loading helper functions

## Phase 2: Wait for Teammates
- Check GitHub daily for Aly's feature files and models
- Check GitHub daily for Ahmed's fused feature files
- If delayed, create synthetic dummy data for development

## Phase 3: Implement E8 First (Simplest - No Training)
- Why first? E8 only combines existing models, no training needed
- Load RF and SVM models, generate probabilities, average them
- This validates that Bayo can load teammate's files correctly

## Phase 4: Implement E9 (Builds on E8)
- Same code as E8 but add weight optimization grid search
- Test different weight combinations on validation set

## Phase 5: Implement E7 (Requires Ahmed's Fused Features)
- Load 2,300-dim fused features
- Train RF for feature importance
- Select top 200, train final classifier
- Analyze which feature groups were selected

## Phase 6: Implement E10 (Most Complex - Neural Network)
- Build dual-input neural network
- Train with early stopping
- Extract and analyze attention weights
- Generate per-class attention heatmap

## Phase 7: Generate Final Results
- Run all 4 experiments on test set
- Generate comparison table
- Create visualization plots
- Push results to GitHub

---

# SECTION 8: COMPLETE CODE TEMPLATE

## 8.1 Main Notebook Structure (bayo_experiments.ipynb)

```python
# CELL 1: IMPORTS
import numpy as np
import pandas as pd
import json
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# CELL 2: CONFIGURATION
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
N_CLASSES = 6

# Paths (adjust based on teammate uploads)
DATA_DIR = '../data/processed'
MODELS_DIR = '../models'
RESULTS_DIR = '../results'

# CELL 3: HELPER FUNCTIONS
def load_pickle(filepath):
    """Load pickle file from data directory"""
    return joblib.load(filepath)

def compute_metrics(y_true, y_pred, feature_dim, extraction_time=0, classification_time=0):
    """Compute all required metrics"""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'per_class_f1': {name: float(f) for name, f in zip(CLASS_NAMES, f1_score(y_true, y_pred, average=None))},
        'per_class_precision': {name: float(p) for name, p in zip(CLASS_NAMES, precision_score(y_true, y_pred, average=None))},
        'per_class_recall': {name: float(r) for name, r in zip(CLASS_NAMES, recall_score(y_true, y_pred, average=None))},
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'feature_dimension': feature_dim,
        'feature_extraction_time_ms': extraction_time,
        'classification_time_ms': classification_time
    }
    return metrics

def save_results(exp_id, exp_name, predictions, probabilities, metrics, output_dir=RESULTS_DIR):
    """Save all results in standardized format"""
    np.save(f'{output_dir}/{exp_id}_predictions.npy', predictions)
    np.save(f'{output_dir}/{exp_id}_probabilities.npy', probabilities)
    with open(f'{output_dir}/{exp_id}_metrics.json', 'w') as f:
        json.dump({
            'experiment_id': exp_id,
            'experiment_name': exp_name,
            'member': 'Bayo',
            **metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }, f, indent=2)
    print(f"Saved results for {exp_id}: {exp_name}")

# CELL 4: LOAD DEPENDENCIES (Wait for teammates)
# Load classical features
try:
    classical_train = load_pickle(f'{DATA_DIR}/classical_features_train.pkl')
    classical_val = load_pickle(f'{DATA_DIR}/classical_features_val.pkl')
    classical_test = load_pickle(f'{DATA_DIR}/classical_features_test.pkl')
    print(f"Loaded classical features: {classical_train.shape}")
except FileNotFoundError:
    print("WARNING: Classical features not yet uploaded by Aly")

# Load deep features
try:
    deep_train = load_pickle(f'{DATA_DIR}/deep_features_train.pkl')
    deep_val = load_pickle(f'{DATA_DIR}/deep_features_val.pkl')
    deep_test = load_pickle(f'{DATA_DIR}/deep_features_test.pkl')
    print(f"Loaded deep features: {deep_train.shape}")
except FileNotFoundError:
    print("WARNING: Deep features not yet uploaded by Aly")

# Load labels
try:
    y_train = load_pickle(f'{DATA_DIR}/crop_labels_train.pkl')
    y_val = load_pickle(f'{DATA_DIR}/crop_labels_val.pkl')
    y_test = load_pickle(f'{DATA_DIR}/crop_labels_test.pkl')
    print(f"Loaded labels: {len(y_train)} train, {len(y_val)} val, {len(y_test)} test")
except FileNotFoundError:
    print("WARNING: Labels not yet uploaded")

# Load trained models
try:
    rf_model = load_pickle(f'{MODELS_DIR}/rf_classifier.pkl')
    print("Loaded RF classifier from Aly")
except FileNotFoundError:
    print("WARNING: RF model not yet uploaded by Aly")

try:
    svm_model = load_pickle(f'{MODELS_DIR}/svm_classifier.pkl')
    print("Loaded SVM classifier from Aly")
except FileNotFoundError:
    print("WARNING: SVM model not yet uploaded by Aly")

# Load fused features (from Ahmed)
try:
    fused_train = load_pickle(f'{DATA_DIR}/fused_features_train.pkl')
    fused_val = load_pickle(f'{DATA_DIR}/fused_features_val.pkl')
    fused_test = load_pickle(f'{DATA_DIR}/fused_features_test.pkl')
    print(f"Loaded fused features: {fused_train.shape}")
except FileNotFoundError:
    print("WARNING: Fused features not yet uploaded by Ahmed")

# CELL 5: E7 - FEATURE SELECTION FUSION
def run_e7_feature_selection():
    """E7: Select top 200 features from 2300-dim fused vector using RF importance"""
    print("="*60)
    print("Running E7: Feature Selection Fusion")
    print("="*60)

    t_start = time.time()

    # Step 1: Train RF to get feature importances
    rf_importance = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_importance.fit(fused_train, y_train)

    # Step 2: Rank and select top 200
    importances = rf_importance.feature_importances_
    top_200_idx = np.argsort(importances)[-200:]
    np.save(f'{RESULTS_DIR}/e7_top200_indices.npy', top_200_idx)

    # Step 3: Select features
    X_train_sel = fused_train[:, top_200_idx]
    X_test_sel = fused_test[:, top_200_idx]

    # Step 4: Train final classifier
    rf_final = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    t_train_start = time.time()
    rf_final.fit(X_train_sel, y_train)
    t_train = time.time() - t_train_start

    # Step 5: Predict
    t_pred_start = time.time()
    y_pred = rf_final.predict(X_test_sel)
    y_proba = rf_final.predict_proba(X_test_sel)
    t_pred = time.time() - t_pred_start

    # Step 6: Metrics
    metrics = compute_metrics(y_test, y_pred, feature_dim=200,
                              classification_time=t_pred*1000)

    # Step 7: Feature group analysis
    n_classical = np.sum(top_200_idx < 252)
    n_deep_early = np.sum((top_200_idx >= 252) & (top_200_idx < 508))
    n_deep_mid = np.sum((top_200_idx >= 508) & (top_200_idx < 1020))
    n_deep_late = np.sum(top_200_idx >= 1020)

    print(f"\nFeature Group Analysis (top 200):")
    print(f"  Classical: {n_classical}/200 ({n_classical/200*100:.1f}%)")
    print(f"  Deep Early: {n_deep_early}/200 ({n_deep_early/200*100:.1f}%)")
    print(f"  Deep Mid: {n_deep_mid}/200 ({n_deep_mid/200*100:.1f}%)")
    print(f"  Deep Late: {n_deep_late}/200 ({n_deep_late/200*100:.1f}%)")

    save_results('E7', 'Feature Selection Fusion', y_pred, y_proba, metrics)

    # Save model
    joblib.dump(rf_final, f'{MODELS_DIR}/e7_feature_selection_rf.pkl')

    total_time = time.time() - t_start
    print(f"E7 completed in {total_time:.2f}s")
    print(f"Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")

    return metrics

# CELL 6: E8 - AVERAGE VOTING
def run_e8_average_voting():
    """E8: Average probabilities from RF and SVM equally"""
    print("="*60)
    print("Running E8: Average Voting")
    print("="*60)

    # Generate probabilities from both classifiers
    rf_proba = rf_model.predict_proba(classical_test)
    svm_proba = svm_model.predict_proba(deep_test)

    # Average equally
    avg_proba = (rf_proba + svm_proba) / 2.0
    y_pred = np.argmax(avg_proba, axis=1)

    metrics = compute_metrics(y_test, y_pred, feature_dim='Separate')

    save_results('E8', 'Average Voting', y_pred, avg_proba, metrics)

    # Save individual probabilities for analysis
    np.save(f'{RESULTS_DIR}/e8_rf_probabilities.npy', rf_proba)
    np.save(f'{RESULTS_DIR}/e8_svm_probabilities.npy', svm_proba)

    print(f"E8 Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")
    return metrics

# CELL 7: E9 - WEIGHTED VOTING
def run_e9_weighted_voting():
    """E9: Find optimal weights for RF and SVM on validation set"""
    print("="*60)
    print("Running E9: Weighted Voting")
    print("="*60)

    # Generate validation probabilities
    rf_proba_val = rf_model.predict_proba(classical_val)
    svm_proba_val = svm_model.predict_proba(deep_val)

    # Grid search for best weights on validation set
    best_acc = 0
    best_w = (0.5, 0.5)
    weight_results = []

    for w_rf in np.arange(0.0, 1.01, 0.05):
        w_svm = 1.0 - w_rf
        fused_val = w_rf * rf_proba_val + w_svm * svm_proba_val
        preds_val = np.argmax(fused_val, axis=1)
        acc = accuracy_score(y_val, preds_val)
        weight_results.append({'rf_weight': w_rf, 'svm_weight': w_svm, 'val_accuracy': acc})
        if acc > best_acc:
            best_acc = acc
            best_w = (w_rf, w_svm)

    # Apply best weights to test set
    rf_proba_test = rf_model.predict_proba(classical_test)
    svm_proba_test = svm_model.predict_proba(deep_test)

    w_rf, w_svm = best_w
    fused_test = w_rf * rf_proba_test + w_svm * svm_proba_test
    y_pred = np.argmax(fused_test, axis=1)

    metrics = compute_metrics(y_test, y_pred, feature_dim='Separate')
    metrics['optimal_weights'] = {'rf': float(w_rf), 'svm': float(w_svm)}

    save_results('E9', 'Weighted Voting', y_pred, fused_test, metrics)

    # Save weight search results
    pd.DataFrame(weight_results).to_csv(f'{RESULTS_DIR}/e9_weight_search.csv', index=False)

    print(f"Optimal weights: RF={w_rf:.2f}, SVM={w_svm:.2f}")
    print(f"E9 Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")
    return metrics

# CELL 8: E10 - ATTENTION FUSION
def build_attention_model():
    """Build dual-input attention fusion model"""
    from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization, Multiply, Softmax
    from tensorflow.keras.models import Model

    # Inputs
    input_classical = Input(shape=(252,), name='classical_input')
    input_deep = Input(shape=(1280,), name='deep_input')

    # Classical branch
    c = Dense(256, activation='relu')(input_classical)
    c = BatchNormalization()(c)

    # Deep branch
    d = Dense(256, activation='relu')(input_deep)
    d = BatchNormalization()(d)

    # Concatenate
    fused = Concatenate()([c, d])
    fused = Dense(256, activation='relu')(fused)
    fused = Dropout(0.3)(fused)
    fused = Dense(128, activation='relu')(fused)
    fused = Dropout(0.3)(fused)

    # Output
    output = Dense(6, activation='softmax', name='output')(fused)

    model = Model(inputs=[input_classical, input_deep], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def run_e10_attention_fusion():
    """E10: Train attention-based fusion network"""
    print("="*60)
    print("Running E10: Attention Fusion")
    print("="*60)

    # Prepare data
    y_train_cat = to_categorical(y_train, 6)
    y_val_cat = to_categorical(y_val, 6)

    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler_c = StandardScaler()
    scaler_d = StandardScaler()

    c_train_s = scaler_c.fit_transform(classical_train)
    c_val_s = scaler_c.transform(classical_val)
    c_test_s = scaler_c.transform(classical_test)

    d_train_s = scaler_d.fit_transform(deep_train)
    d_val_s = scaler_d.transform(deep_val)
    d_test_s = scaler_d.transform(deep_test)

    # Build and train model
    model = build_attention_model()
    print(model.summary())

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, monitor='val_loss'),
        tf.keras.callbacks.ModelCheckpoint(f'{MODELS_DIR}/e10_attention_model.keras',
                                           save_best_only=True, monitor='val_accuracy')
    ]

    history = model.fit(
        [c_train_s, d_train_s], y_train_cat,
        validation_data=([c_val_s, d_val_s], y_val_cat),
        epochs=100, batch_size=32, callbacks=callbacks, verbose=1
    )

    # Evaluate
    y_proba = model.predict([c_test_s, d_test_s])
    y_pred = np.argmax(y_proba, axis=1)

    metrics = compute_metrics(y_test, y_pred, feature_dim=256)

    save_results('E10', 'Attention Fusion', y_pred, y_proba, metrics)

    # Save training history
    pd.DataFrame(history.history).to_json(f'{RESULTS_DIR}/e10_training_history.json')

    print(f"E10 Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")
    return metrics

# CELL 9: RUN ALL EXPERIMENTS
results = {}
try:
    results['E7'] = run_e7_feature_selection()
except Exception as e:
    print(f"E7 failed (likely fused features not ready): {e}")

try:
    results['E8'] = run_e8_average_voting()
except Exception as e:
    print(f"E8 failed (likely models not ready): {e}")

try:
    results['E9'] = run_e9_weighted_voting()
except Exception as e:
    print(f"E9 failed (likely models not ready): {e}")

try:
    results['E10'] = run_e10_attention_fusion()
except Exception as e:
    print(f"E10 failed (likely features not ready): {e}")

# CELL 10: GENERATE COMPARISON TABLE
comparison = []
for exp_id, metrics in results.items():
    comparison.append({
        'Experiment': exp_id,
        'Method': {'E7': 'Feature Selection', 'E8': 'Average Voting',
                   'E9': 'Weighted Voting', 'E10': 'Attention Fusion'}[exp_id],
        'Feature Dim': metrics.get('feature_dimension', 'N/A'),
        'Accuracy': f"{metrics['accuracy']:.4f}",
        'Macro-F1': f"{metrics['macro_f1']:.4f}",
        'Extraction Time (ms)': f"{metrics.get('feature_extraction_time_ms', 0):.2f}",
        'Classification Time (ms)': f"{metrics.get('classification_time_ms', 0):.2f}"
    })

comparison_df = pd.DataFrame(comparison)
print("\n" + "="*80)
print("BAYO'S EXPERIMENTS - RESULTS SUMMARY")
print("="*80)
print(comparison_df.to_string(index=False))
comparison_df.to_csv(f'{RESULTS_DIR}/bayo_results_comparison.csv', index=False)
```

---

# SECTION 9: TROUBLESHOOTING GUIDE

## Common Issues and Solutions:

### Issue 1: SVM doesn't have predict_proba
**Solution**: SVM must be trained with `probability=True`. If Aly's model doesn't have this, use:
```python
# Alternative: use decision function + softmax
decisions = svm_model.decision_function(X_test)
# Apply softmax to get probabilities
from scipy.special import softmax
svm_proba = softmax(decisions, axis=1)
```

### Issue 2: Out of Memory with 2,300-dim features
**Solution**: Use incremental/batch processing:
```python
# Process in batches
batch_size = 1000
predictions = []
for i in range(0, len(X_test), batch_size):
    batch = X_test[i:i+batch_size]
    pred = model.predict(batch)
    predictions.extend(pred)
```

### Issue 3: TensorFlow not available
**Solution**: E10 can be implemented with PyTorch instead:
```python
# PyTorch alternative for E10
import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.classical_branch = nn.Sequential(
            nn.Linear(252, 256), nn.ReLU(), nn.BatchNorm1d(256)
        )
        self.deep_branch = nn.Sequential(
            nn.Linear(1280, 256), nn.ReLU(), nn.BatchNorm1d(256)
        )
        self.fusion = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 6), nn.Softmax(dim=1)
        )

    def forward(self, x_classical, x_deep):
        c = self.classical_branch(x_classical)
        d = self.deep_branch(x_deep)
        fused = torch.cat([c, d], dim=1)
        return self.fusion(fused)
```

### Issue 4: Teammate files have different shapes
**Solution**: Add shape validation:
```python
def validate_features(features, expected_dim, name):
    if features.shape[1] != expected_dim:
        raise ValueError(f"{name} expected {expected_dim} dims, got {features.shape[1]}")
    print(f"{name}: {features.shape} - OK")
```

### Issue 5: Class labels don't match
**Solution**: Verify label encoding:
```python
# Ensure labels are 0-5
unique_labels = np.unique(y_train)
assert set(unique_labels) == {0, 1, 2, 3, 4, 5}, f"Unexpected labels: {unique_labels}"
```

---

# SECTION 10: DELIVERABLES CHECKLIST

## Bayo's Final Deliverables:

### Models:
- [ ] `models/e7_feature_selection/e7_feature_selection_rf.pkl`
- [ ] `models/e10_attention_fusion/e10_attention_model.keras` (or .h5)

### Results:
- [ ] `results/e7_metrics.json`
- [ ] `results/e7_top200_indices.npy`
- [ ] `results/e7_feature_group_analysis.png`
- [ ] `results/e8_metrics.json`
- [ ] `results/e8_avg_voting_probabilities.npy`
- [ ] `results/e9_metrics.json`
- [ ] `results/e9_optimal_weights.json`
- [ ] `results/e9_weight_search.csv`
- [ ] `results/e9_weight_search_plot.png`
- [ ] `results/e10_metrics.json`
- [ ] `results/e10_training_history.json`
- [ ] `results/e10_attention_heatmap.png`
- [ ] `results/bayo_results_comparison.csv`

### Code:
- [ ] `notebooks/bayo_experiments.ipynb` (main notebook)
- [ ] `src/bayo/e7_feature_selection.py` (standalone script version)
- [ ] `src/bayo/e8_average_voting.py` (standalone script version)
- [ ] `src/bayo/e9_weighted_voting.py` (standalone script version)
- [ ] `src/bayo/e10_attention_fusion.py` (standalone script version)

### GitHub:
- [ ] All code committed to `bayo-experiments` branch
- [ ] Pull request created to merge into `main`

---

# SECTION 11: SUMMARY FOR CLAUDE CODE

## What Claude Code Must Do:

1. **Set up GitHub integration** - Clone/pull the shared repository, create `bayo-experiments` branch
2. **Check for teammate files daily** - Look for Aly's features/models and Ahmed's fused features
3. **Implement 4 experiments in order**: E8 (simplest) -> E9 -> E7 (needs fused features) -> E10 (most complex)
4. **Generate standardized results** - JSON metrics, comparison table, visualization plots
5. **Push all work to GitHub** - Commit results, models, and code to the team repository

## Critical Reminders:
- E7 REQUIRES Ahmed's fused features (2,300-dim) - cannot start until uploaded
- E8 and E9 REQUIRE Aly's trained models (RF + SVM) - cannot start until uploaded
- E10 REQUIRES Aly's classical features (252-dim) and deep features (1,280-dim)
- Grid search for E9 uses VALIDATION set only, never test set
- All metrics must use the EXACT same test set as teammates
- Random seed must be 42 for reproducibility
- Use Macro-F1 as primary metric (handles class imbalance)

## Contact Points:
- **Aly**: Provides classical features (252-dim), deep features (1,280-dim), RF model, SVM model
- **Ahmed**: Provides fused features (2,300-dim), PCA and autoencoder code reference
- **Nada**: Uses Bayo's results for final comparison table
