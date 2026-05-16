# Feature Arrays Handoff

This folder contains the saved feature vectors needed by downstream experiments.

## Label Encoding

All `*_y.npy` files contain integer class IDs:

| ID | Class |
|---:|---|
| 0 | BIODEGRADABLE |
| 1 | CARDBOARD |
| 2 | GLASS |
| 3 | METAL |
| 4 | PAPER |
| 5 | PLASTIC |

## E2 Classical Features

Manifest: `classical_feature_manifest.json`

| Split | X file | y file | Shape | Notes |
|---|---|---|---|---|
| train | `classical_train_X.npy` | `classical_train_y.npy` | `(225885, 252)` | 5x augmentation: original, hflip, rot90, rot180, rot270 |
| train clean | `classical_train_clean_X.npy` | `classical_train_clean_y.npy` | `(45177, 252)` | no augmentation; row-aligned with `deep_train_X.npy` for fusion |
| val | `classical_val_X.npy` | `classical_val_y.npy` | `(9935, 252)` | no augmentation |
| test | `classical_test_X.npy` | `classical_test_y.npy` | `(10553, 252)` | no augmentation |

## E3 Deep Features

Manifest: `deep_feature_manifest.json`

| Split | X file | y file | Shape | Notes |
|---|---|---|---|---|
| train | `deep_train_X.npy` | `deep_train_y.npy` | `(45177, 1280)` | EfficientNetB0 late pooled features, one augmented pass per crop |
| val | `deep_val_X.npy` | `deep_val_y.npy` | `(9935, 1280)` | no augmentation |
| test | `deep_test_X.npy` | `deep_test_y.npy` | `(10553, 1280)` | no augmentation |

Load with:

```python
import numpy as np

X_classical = np.load("data/processed/features/classical_train_clean_X.npy")
y_classical = np.load("data/processed/features/classical_train_clean_y.npy")

X_deep = np.load("data/processed/features/deep_train_X.npy")
y_deep = np.load("data/processed/features/deep_train_y.npy")
```

Use `classical_train_clean_*` for E4-E10 fusion so classical and deep rows match.
Use `classical_train_*` only when reproducing the standalone E2 augmented Random Forest.
