"""Early fusion: concatenate classical + deep features, then classify (E4–E7)."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.features.feature_utils import load_split_features, normalize_features
from src.utils.seed import SEED


def concatenate_features(*feature_arrays: np.ndarray) -> np.ndarray:
    return np.concatenate(feature_arrays, axis=1)


def build_fused_dataset(feature_dir: str) -> tuple:
    classical_train, classical_val, classical_test = load_split_features(feature_dir, "classical")
    deep_train, deep_val, deep_test = load_split_features(feature_dir, "deep")

    train = concatenate_features(classical_train, deep_train)  # 2300 dims
    val   = concatenate_features(classical_val, deep_val)
    test  = concatenate_features(classical_test, deep_test)

    return normalize_features(train, val, test)


def train_raw_fusion_classifier(
    X_train: np.ndarray, y_train: np.ndarray
) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                  n_jobs=-1, random_state=SEED)
    clf.fit(X_train, y_train)
    return clf
