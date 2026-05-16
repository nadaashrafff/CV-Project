"""Feature loading, saving, and normalisation utilities."""
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_features(features: np.ndarray, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, features)
    logger.info(f"Saved features {features.shape} → {path}")


def load_features(path: str) -> np.ndarray:
    return np.load(path)


def load_split_features(feature_dir: str, feature_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load train/val/test arrays for a given feature type (e.g. 'classical', 'deep')."""
    base = Path(feature_dir)
    train = load_features(base / feature_type / f"train_{feature_type}.npy")
    val   = load_features(base / feature_type / f"val_{feature_type}.npy")
    test  = load_features(base / feature_type / f"test_{feature_type}.npy")
    return train, val, test


def normalize_features(
    train: np.ndarray, val: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit StandardScaler on train, apply to val/test."""
    scaler = StandardScaler()
    train_n = scaler.fit_transform(train)
    val_n   = scaler.transform(val)
    test_n  = scaler.transform(test)
    return train_n, val_n, test_n
