"""E2: Train Random Forest on classical features."""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.utils.seed import SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> RandomForestClassifier:
    params = dict(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=SEED)
    params.update(kwargs)
    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)
    logger.info(f"Random Forest trained on {X_train.shape}")
    return clf


def save(clf: RandomForestClassifier, path: str) -> None:
    joblib.dump(clf, path)
    logger.info(f"Saved → {path}")


def load(path: str) -> RandomForestClassifier:
    return joblib.load(path)
