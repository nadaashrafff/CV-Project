"""E3: Train SVM on deep features."""
import joblib
import numpy as np
from sklearn.svm import SVC

from src.utils.seed import SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> SVC:
    params = dict(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced",
                  probability=True, random_state=SEED)
    params.update(kwargs)
    clf = SVC(**params)
    clf.fit(X_train, y_train)
    logger.info(f"SVM trained on {X_train.shape}")
    return clf


def save(clf: SVC, path: str) -> None:
    joblib.dump(clf, path)


def load(path: str) -> SVC:
    return joblib.load(path)
