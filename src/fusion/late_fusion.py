"""Late fusion: train classifiers independently, combine predictions (E8–E9)."""
import numpy as np


def average_voting(proba_a: np.ndarray, proba_b: np.ndarray) -> np.ndarray:
    """E8: equal-weight average of two probability arrays."""
    return (proba_a + proba_b) / 2.0


def weighted_voting(
    proba_a: np.ndarray, proba_b: np.ndarray, weight_a: float = 0.4
) -> np.ndarray:
    """E9: weighted combination. weight_b = 1 - weight_a."""
    weight_b = 1.0 - weight_a
    return weight_a * proba_a + weight_b * proba_b


def find_optimal_weight(
    proba_a: np.ndarray, proba_b: np.ndarray, y_true: np.ndarray, steps: int = 19
) -> float:
    """Grid search over weight_a in [0.05, 0.95] to maximise macro-F1 on val set."""
    from sklearn.metrics import f1_score
    best_w, best_f1 = 0.5, 0.0
    for w in np.linspace(0.05, 0.95, steps):
        preds = np.argmax(weighted_voting(proba_a, proba_b, w), axis=1)
        score = f1_score(y_true, preds, average="macro")
        if score > best_f1:
            best_f1, best_w = score, w
    return best_w
