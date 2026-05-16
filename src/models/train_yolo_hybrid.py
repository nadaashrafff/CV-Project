"""E11: YOLO Hybrid — extend YOLOv8 detection with classical feature augmentation.

Strategy: extract classical features per detected crop, concatenate with
YOLO FPN features, re-classify with a lightweight head.
"""
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_hybrid_classifier(yolo_weights: str, classical_feature_dir: str) -> None:
    """
    1. Load pretrained YOLOv8 (E1 weights).
    2. For each detected crop, extract classical 252-dim features.
    3. Extract FPN features via forward hooks.
    4. Concatenate and train a lightweight MLP classifier on top.
    """
    raise NotImplementedError("E11 implementation — assigned to Nada")
