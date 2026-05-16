"""Extract intermediate FPN feature maps from a trained YOLOv8 model (E11)."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.utils.logger import get_logger

logger = get_logger(__name__)

_TRANSFORM = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
])


def load_yolo_backbone(weights_path: str):
    from ultralytics import YOLO
    model = YOLO(weights_path)
    return model.model


def extract_fpn_features(
    img_paths: list[str],
    weights_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> np.ndarray:
    """Extract FPN neck features from YOLOv8 for use in E11 hybrid."""
    raise NotImplementedError(
        "Register forward hooks on model.model.model[-2] (neck) to capture FPN outputs"
    )
