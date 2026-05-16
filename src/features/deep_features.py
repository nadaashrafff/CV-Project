"""EfficientNetB0 deep feature extraction — 1280-dim late features per crop."""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

DEEP_DIM = 1280  # EfficientNetB0 global-average-pool output (num_classes=0)
BATCH_SIZE = 64
IMG_SIZE = 224

_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    _NORMALIZE,
])


class CropDataset(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return _TRANSFORM(img), self.paths[idx]


def build_efficientnet(device: str = "cuda") -> nn.Module:
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
    model.eval().to(device)
    return model


def extract_deep_features(
    crop_paths: list[str],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    set_seed()
    model = build_efficientnet(device)
    loader = DataLoader(CropDataset(crop_paths), batch_size=batch_size, num_workers=2)
    all_feats = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            feats = model(imgs).cpu().numpy()
            all_feats.append(feats)
    return np.vstack(all_feats)


def extract_multilevel_features(
    crop_paths: list[str],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, np.ndarray]:
    """Return early, mid, and late feature maps for E4 raw fusion.
    Implement via forward hooks on model blocks.
    """
    raise NotImplementedError("Hook-based multi-level extraction not yet implemented")
