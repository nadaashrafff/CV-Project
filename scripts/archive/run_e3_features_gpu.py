"""
E3 — EfficientNetB0 Feature Extraction with Training Augmentation (GPU)

Augmentation strategy:
  Training split:  RandomHorizontalFlip + RandomVerticalFlip + RandomRotation(15°)
                   + ColorJitter(brightness/contrast/saturation 0.3, hue 0.1)
  Val / Test:      Resize + Normalize only  (clean evaluation)

Rationale: frozen EfficientNetB0 backbone; augmenting training crops gives the
downstream SVM more diverse 1,280-dim feature vectors, improving decision
boundary quality especially for minority classes.

Run on Colab T4 or any CUDA machine, then run locally:
    python scripts/run_e3_svm.py

Usage:
    python scripts/run_e3_features_gpu.py
    python scripts/run_e3_features_gpu.py --force   # re-extract even if cached

Saves:
    data/processed/features/deep_{train,val,test}_{X,y}.npy
"""
import sys, time, argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from src.features.deep_features import DEEP_DIM
from src.utils.paths import CLASSES, CLASS_TO_IDX
from src.utils.seed import set_seed
from src.utils.logger import get_logger

set_seed()
logger = get_logger("e3_features")

CROPS_DIR    = REPO_ROOT / "data" / "processed" / "dataset_crops"
FEATURES_DIR = REPO_ROOT / "data" / "processed" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Transforms ────────────────────────────────────────────────────────────────
_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    _NORMALIZE,
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    _NORMALIZE,
])


class CropDataset(Dataset):
    def __init__(self, paths: list, labels: list, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def load_crop_paths(split: str):
    paths, labels = [], []
    for cls in CLASSES:
        cls_dir = CROPS_DIR / split / cls
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.glob("*.jpg")):
            paths.append(str(p))
            labels.append(CLASS_TO_IDX[cls])
    return paths, labels


def extract_split(model: nn.Module, split: str, force: bool = False):
    X_path = FEATURES_DIR / f"deep_{split}_X.npy"
    y_path = FEATURES_DIR / f"deep_{split}_y.npy"

    if X_path.exists() and y_path.exists() and not force:
        logger.info(f"[{split}] Cache found — skipping")
        return

    paths, labels = load_crop_paths(split)
    assert paths, f"No crops found: {CROPS_DIR / split}"

    transform = TRAIN_TRANSFORM if split == "train" else EVAL_TRANSFORM
    aug_note  = "augmented" if split == "train" else "clean"
    logger.info(f"[{split}] {len(paths):,} crops ({aug_note}) on {DEVICE} ...")

    dataset = CropDataset(paths, labels, transform)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False,
                         num_workers=0, pin_memory=(DEVICE == "cuda"))

    all_X, all_y = [], []
    t0 = time.perf_counter()

    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(loader):
            imgs  = imgs.to(DEVICE)
            feats = model(imgs).cpu().numpy()
            all_X.append(feats)
            all_y.extend(lbls.tolist() if hasattr(lbls, "tolist") else list(lbls))
            if (i + 1) % 50 == 0:
                done = min((i + 1) * loader.batch_size, len(paths))
                rate = done / max(time.perf_counter() - t0, 1e-6)
                eta  = (len(paths) - done) / rate
                print(f"  {done:>6}/{len(paths)}  {rate:.0f} crops/s  "
                      f"ETA {eta/60:.1f} min")

    X = np.vstack(all_X).astype(np.float32)
    y = np.array(all_y, dtype=np.int32)

    assert X.shape == (len(paths), DEEP_DIM), \
        f"Shape mismatch: expected ({len(paths)}, {DEEP_DIM}), got {X.shape}"

    np.save(X_path, X)
    np.save(y_path, y)
    logger.info(f"[{split}] {X.shape}  saved in "
                f"{time.perf_counter()-t0:.1f}s → {X_path}")


def main(force: bool = False):
    print("=" * 60)
    print("E3  EfficientNetB0 Feature Extraction  [augmented training]")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    assert CROPS_DIR.exists(), (
        f"Crops not found: {CROPS_DIR}\n"
        "Run: python scripts/run_preprocessing.py"
    )

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
    model.eval().to(DEVICE)

    for split in ["train", "val", "test"]:
        extract_split(model, split, force)

    print("\nDone. Feature files saved to:", FEATURES_DIR)
    print("Next: python scripts/run_e3_svm.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Re-extract even if cache exists")
    args = ap.parse_args()
    main(force=args.force)
