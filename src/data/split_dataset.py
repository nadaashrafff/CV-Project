"""Stratified 70/15/15 train/val/test split at image level.

Stratification key: dominant class (class with most bounding boxes per image).
Discards the original Roboflow split which was missing entire classes from val/test.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.utils.paths import COLAB_SPLIT
from src.utils.seed import SEED

logger = get_logger(__name__)

SPLIT_RATIOS = (0.70, 0.15, 0.15)


def get_dominant_class(label_path: Path) -> int:
    """Return the class ID with the most bounding boxes in a label file."""
    lines = label_path.read_text().strip().splitlines()
    classes = [int(l.split()[0]) for l in lines if l.strip()]
    if not classes:
        return -1
    return max(set(classes), key=classes.count)


def build_image_inventory(dataset_root: Path) -> pd.DataFrame:
    """Collect all image paths and their dominant class across the unified pool."""
    records = []
    for img in sorted(dataset_root.rglob("*.jpg")):
        label = img.parent.parent / "labels" / img.with_suffix(".txt").name
        if not label.exists():
            continue
        dom = get_dominant_class(label)
        if dom == -1:
            continue
        records.append({"stem": img.stem, "image": img, "label": label, "dominant_class": dom})
    return pd.DataFrame(records)


def make_split(df: pd.DataFrame, output_dir: Path = COLAB_SPLIT) -> dict[str, pd.DataFrame]:
    """Perform stratified split and copy files into YOLO directory structure."""
    train_val, test = train_test_split(
        df, test_size=SPLIT_RATIOS[2], stratify=df["dominant_class"], random_state=SEED
    )
    val_ratio_adjusted = SPLIT_RATIOS[1] / (SPLIT_RATIOS[0] + SPLIT_RATIOS[1])
    train, val = train_test_split(
        train_val, test_size=val_ratio_adjusted, stratify=train_val["dominant_class"], random_state=SEED
    )

    splits = {"train": train, "val": val, "test": test}
    for split_name, split_df in splits.items():
        for row in split_df.itertuples():
            for src, subdir in [(row.image, "images"), (row.label, "labels")]:
                dst = output_dir / split_name / subdir / Path(src).name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        logger.info(f"{split_name}: {len(split_df)} images")

    return splits
