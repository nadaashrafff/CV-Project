"""Extract 224×224 object crops from ground-truth YOLO bounding boxes."""
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.logger import get_logger
from src.utils.paths import COLAB_CROPS, COLAB_SPLIT, CLASSES

logger = get_logger(__name__)

CROP_SIZE = 224
MIN_CROP_PX = 16


def yolo_box_to_pixels(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    return x1, y1, x2, y2


def extract_crops(
    split_dir: Path = COLAB_SPLIT,
    crops_dir: Path = COLAB_CROPS,
) -> list[dict]:
    records = []
    for split in ("train", "val", "test"):
        img_dir = split_dir / split / "images"
        lbl_dir = split_dir / split / "labels"
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            if not lbl_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            iw, ih = img.size
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                x1, y1, x2, y2 = yolo_box_to_pixels(cx, cy, bw, bh, iw, ih)
                if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
                    continue
                crop = img.crop((x1, y1, x2, y2)).resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)
                class_name = CLASSES[cls_id]
                out_dir = crops_dir / split / class_name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{img_path.stem}_box{len(records)}.jpg"
                crop.save(out_path)
                records.append({"path": str(out_path), "split": split, "class": class_name,
                                 "source_image": img_path.name, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    logger.info(f"Extracted {len(records)} crops")
    return records


def convert_labels(label_dir: Path, output_dir: Path) -> None:
    """Placeholder: convert between annotation formats if needed."""
    raise NotImplementedError
