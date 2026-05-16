"""Label format utilities (YOLO ↔ other formats)."""
from pathlib import Path


def yolo_to_pascal_voc(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    """Convert YOLO normalized bbox to Pascal VOC pixel coordinates."""
    boxes = []
    for line in Path(label_path).read_text().strip().splitlines():
        cls_id, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        boxes.append({"class_id": int(cls_id), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return boxes
