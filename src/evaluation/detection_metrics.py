"""Detection metrics for E1 (YOLO baseline) and E11 (YOLO hybrid).

YOLOv8 computes mAP internally during training. These helpers parse the
results CSV and surface the key numbers for the final comparison table.
"""
import pandas as pd
from pathlib import Path


def parse_yolo_results(results_csv: str) -> dict:
    """Read ultralytics results.csv and return final-epoch metrics."""
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    last = df.iloc[-1]
    return {
        "mAP50":      float(last.get("metrics/mAP50(B)", float("nan"))),
        "mAP50_95":   float(last.get("metrics/mAP50-95(B)", float("nan"))),
        "precision":  float(last.get("metrics/precision(B)", float("nan"))),
        "recall":     float(last.get("metrics/recall(B)", float("nan"))),
    }


def get_model_size_mb(weights_path: str) -> float:
    return Path(weights_path).stat().st_size / 1e6
