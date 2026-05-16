from pathlib import Path

# Google Drive base (mounted in Colab)
DRIVE_BASE = Path("/content/drive/MyDrive/CV")
DATASET_ZIP = DRIVE_BASE / "Trash_Dataset.zip"

# Drive output directories (persistent across sessions)
DRIVE_ANALYSIS = DRIVE_BASE / "01_dataset_analysis"
DRIVE_SPLIT    = DRIVE_BASE / "02_dataset_split"
DRIVE_YOLO     = DRIVE_BASE / "03_yolo_baseline"
DRIVE_CROPS    = DRIVE_BASE / "04_crops"
DRIVE_FEATURES = DRIVE_BASE / "05_features"

# Colab local runtime paths (fast I/O, lost on disconnect)
COLAB_DATASET = Path("/content/dataset")
COLAB_SPLIT   = Path("/content/dataset_split_70_15_15")
COLAB_CROPS   = Path("/content/dataset_crops")

CLASSES = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
