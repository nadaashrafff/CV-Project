"""
Zip the local dataset split for uploading to Google Drive.

Run this BEFORE opening the Colab notebook:
    python scripts/zip_split_for_upload.py

Then upload the resulting zip to Google Drive at:
    MyDrive/CV/dataset_split_70_15_15.zip

The Colab GPU notebook will extract it from there.
"""
import zipfile
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
SPLIT_DIR  = REPO_ROOT / "data" / "processed" / "dataset_split_70_15_15"
ZIP_OUT    = REPO_ROOT / "data" / "processed" / "dataset_split_70_15_15.zip"

assert SPLIT_DIR.exists(), f"Split not found: {SPLIT_DIR}\nRun preprocessing first: python scripts/run_preprocessing.py"

all_files = [f for f in SPLIT_DIR.rglob("*") if f.is_file()]
print(f"Zipping {len(all_files):,} files from {SPLIT_DIR.name}/ ...")

with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for i, f in enumerate(sorted(all_files)):
        arcname = f.relative_to(SPLIT_DIR.parent)   # keeps dataset_split_70_15_15/... prefix
        zf.write(f, arcname)
        if i % 500 == 0:
            print(f"  {i:>6}/{len(all_files)}  {f.name}")

size_mb = ZIP_OUT.stat().st_size / (1024 * 1024)
print(f"\nCreated: {ZIP_OUT}")
print(f"Size   : {size_mb:.1f} MB")
print()
print("Next: upload this zip to Google Drive at  MyDrive/CV/dataset_split_70_15_15.zip")
print("Then open: notebooks/colab_yolo_e1_gpu.ipynb  on Colab (T4 runtime)")
