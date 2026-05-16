"""
Extract yolo_e1_results.zip (downloaded from Colab) into the local repo layout.

Usage:
    python scripts/extract_colab_results.py yolo_e1_results.zip

The zip was created by the Colab notebook's final cell — its internal layout mirrors
the local repo structure, so extraction maps directly to the right folders.

Files extracted:
    models/yolo/yolo_baseline_E1.pt
    models/checkpoints/last/yolo_E1_last.pt
    results/metrics/detection_results.csv
    results/metrics/E1_per_class_metrics.csv
    results/predictions/E1_yolo_predictions.csv
    results/logs/training_logs/E1_training_results.csv
    figures/yolo/  (training curves, confusion matrix, etc.)
"""
import sys
import zipfile
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/extract_colab_results.py <path/to/yolo_e1_results.zip>")
        sys.exit(1)

    zip_path = Path(sys.argv[1])
    if not zip_path.exists():
        print(f"ERROR: zip not found: {zip_path}")
        sys.exit(1)

    print(f"Extracting {zip_path.name}  ({zip_path.stat().st_size / 1e6:.1f} MB) ...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        print(f"  {len(members)} files in archive")

        for member in members:
            dest = REPO_ROOT / member
            dest.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(member) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)

    print("\nExtracted to repo:")
    for rel in ["models/yolo", "models/checkpoints/last",
                "results/metrics", "results/predictions",
                "results/logs/training_logs", "figures/yolo"]:
        p = REPO_ROOT / rel
        if p.exists():
            files = list(p.iterdir())
            print(f"  {rel}/  ({len(files)} files)")

    print("\nDone. Run results are now in the local repo.")

if __name__ == "__main__":
    main()
