"""
E3 — EfficientNetB0 feature extraction on Modal T4 GPU.
Compatible with Modal >= 1.0 (no deprecated Mount API).

Usage:
    modal run scripts/run_e3_features_modal.py
    modal run scripts/run_e3_features_modal.py --force   # re-extract

What this does:
  1. Zips dataset_crops/ locally (once, cached)
  2. Uploads the zip to a Modal Volume
  3. Runs EfficientNetB0 extraction on T4 (~5–10 min)
  4. Downloads 6 .npy files back to data/processed/features/

Next step after this finishes:
    python scripts/run_e3_svm.py --linear
"""
import zipfile
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Modal app ──────────────────────────────────────────────────────────────────
app = modal.App("e3-efficientnet-features")

# Volumes — persisted across runs
CROPS_VOL  = modal.Volume.from_name("e3-crops-v1",    create_if_missing=True)
OUTPUT_VOL = modal.Volume.from_name("e3-features-out", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "timm>=0.9.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
    )
)

CLASSES      = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
DEEP_DIM     = 1280
SEED         = 42


# ── Remote GPU function ────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    volumes={
        "/crops_vol" : CROPS_VOL,
        "/outputs"   : OUTPUT_VOL,
    },
)
def extract_features(force: bool = False):
    import random
    import time
    import zipfile
    from pathlib import Path

    import numpy as np
    import torch
    import timm
    from PIL import Image as PILImage
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Unzip crops on first run (stays in /tmp for session lifetime)
    CROPS_DIR = Path("/tmp/dataset_crops")
    if not (CROPS_DIR / "train").exists() or force:
        zip_path = Path("/crops_vol/dataset_crops.zip")
        assert zip_path.exists(), (
            "dataset_crops.zip not found in Modal Volume.\n"
            "The local entrypoint should have uploaded it — rerun the script."
        )
        print("Extracting crops archive ...")
        CROPS_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(CROPS_DIR))
        print(f"  Extracted to {CROPS_DIR}")

    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
    FEATURES_DIR = Path("/outputs")

    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    _NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
    TRAIN_TF = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),   # convert before ColorJitter to avoid PIL hue overflow
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1),
        _NORM,
    ])
    EVAL_TF = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), _NORM,
    ])

    class CropDataset(Dataset):
        def __init__(self, paths, labels, transform):
            self.paths, self.labels, self.transform = paths, labels, transform
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            img = PILImage.open(self.paths[idx]).convert("RGB")
            return self.transform(img), self.labels[idx]

    def load_crop_paths(split):
        paths, labels = [], []
        for cls in CLASSES:
            cls_dir = CROPS_DIR / split / cls
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.glob("*.jpg")):
                paths.append(str(p))
                labels.append(CLASS_TO_IDX[cls])
        return paths, labels

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
    model.eval().to(DEVICE)

    results = {}
    for split in ["train", "val", "test"]:
        X_path = FEATURES_DIR / f"deep_{split}_X.npy"
        y_path = FEATURES_DIR / f"deep_{split}_y.npy"

        if X_path.exists() and y_path.exists() and not force:
            shape = tuple(np.load(str(X_path)).shape)
            print(f"[{split}] cached — {shape}")
            results[split] = shape
            continue

        paths, labels = load_crop_paths(split)
        assert paths, f"No crops found: {CROPS_DIR / split}"

        tf       = TRAIN_TF if split == "train" else EVAL_TF
        aug_note = "aug" if split == "train" else "clean"
        print(f"[{split}] {len(paths):,} crops ({aug_note}) ...")

        loader = DataLoader(
            CropDataset(paths, labels, tf),
            batch_size=64, shuffle=False, num_workers=4, pin_memory=True,
        )

        all_X, all_y = [], []
        t0 = time.perf_counter()
        with torch.no_grad():
            for i, (imgs, lbls) in enumerate(loader):
                all_X.append(model(imgs.to(DEVICE)).cpu().numpy())
                all_y.extend(lbls.tolist())
                if (i + 1) % 50 == 0:
                    done = min((i + 1) * 64, len(paths))
                    rate = done / max(time.perf_counter() - t0, 1e-6)
                    eta  = (len(paths) - done) / max(rate, 1e-6)
                    print(f"  {done:>6}/{len(paths)}  "
                          f"{rate:.0f} crops/s  ETA {eta/60:.1f} min")

        X = np.vstack(all_X).astype(np.float32)
        y = np.array(all_y, dtype=np.int32)
        assert X.shape == (len(paths), DEEP_DIM), \
            f"Shape mismatch: expected ({len(paths)}, {DEEP_DIM}), got {X.shape}"

        np.save(str(X_path), X)
        np.save(str(y_path), y)
        OUTPUT_VOL.commit()
        print(f"[{split}] saved {X.shape}  ({time.perf_counter()-t0:.1f}s)")
        results[split] = tuple(X.shape)

    return results


# ── Local entrypoint ───────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(force: bool = False):
    print("=" * 60)
    print("E3  EfficientNetB0 Feature Extraction  [Modal T4]")
    print("=" * 60)

    # ── 1. Create crops zip (skipped if already exists) ───────────────────────
    crops_dir = REPO_ROOT / "data" / "processed" / "dataset_crops"
    zip_path  = REPO_ROOT / "data" / "processed" / "dataset_crops.zip"

    assert crops_dir.exists(), f"Crops not found: {crops_dir}"

    if not zip_path.exists():
        print(f"Creating crops zip (one-time) ...")
        all_jpgs = sorted(crops_dir.rglob("*.jpg"))
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for i, f in enumerate(all_jpgs):
                zf.write(f, f.relative_to(crops_dir))
                if (i + 1) % 10000 == 0:
                    print(f"  zipped {i+1:,}/{len(all_jpgs):,}")
        print(f"  {zip_path.stat().st_size/1e6:.0f} MB")
    else:
        print(f"Crops zip exists ({zip_path.stat().st_size/1e6:.0f} MB)")

    # ── 2. Upload zip to Modal Volume (skipped if already uploaded) ────────────
    existing = {entry.path.lstrip("/") for entry in CROPS_VOL.listdir("/")}
    if "dataset_crops.zip" not in existing or force:
        size_mb = zip_path.stat().st_size / 1e6
        print(f"Uploading {size_mb:.0f} MB to Modal Volume ...")
        with CROPS_VOL.batch_upload() as batch:
            batch.put_file(str(zip_path), "dataset_crops.zip")
        print("  Upload complete.")
    else:
        print("Crops zip already in Modal Volume — skipping upload.")

    # ── 3. Run GPU extraction on T4 ───────────────────────────────────────────
    print("\nLaunching GPU extraction on T4 ...")
    shapes = extract_features.remote(force=force)

    print("\nExtracted shapes:")
    for split, shape in shapes.items():
        print(f"  {split:<6}: {shape}")

    # ── 4. Download .npy files from Modal Volume ───────────────────────────────
    features_dir = REPO_ROOT / "data" / "processed" / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading .npy files to {features_dir.relative_to(REPO_ROOT)} ...")

    for entry in OUTPUT_VOL.listdir("/"):
        fname = Path(entry.path).name
        if not fname.endswith(".npy"):
            continue
        local_path = features_dir / fname
        print(f"  {fname} ...", end="", flush=True)
        with open(local_path, "wb") as f:
            for chunk in OUTPUT_VOL.read_file(entry.path):
                f.write(chunk)
        print(f"  {local_path.stat().st_size/1e6:.1f} MB")

    print("\n" + "=" * 60)
    print("All done. Run SVM training next:")
    print("  python scripts/run_e3_svm.py --linear   # fast (minutes)")
    print("  python scripts/run_e3_svm.py            # RBF (2–6 hrs)")
    print("=" * 60)
