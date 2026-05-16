"""
E1 — YOLO backbone feature extraction [Modal A10G]

Uses the trained yolov8n_E1_best.pt (already in e1-yolo-results Volume) to
extract 256-dim SPPF backbone features from all 65K crops (for E11 fusion).

Volumes:
  e1-yolo-results (in/out) — reads best.pt, writes features/yolo_*.npy
  e3-crops-v1     (in)     — dataset_crops.zip (already uploaded for E2/E3)

Usage:
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run scripts/run_e1_extract_features_modal.py

Downloads to local repo:
    data/processed/features/yolo_train_X.npy   (45177 x 256)
    data/processed/features/yolo_train_y.npy
    data/processed/features/yolo_val_X.npy     (9935  x 256)
    data/processed/features/yolo_val_y.npy
    data/processed/features/yolo_test_X.npy    (10553 x 256)
    data/processed/features/yolo_test_y.npy
    data/processed/features/yolo_feature_manifest.json
"""
from datetime import datetime
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("e1-yolo-features")

CROPS_VOL   = modal.Volume.from_name("e3-crops-v1")
RESULTS_VOL = modal.Volume.from_name("e1-yolo-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "ultralytics>=8.2.0",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "pyyaml>=6.0",
    )
)

CLASSES      = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
YOLO_DIM     = 256
SEED         = 42


@app.function(
    image=image,
    gpu="A10G",
    cpu=4,
    memory=16384,
    timeout=1800,
    volumes={
        "/crops_vol":   CROPS_VOL,
        "/results_vol": RESULTS_VOL,
    },
)
def extract_yolo_features():
    import json
    import time
    import zipfile
    from pathlib import Path

    import numpy as np
    import torch
    from PIL import Image as PILImage
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from ultralytics import YOLO

    torch.manual_seed(SEED)

    RESULTS_DIR = Path("/results_vol")
    feat_dir = RESULTS_DIR / "features"
    feat_dir.mkdir(exist_ok=True)

    best_weights = RESULTS_DIR / "yolo_runs" / "E1_baseline_v2" / "weights" / "best.pt"
    assert best_weights.exists(), f"best.pt not found at {best_weights}"
    print(f"Loaded weights: {best_weights}")

    CROPS_DIR = Path("/tmp/dataset_crops")
    if not (CROPS_DIR / "train").exists():
        print("Extracting crops zip ...")
        CROPS_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile("/crops_vol/dataset_crops.zip", "r") as zf:
            zf.extractall(CROPS_DIR)
    print(f"  Crops at: {CROPS_DIR}")

    # Register SPPF hook (layer 9 in YOLOv8n -> 256-dim output)
    feat_model  = YOLO(str(best_weights))
    nn_backbone = feat_model.model.cuda()
    nn_backbone.train(False)  # inference mode (equivalent to .eval())

    _buf = {}
    def _sppf_hook(module, inp, out):
        _buf["sppf"] = out.detach()

    _handle = nn_backbone.model[9].register_forward_hook(_sppf_hook)

    # Resize to 224 + ToTensor [0,1] — no ImageNet norm (YOLO convention)
    yolo_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    class CropDS(Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, i):
            path, label = self.items[i]
            img = PILImage.open(path).convert("RGB")
            return yolo_tf(img), label

    def collect_items(split):
        items = []
        split_dir = CROPS_DIR / split
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.name not in CLASS_TO_IDX:
                continue
            label = CLASS_TO_IDX[cls_dir.name]
            for img_path in sorted(cls_dir.glob("*.jpg")):
                items.append((img_path, label))
        return items

    def extract_split(split, batch_sz=512):
        items = collect_items(split)
        dl = DataLoader(CropDS(items), batch_size=batch_sz, num_workers=4, pin_memory=True)
        X_list, y_list = [], []
        for imgs, labs in dl:
            imgs = imgs.cuda()
            with torch.no_grad():
                nn_backbone(imgs)
            gap = _buf["sppf"].mean(dim=[2, 3]).cpu().numpy()  # [B, 256]
            X_list.append(gap)
            y_list.append(labs.numpy())
        return np.vstack(X_list), np.concatenate(y_list)

    feat_meta = {}
    for split in ["train", "val", "test"]:
        print(f"  Extracting {split} ...", end="", flush=True)
        t0 = time.time()
        X, y = extract_split(split)
        elapsed = time.time() - t0
        np.save(feat_dir / f"yolo_{split}_X.npy", X)
        np.save(feat_dir / f"yolo_{split}_y.npy", y)
        feat_meta[split] = {
            "shape": list(X.shape),
            "dtype": str(X.dtype),
            "n_samples": int(X.shape[0]),
            "feature_dim": int(X.shape[1]),
            "extraction_s": round(elapsed, 1),
        }
        print(f"  {X.shape}  ({elapsed:.0f}s)")

    _handle.remove()

    manifest = {
        "experiment"   : "E1",
        "extractor"    : "YOLOv8n SPPF (layer 9) + GlobalAveragePool",
        "feature_dim"  : YOLO_DIM,
        "model_weights": "yolov8n_E1_best.pt",
        "note"         : "Row-aligned with classical_train_clean_X.npy and deep_train_X.npy (no augmentation)",
        "splits"       : feat_meta,
        "created"      : datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open(feat_dir / "yolo_feature_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    RESULTS_VOL.commit()
    print("Feature extraction complete.")
    return feat_meta


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("E1  YOLO backbone feature extraction  [Modal A10G]")
    print("=" * 60)

    print("Launching on Modal A10G (~10 min) ...")
    feat_meta = extract_yolo_features.remote()

    print("\nFeature shapes:")
    for split, m in feat_meta.items():
        print(f"  {split}: {m['shape']}  ({m['extraction_s']}s)")

    vol = modal.Volume.from_name("e1-yolo-results")
    local_feat = REPO_ROOT / "data" / "processed" / "features"
    local_feat.mkdir(parents=True, exist_ok=True)

    files = [
        "features/yolo_train_X.npy",
        "features/yolo_train_y.npy",
        "features/yolo_val_X.npy",
        "features/yolo_val_y.npy",
        "features/yolo_test_X.npy",
        "features/yolo_test_y.npy",
        "features/yolo_feature_manifest.json",
    ]
    print("\nDownloading ...")
    for remote in files:
        local = local_feat / Path(remote).name
        print(f"  {Path(remote).name} ...", end="", flush=True)
        try:
            with open(local, "wb") as f:
                for chunk in vol.read_file(remote):
                    f.write(chunk)
            print(f"  {local.stat().st_size / 1024:.0f} KB")
        except Exception as e:
            print(f"  SKIP ({e})")

    print(f"\nDone. YOLO features -> data/processed/features/yolo_{{train,val,test}}_X.npy")
