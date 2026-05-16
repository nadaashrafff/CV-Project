"""
E1 — YOLOv8-nano detection baseline + YOLO backbone feature extraction [Modal H100]

Steps:
  1. Train YOLOv8n on dataset_split_70_15_15 (100 epochs, v2 hyperparams)
  2. Evaluate best.pt on test split → detection metrics
  3. Extract YOLO backbone (SPPF, 256-dim) features from all crops → for E11

Epoch choice: 100 (not 120). Colab logs showed mAP50 plateaued by epoch ~90
in the mosaic phase. close_mosaic=15 still gives 15 fine-tuning epochs (86-100).

GPU: H100 (80 GB). Expected runtime: ~50 min total.

Volumes used:
  e1-yolo-data    (in)  — dataset_split_70_15_15.zip
  e3-crops-v1     (in)  — dataset_crops.zip  (already uploaded for E2/E3)
  e1-yolo-results (out) — weights, metrics, features, figures

Usage:
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run scripts/run_e1_yolo_modal.py

Downloads to local repo:
    models/yolo/yolov8n_E1_best.pt
    models/yolo/yolov8n_E1_last.pt
    results/metrics/detection_results.csv
    results/metrics/E1_per_class_metrics.csv
    results/predictions/E1_yolo_predictions.csv
    data/processed/features/yolo_train_X.npy   (45177 × 256)
    data/processed/features/yolo_train_y.npy
    data/processed/features/yolo_val_X.npy     (9935  × 256)
    data/processed/features/yolo_val_y.npy
    data/processed/features/yolo_test_X.npy    (10553 × 256)
    data/processed/features/yolo_test_y.npy
    data/processed/features/yolo_feature_manifest.json
    figures/yolo/  (training curves, confusion matrices, PR/F1 curves)
    archive/colab_e1_v2/results.csv
"""
from datetime import datetime
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("e1-yolo-baseline")

DATA_VOL    = modal.Volume.from_name("e1-yolo-data",    create_if_missing=True)
CROPS_VOL   = modal.Volume.from_name("e3-crops-v1")          # already populated by E2/E3
RESULTS_VOL = modal.Volume.from_name("e1-yolo-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")   # ultralytics/cv2 needs libGL.so.1
    .pip_install(
        "ultralytics>=8.2.0",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "pyyaml>=6.0",
    )
)

CLASSES      = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
YOLO_DIM     = 256    # SPPF output channels for YOLOv8n
SEED         = 42


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=10800,      # 3 hours; actual runtime ~50 min
    volumes={
        "/data_vol":    DATA_VOL,
        "/crops_vol":   CROPS_VOL,
        "/results_vol": RESULTS_VOL,
    },
)
def run_e1():
    import json
    import shutil
    import time
    import zipfile
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from PIL import Image as PILImage
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from ultralytics import YOLO

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    RESULTS_DIR = Path("/results_vol")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(exist_ok=True)
    (RESULTS_DIR / "features").mkdir(exist_ok=True)

    # ── 1. Extract YOLO-format dataset ────────────────────────────────────────
    print("[1/4] Extracting YOLO dataset ...")
    extract_dir = Path("/tmp/dataset")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile("/data_vol/dataset_split_70_15_15.zip", "r") as zf:
        zf.extractall(extract_dir)

    data_yamls = list(extract_dir.rglob("data.yaml"))
    assert data_yamls, "No data.yaml found in zip"
    data_yaml    = data_yamls[0]
    dataset_root = data_yaml.parent

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    cfg["path"]  = str(dataset_root)
    cfg["train"] = "train/images"
    cfg["val"]   = "val/images"
    cfg["test"]  = "test/images"
    with open(data_yaml, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)
    print(f"  dataset_root : {dataset_root}")
    print(f"  classes      : {cfg.get('names')}")

    # ── 2. Train ──────────────────────────────────────────────────────────────
    # batch=16 kept identical to Colab run so LR dynamics are unchanged.
    # Speed gain is purely from H100 compute. amp=True uses bf16 on H100.
    print("\n[2/4] Training YOLOv8n — 100 epochs, imgsz=640, freeze=10 ...")
    model    = YOLO("yolov8n.pt")
    train_t0 = time.time()
    model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=16,
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        freeze=10,
        close_mosaic=15,
        copy_paste=0.0,
        patience=25,
        workers=4,
        device=0,
        seed=SEED,
        name="E1_baseline_v2",
        project=str(RESULTS_DIR / "yolo_runs"),
        exist_ok=True,
        plots=True,
        save=True,
        save_period=-1,
        amp=True,
    )
    train_mins = (time.time() - train_t0) / 60
    print(f"  Training finished in {train_mins:.1f} min")

    run_dir      = RESULTS_DIR / "yolo_runs" / "E1_baseline_v2"
    best_weights = run_dir / "weights" / "best.pt"

    # ── 3. Evaluate on test split ─────────────────────────────────────────────
    print("\n[3/4] Evaluating best.pt on test split ...")
    test_model = YOLO(str(best_weights))

    test_metrics = test_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        batch=32,
        conf=0.25,
        iou=0.45,
        device=0,
        plots=True,
        save_json=False,
        name="E1_test_val",
        project=str(RESULTS_DIR / "yolo_runs"),
        exist_ok=True,
    )

    map50   = float(test_metrics.box.map50)
    map5095 = float(test_metrics.box.map)
    prec    = float(test_metrics.box.mp)
    recall  = float(test_metrics.box.mr)

    speed    = test_metrics.speed
    total_ms = speed.get("preprocess", 0) + speed.get("inference", 0) + speed.get("postprocess", 0)
    fps      = round(1000.0 / total_ms, 2) if total_ms > 0 else float("nan")
    model_size_mb = round(best_weights.stat().st_size / 1e6, 3)

    # Per-class AP50 (reliably available)
    per_class_map50 = {}
    try:
        ap50_arr = list(test_metrics.box.ap50)
        for i, c in enumerate(CLASSES):
            per_class_map50[c] = float(ap50_arr[i]) if i < len(ap50_arr) else float("nan")
    except Exception:
        for c in CLASSES:
            per_class_map50[c] = float("nan")

    # Per-class precision/recall (version-dependent — fail gracefully)
    per_class_p, per_class_r, per_class_f1 = {}, {}, {}
    try:
        p_arr = list(test_metrics.box.p)
        r_arr = list(test_metrics.box.r)
        for i, c in enumerate(CLASSES):
            p = float(p_arr[i]) if i < len(p_arr) else float("nan")
            r = float(r_arr[i]) if i < len(r_arr) else float("nan")
            per_class_p[c]  = p
            per_class_r[c]  = r
            per_class_f1[c] = round(2 * p * r / (p + r), 4) if (p + r) > 0 else float("nan")
    except Exception as e:
        print(f"  [warn] per-class P/R unavailable ({e}) — will use NaN")
        for c in CLASSES:
            per_class_p[c] = per_class_r[c] = per_class_f1[c] = float("nan")

    print(f"  mAP@0.5      : {map50:.4f}")
    print(f"  mAP@0.5:0.95 : {map5095:.4f}")
    print(f"  Precision    : {prec:.4f}")
    print(f"  Recall       : {recall:.4f}")
    print(f"  FPS          : {fps}")
    print(f"  Model size   : {model_size_mb} MB")
    print("  Per-class mAP50 / Precision / Recall:")
    for c in CLASSES:
        v = per_class_map50.get(c, float("nan"))
        p = per_class_p.get(c, float("nan"))
        r = per_class_r.get(c, float("nan"))
        bar = "#" * int(v * 30)
        print(f"    {c:<16}  mAP50={v:.4f}  P={p:.4f}  R={r:.4f}  {bar}")

    # Save detection_results.csv (one row per experiment)
    det_row = {
        "experiment"    : "E1",
        "model"         : "YOLOv8n",
        "timestamp"     : datetime.now().strftime("%Y-%m-%d %H:%M"),
        "map50"         : round(map50,    4),
        "map50_95"      : round(map5095,  4),
        "precision"     : round(prec,     4),
        "recall"        : round(recall,   4),
        "fps"           : fps,
        "model_size_mb" : model_size_mb,
        "epochs_trained": 100,
        "image_size"    : 640,
        "feature_dim"   : YOLO_DIM,
        **{f"map50_{c}":  round(per_class_map50.get(c, float("nan")), 4) for c in CLASSES},
        **{f"prec_{c}":   round(per_class_p.get(c,     float("nan")), 4) for c in CLASSES},
        **{f"rec_{c}":    round(per_class_r.get(c,     float("nan")), 4) for c in CLASSES},
        **{f"f1_{c}":     round(per_class_f1.get(c,    float("nan")), 4) for c in CLASSES},
    }
    pd.DataFrame([det_row]).to_csv(RESULTS_DIR / "detection_results.csv", index=False)

    # Save per-class metrics CSV (matches E2/E3 format + detection columns)
    pd.DataFrame([{
        "experiment": "E1",
        "class"     : c,
        "map50"     : round(per_class_map50.get(c, float("nan")), 4),
        "precision" : round(per_class_p.get(c,     float("nan")), 4),
        "recall"    : round(per_class_r.get(c,     float("nan")), 4),
        "f1"        : round(per_class_f1.get(c,    float("nan")), 4),
    } for c in CLASSES]).to_csv(RESULTS_DIR / "E1_per_class_metrics.csv", index=False)

    # Save per-image predictions from test split (YOLO label files)
    test_val_dir = RESULTS_DIR / "yolo_runs" / "E1_test_val"
    pred_rows = []
    label_dir = test_val_dir / "labels"
    img_dir   = dataset_root / "test" / "images"
    if label_dir.exists() and img_dir.exists():
        for lbl_file in sorted(label_dir.glob("*.txt")):
            img_file = img_dir / (lbl_file.stem + ".jpg")
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        conf   = float(parts[4]) if len(parts) > 4 else float("nan")
                        pred_rows.append({
                            "image"    : lbl_file.stem,
                            "pred_class": CLASSES[cls_id] if cls_id < len(CLASSES) else cls_id,
                            "confidence": round(conf, 4),
                            "x_center" : round(float(parts[1]), 4),
                            "y_center" : round(float(parts[2]), 4),
                            "width"    : round(float(parts[3]), 4),
                            "height"   : round(float(parts[4]) if len(parts) > 4 else 0.0, 4),
                        })
    if pred_rows:
        pd.DataFrame(pred_rows).to_csv(RESULTS_DIR / "E1_yolo_predictions.csv", index=False)
        print(f"  Saved {len(pred_rows)} detections to E1_yolo_predictions.csv")
    else:
        print("  [warn] No prediction label files found — skipping predictions CSV")

    # Copy figures
    for png in run_dir.glob("*.png"):
        shutil.copy(png, RESULTS_DIR / "figures" / png.name)
    if test_val_dir.exists():
        for png in test_val_dir.glob("*.png"):
            shutil.copy(png, RESULTS_DIR / "figures" / f"test_{png.name}")

    # ── 4. Extract YOLO backbone features from crops (for E11) ────────────────
    print("\n[4/4] Extracting YOLO backbone features from crops (for E11) ...")
    CROPS_DIR = Path("/tmp/dataset_crops")
    if not (CROPS_DIR / "train").exists():
        print("  Extracting crops zip ...")
        with zipfile.ZipFile("/crops_vol/dataset_crops.zip", "r") as zf:
            zf.extractall("/tmp")
    else:
        print("  Crops already extracted.")

    # Hook on SPPF (layer 9 in YOLOv8n) to capture 256-dim backbone output
    feat_model  = YOLO(str(best_weights))
    nn_backbone = feat_model.model
    nn_backbone.eval()

    _buf = {}
    def _sppf_hook(module, inp, out):
        _buf["sppf"] = out.detach()

    _handle = nn_backbone.model[9].register_forward_hook(_sppf_hook)

    # YOLO preprocessing: resize + ToTensor (divide by 255, no ImageNet norm)
    yolo_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    class CropDS(Dataset):
        def __init__(self, items):     # items: list of (Path, label_int)
            self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, i):
            path, label = self.items[i]
            img = PILImage.open(path).convert("RGB")
            return yolo_tf(img), label

    def collect_items(split):
        items = []
        split_dir = CROPS_DIR / split
        for cls_name in sorted(split_dir.iterdir()):
            if cls_name.name not in CLASS_TO_IDX:
                continue
            label = CLASS_TO_IDX[cls_name.name]
            for img_path in sorted(cls_name.glob("*.jpg")):
                items.append((img_path, label))
        return items

    def extract_split_feats(split, batch_sz=256):
        items = collect_items(split)
        dl    = DataLoader(CropDS(items), batch_size=batch_sz,
                           num_workers=4, pin_memory=True)
        X_list, y_list = [], []
        for imgs, labs in dl:
            imgs = imgs.cuda()
            with torch.no_grad():
                nn_backbone(imgs)
            gap = _buf["sppf"].mean(dim=[2, 3]).cpu().numpy()  # [B, 256]
            X_list.append(gap)
            y_list.append(labs.numpy())
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        return X, y, [str(p) for p, _ in items]

    feat_meta = {}
    for split in ["train", "val", "test"]:
        print(f"  {split} ...", end="", flush=True)
        t0     = time.time()
        X, y, paths = extract_split_feats(split)
        elapsed = time.time() - t0
        np.save(RESULTS_DIR / "features" / f"yolo_{split}_X.npy", X)
        np.save(RESULTS_DIR / "features" / f"yolo_{split}_y.npy", y)
        feat_meta[split] = {"shape": list(X.shape), "dtype": str(X.dtype),
                            "n_samples": int(X.shape[0]),
                            "feature_dim": int(X.shape[1]),
                            "extraction_s": round(elapsed, 1)}
        print(f"  {X.shape}  ({elapsed:.0f}s)")

    _handle.remove()

    manifest = {
        "experiment"    : "E1",
        "extractor"     : "YOLOv8n SPPF (layer 9) + GlobalAveragePool",
        "feature_dim"   : YOLO_DIM,
        "model_weights" : "yolov8n_E1_best.pt",
        "note"          : "Row-aligned with classical_train_clean_X.npy and deep_train_X.npy (original crops, no augmentation)",
        "splits"        : feat_meta,
        "created"       : datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open(RESULTS_DIR / "features" / "yolo_feature_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    RESULTS_VOL.commit()
    print("\nAll outputs saved to Modal Volume.")
    return det_row


@app.local_entrypoint()
def main():
    import pandas as pd

    print("=" * 60)
    print("E1  YOLOv8-nano  [Modal H100 GPU]")
    print("=" * 60)

    # Upload dataset if not already in Volume
    zip_src = REPO_ROOT / "data" / "processed" / "dataset_split_70_15_15.zip"
    assert zip_src.exists(), f"Dataset zip not found: {zip_src}"
    entries = {e.path.lstrip("/") for e in DATA_VOL.listdir("/")}
    if "dataset_split_70_15_15.zip" not in entries:
        size_mb = zip_src.stat().st_size / 1e6
        print(f"Uploading dataset ({size_mb:.0f} MB) to e1-yolo-data Volume ...")
        with DATA_VOL.batch_upload() as batch:
            batch.put_file(str(zip_src), "dataset_split_70_15_15.zip")
        print("  Uploaded.")
    else:
        print("  dataset_split_70_15_15.zip already in Volume — skipping upload.")

    print("\nLaunching run_e1 on Modal H100 (~50 min) ...")
    row = run_e1.remote()

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("E1  YOLOv8n  |  RESULTS")
    print("=" * 60)
    print(f"  mAP@0.5      : {row['map50']:.4f}")
    print(f"  mAP@0.5:0.95 : {row['map50_95']:.4f}")
    print(f"  Precision    : {row['precision']:.4f}")
    print(f"  Recall       : {row['recall']:.4f}")
    print(f"  FPS          : {row['fps']}")
    print(f"  Model size   : {row['model_size_mb']} MB")
    print("\n  Per-class mAP50:")
    for c in CLASSES:
        v   = row.get(f"map50_{c}", float("nan"))
        bar = "#" * int(v * 40) if v == v else ""
        print(f"    {c:<16} {v:.4f}  {bar}")

    # ── Download all outputs ───────────────────────────────────────────────────
    vol = modal.Volume.from_name("e1-yolo-results")

    local_yolo  = REPO_ROOT / "models" / "yolo"
    local_figs  = REPO_ROOT / "figures" / "yolo"
    local_met   = REPO_ROOT / "results" / "metrics"
    local_pred  = REPO_ROOT / "results" / "predictions"
    local_feat  = REPO_ROOT / "data" / "processed" / "features"
    local_arc   = REPO_ROOT / "archive" / "colab_e1_v2"
    for d in [local_yolo, local_figs, local_met, local_pred, local_feat, local_arc]:
        d.mkdir(parents=True, exist_ok=True)

    downloads = {
        "yolo_runs/E1_baseline_v2/weights/best.pt": local_yolo / "yolov8n_E1_best.pt",
        "yolo_runs/E1_baseline_v2/weights/last.pt": local_yolo / "yolov8n_E1_last.pt",
        "yolo_runs/E1_baseline_v2/results.csv"    : local_arc  / "results.csv",
        "detection_results.csv"                    : local_met  / "detection_results.csv",
        "E1_per_class_metrics.csv"                 : local_met  / "E1_per_class_metrics.csv",
        "E1_yolo_predictions.csv"                  : local_pred / "E1_yolo_predictions.csv",
        "features/yolo_train_X.npy"               : local_feat / "yolo_train_X.npy",
        "features/yolo_train_y.npy"               : local_feat / "yolo_train_y.npy",
        "features/yolo_val_X.npy"                 : local_feat / "yolo_val_X.npy",
        "features/yolo_val_y.npy"                 : local_feat / "yolo_val_y.npy",
        "features/yolo_test_X.npy"                : local_feat / "yolo_test_X.npy",
        "features/yolo_test_y.npy"                : local_feat / "yolo_test_y.npy",
        "features/yolo_feature_manifest.json"     : local_feat / "yolo_feature_manifest.json",
    }

    print("\nDownloading outputs ...")
    for remote, local in downloads.items():
        print(f"  {remote} ...", end="", flush=True)
        try:
            with open(local, "wb") as f:
                for chunk in vol.read_file(remote):
                    f.write(chunk)
            print(f"  {local.stat().st_size / 1024:.0f} KB")
        except Exception as e:
            print(f"  SKIP ({e})")

    # Download all PNG figures
    try:
        for entry in vol.listdir("/figures/"):
            fname = Path(entry.path).name
            if not fname.endswith(".png"):
                continue
            local_f = local_figs / fname
            print(f"  figures/{fname} ...", end="", flush=True)
            with open(local_f, "wb") as f:
                for chunk in vol.read_file(f"figures/{fname}"):
                    f.write(chunk)
            print(f"  {local_f.stat().st_size / 1024:.0f} KB")
    except Exception as e:
        print(f"  figures listing: {e}")

    print(f"\nE1 complete.")
    print(f"  Best weights  → {local_yolo / 'yolov8n_E1_best.pt'}")
    print(f"  YOLO features → {local_feat}/yolo_{{train,val,test}}_X.npy  (shape: N×{YOLO_DIM})")
