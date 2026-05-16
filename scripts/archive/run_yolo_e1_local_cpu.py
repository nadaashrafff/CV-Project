"""
Run the full YOLOv8-nano E1 baseline pipeline locally.

Prerequisites:
    python scripts/run_preprocessing.py   (must complete first)
    pip install ultralytics>=8.0.0

Usage:
    python scripts/run_yolo_e1.py

Saves:
    models/yolo/yolo_baseline_E1.pt           best weights
    models/checkpoints/last/yolo_E1_last.pt   last checkpoint
    results/metrics/detection_results.csv     E1 row (appended/updated)
    results/metrics/E1_per_class_metrics.csv  per-class mAP
    results/predictions/E1_yolo_predictions.csv
    results/logs/training_logs/E1_training_results.csv
    figures/yolo/                             training curves + confusion matrix
"""
import sys, shutil, time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Dependency check: preprocessing must have run ─────────────────────────────
SPLIT_OUT = REPO_ROOT / "data" / "processed" / "dataset_split_70_15_15"
DATA_YAML = SPLIT_OUT / "data.yaml"

if not DATA_YAML.exists():
    print("ERROR: data.yaml not found.")
    print(f"  Expected: {DATA_YAML}")
    print("  Run preprocessing first:  python scripts/run_preprocessing.py")
    sys.exit(1)

from ultralytics import YOLO
import numpy as np
import pandas as pd

from src.utils.seed import set_seed, SEED
from src.utils.paths import CLASSES
from src.utils.logger import get_logger
from src.evaluation.detection_metrics import get_model_size_mb

set_seed()
logger = get_logger("yolo_e1")

# ── Output paths ──────────────────────────────────────────────────────────────
MODELS_YOLO   = REPO_ROOT / "models" / "yolo"
CKPT_LAST     = REPO_ROOT / "models" / "checkpoints" / "last"
RESULTS_DIR   = REPO_ROOT / "results" / "metrics"
PREDS_DIR     = REPO_ROOT / "results" / "predictions"
FIGS_YOLO     = REPO_ROOT / "figures" / "yolo"
LOG_DIR       = REPO_ROOT / "results" / "logs" / "training_logs"
TRAIN_WORK    = REPO_ROOT / "data" / "processed" / "yolo_training"

for d in [MODELS_YOLO, CKPT_LAST, RESULTS_DIR, PREDS_DIR, FIGS_YOLO, LOG_DIR, TRAIN_WORK]:
    d.mkdir(parents=True, exist_ok=True)

REPO_BEST = MODELS_YOLO / "yolo_baseline_E1.pt"
REPO_LAST = CKPT_LAST   / "yolo_E1_last.pt"
TRAIN_RUN = TRAIN_WORK  / "E1_baseline"

# ── Verify split ──────────────────────────────────────────────────────────────
print("=" * 70)
print("YOLOv8-nano E1 Baseline Pipeline")
print("=" * 70)
for sp in ["train", "val", "test"]:
    ni = len(list((SPLIT_OUT / sp / "images").glob("*.jpg")))
    nl = len(list((SPLIT_OUT / sp / "labels").glob("*.txt")))
    flag = "" if ni == nl else "  <- MISMATCH"
    print(f"  {sp:<8}: {ni:>5} images  {nl:>5} labels{flag}")
print(f"  data.yaml: {DATA_YAML}\n")

# ── Train ─────────────────────────────────────────────────────────────────────
print("=" * 70)
print("TRAINING  (100 epochs, patience=20)")
print("=" * 70)

model = YOLO("yolov8n.pt")
model.train(
    data       = str(DATA_YAML),
    epochs     = 120,
    imgsz      = 416,   # kept small for CPU RAM — use Colab for 640
    batch      = 4,     # CPU RAM constraint (7.7 GB total)
    workers    = 0,
    patience   = 25,
    project    = str(TRAIN_WORK),
    name       = "E1_baseline",
    seed       = SEED,
    exist_ok   = True,
    verbose    = True,
    freeze     = 10,    # freeze backbone — dataset too small for full fine-tune
    lr0        = 0.001, # low LR essential when backbone is frozen
    lrf        = 0.01,
    # Augmentation
    hsv_h=0.015, hsv_s=0.7,  hsv_v=0.4,
    degrees=10.0, translate=0.1, scale=0.5,
    perspective=0.001,
    flipud=0.0,  fliplr=0.5,
    mosaic=1.0,  mixup=0.05,
    copy_paste=0.0,     # removed: requires instance masks, bbox-only dataset
    close_mosaic=15,    # disable mosaic for final 15 epochs to stabilise
    conf=0.25,   iou=0.45,
)
print(f"\nTraining complete: {TRAIN_RUN}")

# ── Save weights + training log ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("SAVING WEIGHTS")
print("=" * 70)

BEST_PT = TRAIN_RUN / "weights" / "best.pt"
LAST_PT = TRAIN_RUN / "weights" / "last.pt"

shutil.copy2(BEST_PT, REPO_BEST)
shutil.copy2(LAST_PT, REPO_LAST)
shutil.copy2(TRAIN_RUN / "results.csv", LOG_DIR / "E1_training_results.csv")

print(f"  Best weights  -> {REPO_BEST}")
print(f"  Last weights  -> {REPO_LAST}")
print(f"  Training log  -> {LOG_DIR / 'E1_training_results.csv'}")

# ── Evaluate on test set ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EVALUATION  (test split)")
print("=" * 70)

best_model   = YOLO(str(REPO_BEST))
test_metrics = best_model.val(
    data=str(DATA_YAML), split="test",
    imgsz=416, conf=0.25, iou=0.45, verbose=True,
)
rd = test_metrics.results_dict
print("\nTest metrics:")
for k, v in rd.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# ── FPS benchmark ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FPS BENCHMARK  (100 images)")
print("=" * 70)

test_images = sorted((SPLIT_OUT / "test" / "images").glob("*.jpg"))
for _ in range(2):   # warm-up
    best_model.predict(str(test_images[0]), conf=0.25, iou=0.45, verbose=False)

BENCH_N  = min(100, len(test_images))
times_ms = []
for img_path in test_images[:BENCH_N]:
    t0 = time.perf_counter()
    best_model.predict(str(img_path), conf=0.25, iou=0.45, verbose=False)
    times_ms.append((time.perf_counter() - t0) * 1000)

inference_ms = float(np.mean(times_ms))
fps          = 1000.0 / inference_ms
model_mb     = get_model_size_mb(str(REPO_BEST))
print(f"  Mean inference : {inference_ms:.1f} ms")
print(f"  FPS            : {fps:.1f}")
print(f"  Model size     : {model_mb:.2f} MB")

# ── Save metrics CSV ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SAVING METRICS")
print("=" * 70)

class_indices    = list(test_metrics.box.ap_class_index)
per_class_ap50   = list(test_metrics.box.ap50)
per_class_ap5095 = list(test_metrics.box.maps)

ap50_by_class   = {c: float("nan") for c in CLASSES}
ap5095_by_class = {c: float("nan") for c in CLASSES}
for idx, ap50_v, ap5095_v in zip(class_indices, per_class_ap50, per_class_ap5095):
    ap50_by_class[CLASSES[idx]]   = round(float(ap50_v),   4)
    ap5095_by_class[CLASSES[idx]] = round(float(ap5095_v), 4)

training_log = pd.read_csv(LOG_DIR / "E1_training_results.csv")

detection_row = {
    "experiment"    : "E1",
    "model"         : "yolov8n",
    "timestamp"     : datetime.now().strftime("%Y-%m-%d %H:%M"),
    "mAP50"         : round(float(rd.get("metrics/mAP50(B)",    float("nan"))), 4),
    "mAP50_95"      : round(float(rd.get("metrics/mAP50-95(B)", float("nan"))), 4),
    "precision"     : round(float(rd.get("metrics/precision(B)",float("nan"))), 4),
    "recall"        : round(float(rd.get("metrics/recall(B)",   float("nan"))), 4),
    "fps"           : round(fps, 2),
    "inference_ms"  : round(inference_ms, 2),
    "model_size_mb" : round(model_mb, 3),
    "epochs_trained": len(training_log),
    "image_size"    : 416,
    "dataset_split" : "70/15/15",
    **{f"mAP50_{c}":   ap50_by_class[c]   for c in CLASSES},
    **{f"mAP5095_{c}": ap5095_by_class[c] for c in CLASSES},
}

csv_path = RESULTS_DIR / "detection_results.csv"
if csv_path.exists():
    existing = pd.read_csv(csv_path)
    existing = existing[existing["experiment"] != "E1"]
    df_out   = pd.concat([existing, pd.DataFrame([detection_row])], ignore_index=True)
else:
    df_out = pd.DataFrame([detection_row])
df_out.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")

pd.DataFrame([{"experiment": "E1", "class": c,
               "mAP50": ap50_by_class[c], "mAP50_95": ap5095_by_class[c]}
              for c in CLASSES]).to_csv(RESULTS_DIR / "E1_per_class_metrics.csv", index=False)
print(f"  Saved: E1_per_class_metrics.csv")

# ── Save predictions ──────────────────────────────────────────────────────────
pred_records = []
for result in best_model.predict([str(p) for p in test_images],
                                  conf=0.25, iou=0.45, verbose=False, stream=True):
    img_name = Path(result.path).name
    for box in result.boxes:
        cid = int(box.cls.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        pred_records.append({
            "image": img_name, "class_id": cid, "class_name": CLASSES[cid],
            "confidence": round(float(box.conf.item()), 4),
            "x1": round(x1, 1), "y1": round(y1, 1),
            "x2": round(x2, 1), "y2": round(y2, 1),
        })
preds_path = PREDS_DIR / "E1_yolo_predictions.csv"
pd.DataFrame(pred_records).to_csv(preds_path, index=False)
print(f"  Saved: {preds_path}  ({len(pred_records)} predictions)")

# ── Copy training figures ──────────────────────────────────────────────────────
copied = []
for pattern in ["*.png", "val_batch*.jpg"]:
    for src in TRAIN_RUN.glob(pattern):
        shutil.copy2(src, FIGS_YOLO / src.name)
        copied.append(src.name)
for src in TRAIN_RUN.rglob("confusion_matrix*.png"):
    dst = FIGS_YOLO / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
        copied.append(src.name)
print(f"  Copied {len(copied)} figures -> {FIGS_YOLO}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("E1  YOLOv8-nano  |  FINAL RESULTS")
print("=" * 70)
print(f"  mAP@0.5        : {detection_row['mAP50']:.4f}")
print(f"  mAP@0.5:0.95   : {detection_row['mAP50_95']:.4f}")
print(f"  Precision      : {detection_row['precision']:.4f}")
print(f"  Recall         : {detection_row['recall']:.4f}")
print(f"  FPS            : {detection_row['fps']:.1f}")
print(f"  Inference/img  : {detection_row['inference_ms']:.1f} ms")
print(f"  Model size     : {detection_row['model_size_mb']:.2f} MB")
print(f"  Epochs trained : {detection_row['epochs_trained']}")
print()
print("  Per-class mAP@0.5:")
for cls in CLASSES:
    v   = ap50_by_class[cls]
    bar = "#" * int(v * 40) if v == v else "(nan)"
    print(f"    {cls:<16} {v:.4f}  {bar}")
print()
print("  Saved to repo:")
print(f"    {REPO_BEST.relative_to(REPO_ROOT)}")
print(f"    {REPO_LAST.relative_to(REPO_ROOT)}")
print(f"    results/metrics/detection_results.csv")
print(f"    results/metrics/E1_per_class_metrics.csv")
print(f"    results/predictions/E1_yolo_predictions.csv")
print(f"    results/logs/training_logs/E1_training_results.csv")
print(f"    figures/yolo/  ({len(copied)} files)")
