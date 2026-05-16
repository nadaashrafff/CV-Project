"""Patch colab_yolo_e1_gpu.ipynb with corrected training hyperparameters."""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "colab_yolo_e1_gpu.ipynb"

PATCHES = {

"cell-train": """\
# ── STEP 3: Train YOLOv8-nano on T4 ─────────────────────────────────────────
from ultralytics import YOLO

SEED = 42
model = YOLO("yolov8n.pt")
model.train(
    data       = str(DATA_YAML),
    epochs     = 120,           # more room to converge
    imgsz      = 640,           # standard YOLO resolution (was 416 — too small)
    batch      = 16,
    patience   = 25,
    project    = str(COLAB_TRAIN),
    name       = "E1_baseline_v2",
    seed       = SEED,
    exist_ok   = True,
    freeze     = 10,            # keep backbone frozen — dataset too small for full fine-tune
    device     = 0,
    verbose    = True,
    # Learning rate — low LR is critical when backbone is frozen
    lr0        = 0.001,         # was 0.01 — too high for frozen-backbone fine-tune
    lrf        = 0.01,          # cosine decay to lr0*lrf = 0.00001
    # Augmentation
    hsv_h=0.015, hsv_s=0.7,  hsv_v=0.4,
    degrees=10.0, translate=0.1, scale=0.5,
    perspective=0.001,
    flipud=0.0,  fliplr=0.5,
    mosaic=1.0,  mixup=0.05,
    copy_paste=0.0,             # was 0.1 — requires instance masks, bbox-only dataset
    close_mosaic=15,            # disable mosaic for final 15 epochs to stabilise
    conf=0.25,   iou=0.45,
)
TRAIN_RUN = COLAB_TRAIN / "E1_baseline_v2"
print(f"\\nTraining complete: {TRAIN_RUN}")
""",

"cell-eval": """\
# ── STEP 4: Evaluate on test set ─────────────────────────────────────────────
TRAIN_RUN  = COLAB_TRAIN / "E1_baseline_v2"
BEST_PT    = TRAIN_RUN / "weights" / "best.pt"

best_model   = YOLO(str(BEST_PT))
test_metrics = best_model.val(
    data=str(DATA_YAML), split="test",
    imgsz=640, conf=0.25, iou=0.45, verbose=True,
)
rd = test_metrics.results_dict
print("\\nTest metrics:")
for k, v in rd.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
""",

"cell-metrics": """\
# ── STEP 6: Build all result files ───────────────────────────────────────────
CLASSES = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]

# Per-class mAP
class_indices    = list(test_metrics.box.ap_class_index)
per_class_ap50   = list(test_metrics.box.ap50)
per_class_ap5095 = list(test_metrics.box.maps)
ap50_by_cls      = {c: float("nan") for c in CLASSES}
ap5095_by_cls    = {c: float("nan") for c in CLASSES}
for idx, a50, a5095 in zip(class_indices, per_class_ap50, per_class_ap5095):
    ap50_by_cls[CLASSES[idx]]   = round(float(a50),   4)
    ap5095_by_cls[CLASSES[idx]] = round(float(a5095), 4)

training_log = pd.read_csv(TRAIN_RUN / "results.csv")

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
    "image_size"    : 640,
    "dataset_split" : "70/15/15",
    **{f"mAP50_{c}":   ap50_by_cls[c]   for c in CLASSES},
    **{f"mAP5095_{c}": ap5095_by_cls[c] for c in CLASSES},
}

# Predictions CSV — stream to avoid OOM
pred_records = []
for result in best_model.predict([str(p) for p in test_images],
                                  conf=0.25, iou=0.45, verbose=False, stream=True):
    img_name = Path(result.path).name
    for box in result.boxes:
        cid = int(box.cls.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        pred_records.append({"image": img_name, "class_id": cid,
                              "class_name": CLASSES[cid],
                              "confidence": round(float(box.conf.item()), 4),
                              "x1": round(x1,1), "y1": round(y1,1),
                              "x2": round(x2,1), "y2": round(y2,1)})

print(f"Predictions: {len(pred_records)}")
""",

}

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

patched = 0
for cell in nb["cells"]:
    cid = cell.get("id")
    if cid in PATCHES:
        cell["source"] = PATCHES[cid]
        cell["outputs"] = []          # clear stale outputs
        patched += 1

if patched != len(PATCHES):
    raise RuntimeError(f"Expected {len(PATCHES)} patches, applied {patched}")

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Patched {patched} cells in {NB_PATH.name}")
print("Changes:")
print("  cell-train  : imgsz 416->640, lr0 0.01->0.001, copy_paste 0.1->0.0, close_mosaic=15, epochs=120")
print("  cell-eval   : imgsz 416->640, TRAIN_RUN -> E1_baseline_v2")
print("  cell-metrics: image_size 416->640")
