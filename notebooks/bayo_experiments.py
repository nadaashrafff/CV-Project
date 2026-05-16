"""
Bayo — Experiments E7, E8, E9, E10
Project: Compression-Aware Multiscale Feature Fusion for Waste Classification

Outputs saved to: new cv/models/, new cv/results/, new cv/figures/
Feature arrays read from: CV-Project/data/processed/features/
"""

import sys, time, json, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score, top_k_accuracy_score,
)

# ── Constants ────────────────────────────────────────────────────────────────
SEED        = 42
CLASSES     = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
N_CLASSES   = 6

np.random.seed(SEED)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent           # new cv/
FEAT_DIR = Path("/Users/omarbayoumi/Documents/CV-Project/data/processed/features")
CLF_DIR  = Path("/Users/omarbayoumi/Documents/CV-Project/models/classifiers")

MDL_DIR  = BASE / "models";   MDL_DIR.mkdir(exist_ok=True)
RES_DIR  = BASE / "results";  RES_DIR.mkdir(exist_ok=True)
FIG_DIR  = BASE / "figures";  FIG_DIR.mkdir(exist_ok=True)

METRICS_CSV = RES_DIR / "classification_results.csv"


# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob, exp_id, model_name, feature_dim,
                    extraction_s, n_train, classification_s, n_test):
    infer_ms   = (classification_s / n_test) * 1000
    per_f1     = f1_score(y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0)
    per_prec   = precision_score(y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0)
    per_rec    = recall_score(y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0)
    auc = float("nan")
    try:
        auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        pass
    row = {
        "experiment"        : exp_id,
        "model"             : model_name,
        "feature_dim"       : feature_dim,
        "accuracy"          : round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1"          : round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1"       : round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "balanced_accuracy" : round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "mcc"               : round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "cohen_kappa"       : round(float(cohen_kappa_score(y_true, y_pred)), 4),
        "auc_roc_macro"     : round(auc, 4) if not np.isnan(auc) else float("nan"),
        "inference_ms_per_crop": round(infer_ms, 4),
        **{f"f1_{CLASSES[i]}":   round(float(v), 4) for i, v in enumerate(per_f1)},
        **{f"prec_{CLASSES[i]}": round(float(v), 4) for i, v in enumerate(per_prec)},
        **{f"rec_{CLASSES[i]}":  round(float(v), 4) for i, v in enumerate(per_rec)},
    }
    return row


def save_row(row, experiment_id):
    if METRICS_CSV.exists():
        df = pd.read_csv(METRICS_CSV)
        df = df[df["experiment"] != experiment_id]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(METRICS_CSV, index=False)


def save_cm_fig(cm, title, fname):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(FIG_DIR / fname, dpi=150); plt.close()


# ── Load feature arrays ───────────────────────────────────────────────────────
print("Loading features...")
X_c_train = np.load(FEAT_DIR / "classical_train_clean_X.npy")   # (45177, 252)
y_train    = np.load(FEAT_DIR / "classical_train_clean_y.npy")
X_c_val   = np.load(FEAT_DIR / "classical_val_X.npy")           # (9935, 252)
y_val      = np.load(FEAT_DIR / "classical_val_y.npy")
X_c_test  = np.load(FEAT_DIR / "classical_test_X.npy")          # (10553, 252)
y_test     = np.load(FEAT_DIR / "classical_test_y.npy")

X_d_train = np.load(FEAT_DIR / "deep_train_X.npy")              # (45177, 1280)
X_d_val   = np.load(FEAT_DIR / "deep_val_X.npy")
X_d_test  = np.load(FEAT_DIR / "deep_test_X.npy")

print(f"Train {X_c_train.shape}  Val {X_c_val.shape}  Test {X_c_test.shape}")


# ════════════════════════════════════════════════════════════════════════════
# E7 — Feature Selection Fusion (top-200 from 1532-dim fused vector)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("E7: Feature Selection Fusion")
print("="*60)

X_fused_train = np.concatenate([X_c_train, X_d_train], axis=1)  # (45177, 1532)
X_fused_val   = np.concatenate([X_c_val,   X_d_val  ], axis=1)
X_fused_test  = np.concatenate([X_c_test,  X_d_test ], axis=1)
FUSED_DIM = X_fused_train.shape[1]   # 1532

print(f"Fused dim: {FUSED_DIM}  (classical 0-251 | deep 252-1531)")
print("Step 1 — Selector RF for feature importance...")
t0 = time.time()
rf_selector = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=SEED
)
rf_selector.fit(X_fused_train, y_train)
importances = rf_selector.feature_importances_
top_idx     = np.argsort(importances)[::-1][:200]   # top-200, descending importance
selector_s  = time.time() - t0
print(f"  Done in {selector_s:.1f}s")

n_classical = int((top_idx < 252).sum())
n_deep      = int((top_idx >= 252).sum())
print(f"  Classical selected: {n_classical}/200 ({100*n_classical/200:.0f}%)")
print(f"  Deep selected:      {n_deep}/200 ({100*n_deep/200:.0f}%)")

np.save(MDL_DIR / "e7_top200_feature_indices.npy", top_idx)

X_sel_tr = X_fused_train[:, top_idx]
X_sel_va = X_fused_val[:,   top_idx]
X_sel_te = X_fused_test[:,  top_idx]

print("Step 2 — Final RF on 200 selected features...")
t0 = time.time()
rf_e7 = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=SEED
)
rf_e7.fit(X_sel_tr, y_train)
print(f"  Trained in {time.time()-t0:.1f}s")
joblib.dump(rf_e7, MDL_DIR / "e7_feature_selection_rf.pkl")

# Evaluate
y_val_pred_e7  = rf_e7.predict(X_sel_va)
y_val_proba_e7 = rf_e7.predict_proba(X_sel_va)
val_f1_e7      = f1_score(y_val, y_val_pred_e7, average="macro", zero_division=0)

t0 = time.time()
y_pred_e7  = rf_e7.predict(X_sel_te)
y_proba_e7 = rf_e7.predict_proba(X_sel_te)
clf_s_e7   = time.time() - t0

test_f1_e7 = f1_score(y_test, y_pred_e7, average="macro", zero_division=0)
print(f"  val  macro-F1: {val_f1_e7:.4f}")
print(f"  test macro-F1: {test_f1_e7:.4f}  |  acc: {accuracy_score(y_test, y_pred_e7):.4f}")

np.save(RES_DIR / "e7_predictions.npy",   y_pred_e7)
np.save(RES_DIR / "e7_probabilities.npy", y_proba_e7)

# Feature group bar chart
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(["Classical\n(252-dim)", "Deep\n(1280-dim)"],
              [n_classical, n_deep], color=["#4C72B0", "#DD8452"], edgecolor="white")
for bar, cnt in zip(bars, [n_classical, n_deep]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{cnt} ({100*cnt/200:.0f}%)", ha="center", va="bottom", fontsize=11)
ax.set_ylim(0, 230)
ax.set_ylabel("Features selected (out of 200)")
ax.set_title("E7 — Feature Group Breakdown")
plt.tight_layout(); plt.savefig(FIG_DIR / "E7_feature_group_analysis.png", dpi=150); plt.close()

cm_e7 = confusion_matrix(y_test, y_pred_e7, labels=list(range(N_CLASSES)))
save_cm_fig(cm_e7, "E7 — Feature Selection (200-dim)", "E7_confusion_matrix.png")

row_e7 = compute_metrics(y_test, y_pred_e7, y_proba_e7, "E7", "RF_FeatureSelection_200",
                          200, selector_s, len(y_train), clf_s_e7, len(y_test))
row_e7.update({"val_macro_f1": round(float(val_f1_e7), 4),
               "val_accuracy": round(float(accuracy_score(y_val, y_val_pred_e7)), 4),
               "feature_group_classical": n_classical, "feature_group_deep": n_deep,
               "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})
save_row(row_e7, "E7")
print("  Saved.")


# ════════════════════════════════════════════════════════════════════════════
# E8 — Average Voting
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("E8: Average Voting (50/50 RF + SVM)")
print("="*60)

rf_e2      = joblib.load(CLF_DIR / "random_forest_E2.pkl")
svm_bundle = joblib.load(CLF_DIR / "svm_E3.pkl")
svm_clf    = svm_bundle["clf"]
svm_scaler = svm_bundle["scaler"]

X_d_val_s  = svm_scaler.transform(X_d_val)
X_d_test_s = svm_scaler.transform(X_d_test)

print("Generating probabilities from RF and SVM...")
rf_proba_val   = rf_e2.predict_proba(X_c_val)
rf_proba_test  = rf_e2.predict_proba(X_c_test)
svm_proba_val  = svm_clf.predict_proba(X_d_val_s)
svm_proba_test = svm_clf.predict_proba(X_d_test_s)

t0 = time.time()
avg_proba_test = (rf_proba_test + svm_proba_test) / 2.0
y_pred_e8      = np.argmax(avg_proba_test, axis=1)
clf_s_e8       = time.time() - t0

avg_proba_val = (rf_proba_val + svm_proba_val) / 2.0
y_val_pred_e8 = np.argmax(avg_proba_val, axis=1)
val_f1_e8     = f1_score(y_val, y_val_pred_e8, average="macro", zero_division=0)
test_f1_e8    = f1_score(y_test, y_pred_e8, average="macro", zero_division=0)

print(f"  val  macro-F1: {val_f1_e8:.4f}")
print(f"  test macro-F1: {test_f1_e8:.4f}  |  acc: {accuracy_score(y_test, y_pred_e8):.4f}")

np.save(RES_DIR / "e8_predictions.npy",            y_pred_e8)
np.save(RES_DIR / "e8_probabilities.npy",          avg_proba_test)
np.save(RES_DIR / "e8_rf_probabilities_test.npy",  rf_proba_test)
np.save(RES_DIR / "e8_svm_probabilities_test.npy", svm_proba_test)

cm_e8 = confusion_matrix(y_test, y_pred_e8, labels=list(range(N_CLASSES)))
save_cm_fig(cm_e8, "E8 — Average Voting (50/50)", "E8_confusion_matrix.png")

row_e8 = compute_metrics(y_test, y_pred_e8, avg_proba_test, "E8", "AverageVoting_RF+SVM",
                          252 + 1280, 0.0, len(y_train), clf_s_e8, len(y_test))
row_e8.update({"val_macro_f1": round(float(val_f1_e8), 4),
               "val_accuracy": round(float(accuracy_score(y_val, y_val_pred_e8)), 4),
               "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})
save_row(row_e8, "E8")
print("  Saved.")


# ════════════════════════════════════════════════════════════════════════════
# E9 — Weighted Voting (grid search on val macro-F1)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("E9: Weighted Voting (grid search)")
print("="*60)

STEPS   = 19
weights = np.linspace(0.05, 0.95, STEPS)
records, best_f1, best_w_rf = [], -1.0, 0.5

for w_rf in weights:
    fused = w_rf * rf_proba_val + (1 - w_rf) * svm_proba_val
    preds = np.argmax(fused, axis=1)
    acc   = float(accuracy_score(y_val, preds))
    mf1   = float(f1_score(y_val, preds, average="macro", zero_division=0))
    records.append({"w_rf": round(w_rf, 4), "w_svm": round(1 - w_rf, 4),
                    "val_accuracy": round(acc, 4), "val_macro_f1": round(mf1, 4)})
    if mf1 > best_f1:
        best_f1, best_w_rf = mf1, w_rf

best_w_svm = 1.0 - best_w_rf
print(f"  Best  w_rf={best_w_rf:.4f}  w_svm={best_w_svm:.4f}  val_macro_f1={best_f1:.4f}")

df_ws = pd.DataFrame(records)
df_ws.to_csv(RES_DIR / "e9_weight_search_results.csv", index=False)

with open(RES_DIR / "e9_optimal_weights.json", "w") as f:
    json.dump({"w_rf": round(best_w_rf, 4), "w_svm": round(best_w_svm, 4),
               "val_macro_f1": round(best_f1, 4)}, f, indent=2)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df_ws["w_rf"], df_ws["val_macro_f1"], marker="o", label="macro-F1")
ax.plot(df_ws["w_rf"], df_ws["val_accuracy"], marker="s", ls="--", label="accuracy")
ax.axvline(best_w_rf, color="red", ls=":", lw=1.5, label=f"best w_rf={best_w_rf:.2f}")
ax.set_xlabel("w_rf (RF weight)   [w_svm = 1 – w_rf]")
ax.set_ylabel("Validation score")
ax.set_title("E9 — Weight Grid Search on Validation Set")
ax.legend(); plt.tight_layout()
plt.savefig(FIG_DIR / "E9_weight_search.png", dpi=150); plt.close()

# Apply best weights to test set ONCE
t0 = time.time()
w_proba_test  = best_w_rf * rf_proba_test + best_w_svm * svm_proba_test
y_pred_e9     = np.argmax(w_proba_test, axis=1)
clf_s_e9      = time.time() - t0
test_f1_e9    = f1_score(y_test, y_pred_e9, average="macro", zero_division=0)

print(f"  test macro-F1: {test_f1_e9:.4f}  |  acc: {accuracy_score(y_test, y_pred_e9):.4f}")

np.save(RES_DIR / "e9_predictions.npy",   y_pred_e9)
np.save(RES_DIR / "e9_probabilities.npy", w_proba_test)

cm_e9 = confusion_matrix(y_test, y_pred_e9, labels=list(range(N_CLASSES)))
save_cm_fig(cm_e9, f"E9 — Weighted Voting (w_rf={best_w_rf:.2f})", "E9_confusion_matrix.png")

row_e9 = compute_metrics(y_test, y_pred_e9, w_proba_test, "E9",
                          f"WeightedVoting_RF+SVM_wrf{best_w_rf:.2f}",
                          252 + 1280, 0.0, len(y_train), clf_s_e9, len(y_test))
row_e9.update({"val_macro_f1": round(best_f1, 4),
               "optimal_w_rf": round(best_w_rf, 4),
               "optimal_w_svm": round(best_w_svm, 4),
               "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})
save_row(row_e9, "E9")
print("  Saved.")


# ════════════════════════════════════════════════════════════════════════════
# E10 — Attention / Gating Fusion (PyTorch)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("E10: Attention Fusion (PyTorch gating network)")
print("="*60)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("  PyTorch not installed — skipping E10.")

if HAS_TORCH:
    torch.manual_seed(SEED)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {DEVICE}")

    # Normalise (fit on train only)
    sc_c = StandardScaler().fit(X_c_train)
    sc_d = StandardScaler().fit(X_d_train)
    joblib.dump(sc_c, MDL_DIR / "e10_scaler_classical.pkl")
    joblib.dump(sc_d, MDL_DIR / "e10_scaler_deep.pkl")

    Xc_tr = sc_c.transform(X_c_train).astype(np.float32)
    Xc_va = sc_c.transform(X_c_val).astype(np.float32)
    Xc_te = sc_c.transform(X_c_test).astype(np.float32)
    Xd_tr = sc_d.transform(X_d_train).astype(np.float32)
    Xd_va = sc_d.transform(X_d_val).astype(np.float32)
    Xd_te = sc_d.transform(X_d_test).astype(np.float32)

    class AttentionFusion(nn.Module):
        def __init__(self, c_dim=252, d_dim=1280, h=128, n_cls=6, drop=0.3):
            super().__init__()
            self.gate = nn.Sequential(
                nn.Linear(c_dim + d_dim, h), nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(h, 2), nn.Softmax(dim=1),
            )
            self.c_proj = nn.Linear(c_dim, h)
            self.d_proj = nn.Linear(d_dim, h)
            self.head   = nn.Linear(h, n_cls)

        def forward(self, xc, xd):
            g      = self.gate(torch.cat([xc, xd], dim=1))   # (B, 2)
            fused  = g[:, 0:1] * self.c_proj(xc) + g[:, 1:2] * self.d_proj(xd)
            return self.head(fused)

    model     = AttentionFusion().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.from_numpy(Xc_tr), torch.from_numpy(Xd_tr), torch.LongTensor(y_train)
    )
    loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    history  = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_f1_e10, patience_cnt, best_state = -1.0, 0, None
    PATIENCE = 10

    print("  Training...")
    t_train = time.time()
    for epoch in range(1, 101):
        model.train()
        total = 0.0
        for xc, xd, yt in loader:
            xc, xd, yt = xc.to(DEVICE), xd.to(DEVICE), yt.to(DEVICE)
            loss = criterion(model(xc, xd), yt)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
        avg_loss = total / len(loader)

        model.eval()
        with torch.no_grad():
            logits_v = model(torch.from_numpy(Xc_va).to(DEVICE),
                             torch.from_numpy(Xd_va).to(DEVICE))
            val_loss = criterion(logits_v, torch.LongTensor(y_val).to(DEVICE)).item()
            val_preds = logits_v.argmax(1).cpu().numpy()
        val_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0))

        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_f1)
        scheduler.step(val_f1)

        if val_f1 > best_f1_e10:
            best_f1_e10  = val_f1
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}  loss={avg_loss:.4f}  val_f1={val_f1:.4f}  lr={optimizer.param_groups[0]['lr']:.1e}")
        if patience_cnt >= PATIENCE:
            print(f"    Early stop at epoch {epoch}")
            break

    print(f"  Training done in {time.time()-t_train:.0f}s  |  best val macro-F1: {best_f1_e10:.4f}")
    model.load_state_dict(best_state)

    # Save model
    e10_path = MDL_DIR / "e10_attention_model.pt"
    torch.save({"model_state_dict": model.state_dict(),
                "c_dim": 252, "d_dim": 1280, "h": 128, "n_cls": 6}, e10_path)
    model_mb = e10_path.stat().st_size / 1e6

    with open(RES_DIR / "e10_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="train"); ax1.plot(history["val_loss"], label="val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("E10 — Loss"); ax1.legend()
    ax2.plot(history["val_macro_f1"], color="green")
    ax2.axhline(0.7848, color="red", ls="--", lw=1, label="E3 baseline")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val macro-F1"); ax2.set_title("E10 — Val Macro-F1"); ax2.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "E10_training_curves.png", dpi=150); plt.close()

    # Test evaluation
    model.eval()
    with torch.no_grad():
        xc_te_t = torch.from_numpy(Xc_te).to(DEVICE)
        xd_te_t = torch.from_numpy(Xd_te).to(DEVICE)
        t0 = time.time()
        logits_te = model(xc_te_t, xd_te_t)
        clf_s_e10 = time.time() - t0
        proba_te  = torch.softmax(logits_te, dim=1).cpu().numpy()
        preds_te  = np.argmax(proba_te, axis=1)

    test_f1_e10 = f1_score(y_test, preds_te, average="macro", zero_division=0)
    print(f"  test macro-F1: {test_f1_e10:.4f}  |  acc: {accuracy_score(y_test, preds_te):.4f}")

    np.save(RES_DIR / "e10_predictions.npy",   preds_te)
    np.save(RES_DIR / "e10_probabilities.npy", proba_te)

    # Attention weight extraction
    attn_out = []
    def _hook(m, i, o): attn_out.append(o.detach().cpu().numpy())
    h = model.gate[-1].register_forward_hook(_hook)
    with torch.no_grad():
        model(xc_te_t, xd_te_t)
    h.remove()
    attn_weights = np.vstack(attn_out)
    np.save(RES_DIR / "e10_attention_weights.npy", attn_weights)

    records_a = []
    for cls_id, cls_name in enumerate(CLASSES):
        mask = (y_test == cls_id)
        avg  = attn_weights[mask].mean(axis=0)
        records_a.append({"class": cls_name, "avg_classical_w": round(float(avg[0]), 4),
                           "avg_deep_w": round(float(avg[1]), 4), "n_samples": int(mask.sum())})
    df_attn = pd.DataFrame(records_a)
    df_attn.to_csv(RES_DIR / "e10_attention_by_class.csv", index=False)
    print("  Attention weights per class:")
    print(df_attn.to_string(index=False))

    # Attention heatmap
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(df_attn[["avg_classical_w", "avg_deep_w"]].values,
                annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=["Classical", "Deep"], yticklabels=CLASSES,
                vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("E10 — Average Attention Weights per Class")
    plt.tight_layout(); plt.savefig(FIG_DIR / "E10_attention_heatmap.png", dpi=150); plt.close()

    # Confusion matrix
    cm_e10 = confusion_matrix(y_test, preds_te, labels=list(range(N_CLASSES)))
    save_cm_fig(cm_e10, "E10 — Attention Fusion", "E10_confusion_matrix.png")

    row_e10 = compute_metrics(y_test, preds_te, proba_te, "E10", "AttentionFusion_PyTorch",
                               128, 0.0, len(y_train), clf_s_e10, len(y_test))
    row_e10.update({"val_macro_f1": round(best_f1_e10, 4), "model_size_mb": round(model_mb, 4),
                    "epochs_trained": len(history["train_loss"]),
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})
    save_row(row_e10, "E10")
    print("  Saved.")


# ════════════════════════════════════════════════════════════════════════════
# Final summary table
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY — Bayo Experiments vs Baselines")
print("="*60)
df = pd.read_csv(METRICS_CSV)
cols = ["experiment", "model", "feature_dim", "accuracy", "macro_f1",
        "balanced_accuracy", "inference_ms_per_crop"]
show = df[df["experiment"].isin(["E7", "E8", "E9", "E10"])][cols]
print(show.sort_values("macro_f1", ascending=False).to_string(index=False))
print(f"\nAll outputs saved to: {BASE}")
