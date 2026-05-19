# ── Cell 0: Environment detection (Colab OR local) ───────────────────────────
import sys, os
try:
    from google.colab import drive
    IN_COLAB = True
    drive.mount('/content/drive', force_remount=False)
    REPO_ROOT = '/content/drive/MyDrive/CV/repo'
    assert os.path.isdir(REPO_ROOT), (
        f'Repo not found at {REPO_ROOT}. '
        'Clone with: !git clone <REPO_URL> /content/drive/MyDrive/CV/repo'
    )
except ImportError:
    IN_COLAB = False
    # Running locally — resolve repo root from this notebook's location
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

print(f'✅ Environment : {"Google Colab" if IN_COLAB else "Local"}')
print(f'✅ Repo root   : {REPO_ROOT}')
assert os.path.isdir(os.path.join(REPO_ROOT, 'src')), f'src/ not found under {REPO_ROOT}'

# ── Cell 1: Install any missing packages ─────────────────────────────────────
# Most are pre-installed in Colab; this is a safety net


# ── Cell 1b: All imports ──────────────────────────────────────────────────────
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120
import seaborn as sns
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, balanced_accuracy_score,
    roc_auc_score, ConfusionMatrixDisplay, matthews_corrcoef
)
from sklearn.utils import shuffle as sk_shuffle

# Sklearn MLP Regressor (fallback for E6 autoencoder since TF is missing)
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network._base import ACTIVATIONS

# Utilities
import joblib
from tqdm.auto import tqdm

# ── Locked seed (must match all other notebooks) ──────────────────────────────
SEED = 42
np.random.seed(SEED)
np.random.seed(SEED)

# ── Class definitions (locked — match Aly's YAML) ─────────────────────────────
CLASSES = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
NUM_CLASSES = len(CLASSES)

print('✅ Imports complete')
print(f'   Seed: {SEED}')
print(f'   Classes: {CLASSES}')

# ── Cell 2: All paths derived from REPO_ROOT ─────────────────────────────────

FEATURES_DIR    = os.path.join(REPO_ROOT, 'data', 'processed', 'features')
MODELS_DIR      = os.path.join(REPO_ROOT, 'models', 'classifiers')
METRICS_DIR     = os.path.join(REPO_ROOT, 'results', 'metrics')
PREDICTIONS_DIR = os.path.join(REPO_ROOT, 'results', 'predictions')
FIGURES_DIR     = os.path.join(REPO_ROOT, 'figures', 'classification')

# Drive backup — only meaningful in Colab; locally falls back to results/ahmed_outputs
if IN_COLAB:
    DRIVE_OUT = '/content/drive/MyDrive/CV/05_early_fusion'
else:
    DRIVE_OUT = os.path.join(REPO_ROOT, 'results', 'ahmed_outputs')

for d in [MODELS_DIR, METRICS_DIR, PREDICTIONS_DIR, FIGURES_DIR, DRIVE_OUT]:
    os.makedirs(d, exist_ok=True)

# ── Verify feature arrays ─────────────────────────────────────────────────────
REQUIRED_FILES = [
    'classical_train_clean_X.npy', 'classical_train_clean_y.npy',
    'classical_val_X.npy',         'classical_val_y.npy',
    'classical_test_X.npy',        'classical_test_y.npy',
    'deep_train_X.npy',            'deep_train_y.npy',
    'deep_val_X.npy',              'deep_val_y.npy',
    'deep_test_X.npy',             'deep_test_y.npy',
]

missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(FEATURES_DIR, f))]
if missing:
    print('❌ Feature arrays not found locally.')
    print('   To use real data: download .npy files from GitHub Release v1.0 into:')
    print(f'   {FEATURES_DIR}')
else:
    print('✅ All feature arrays found')
    print(f'   Location: {FEATURES_DIR}')

print(f'\n   Models     → {MODELS_DIR}')
print(f'   Metrics    → {METRICS_DIR}')
print(f'   Figures    → {FIGURES_DIR}')
print(f'   Outputs    → {DRIVE_OUT}')


# ── Cell 3: Load all feature arrays ──────────────────────────────────────────
# ⚠️  CRITICAL: Use classical_train_CLEAN_X (45177 rows, NOT the 5x-augmented
#     classical_train_X which has 225885 rows and is NOT row-aligned with deep features)

print('Loading feature arrays from Aly\'s E2/E3 outputs...')
t0 = time.time()

# Classical features (E2) — 252-dim
C_train_X = np.load(os.path.join(FEATURES_DIR, 'classical_train_clean_X.npy'))
C_train_y = np.load(os.path.join(FEATURES_DIR, 'classical_train_clean_y.npy'))
C_val_X   = np.load(os.path.join(FEATURES_DIR, 'classical_val_X.npy'))
C_val_y   = np.load(os.path.join(FEATURES_DIR, 'classical_val_y.npy'))
C_test_X  = np.load(os.path.join(FEATURES_DIR, 'classical_test_X.npy'))
C_test_y  = np.load(os.path.join(FEATURES_DIR, 'classical_test_y.npy'))

# Deep features (E3) — 1280-dim EfficientNetB0 global pool
D_train_X = np.load(os.path.join(FEATURES_DIR, 'deep_train_X.npy'))
D_train_y = np.load(os.path.join(FEATURES_DIR, 'deep_train_y.npy'))
D_val_X   = np.load(os.path.join(FEATURES_DIR, 'deep_val_X.npy'))
D_val_y   = np.load(os.path.join(FEATURES_DIR, 'deep_val_y.npy'))
D_test_X  = np.load(os.path.join(FEATURES_DIR, 'deep_test_X.npy'))
D_test_y  = np.load(os.path.join(FEATURES_DIR, 'deep_test_y.npy'))

print(f'✅ Loaded in {time.time()-t0:.1f}s')

# ── Sanity checks ─────────────────────────────────────────────────────────────
assert C_train_X.shape == (45177, 252),  f'Classical train shape mismatch: {C_train_X.shape}'
assert D_train_X.shape == (45177, 1280), f'Deep train shape mismatch: {D_train_X.shape}'
assert C_train_X.shape[0] == D_train_X.shape[0], 'Row count mismatch — arrays not aligned!'
assert np.array_equal(C_train_y, D_train_y), 'Train labels mismatch between classical and deep!'
assert np.array_equal(C_val_y,   D_val_y),   'Val labels mismatch!'
assert np.array_equal(C_test_y,  D_test_y),  'Test labels mismatch!'

# Use classical labels as ground truth (they are identical across modalities)
y_train = C_train_y
y_val   = C_val_y
y_test  = C_test_y

print('\n── Feature array shapes ────────────────────────────────────')
print(f'  Classical train : {C_train_X.shape}  labels: {C_train_y.shape}')
print(f'  Classical val   : {C_val_X.shape}  labels: {C_val_y.shape}')
print(f'  Classical test  : {C_test_X.shape}  labels: {C_test_y.shape}')
print(f'  Deep      train : {D_train_X.shape}  labels: {D_train_y.shape}')
print(f'  Deep      val   : {D_val_X.shape}  labels: {D_val_y.shape}')
print(f'  Deep      test  : {D_test_X.shape}  labels: {D_test_y.shape}')
print(f'\n  Classes (0-5): {CLASSES}')
print(f'  Train label distribution:')
for i, cls in enumerate(CLASSES):
    n = np.sum(y_train == i)
    print(f'    {i} {cls:15s}: {n:6d} ({100*n/len(y_train):.1f}%)')

# ── Cell 4: Fuse & scale ──────────────────────────────────────────────────────

print('Fusing classical + deep features...')

# Raw concatenation — the foundation for E4–E7
X_fused_train = np.hstack([C_train_X, D_train_X])   # (45177, 1532)
X_fused_val   = np.hstack([C_val_X,   D_val_X])     # (9935,  1532)
X_fused_test  = np.hstack([C_test_X,  D_test_X])    # (10553, 1532)

FUSED_DIM = X_fused_train.shape[1]
assert FUSED_DIM == 252 + 1280, f'Unexpected fused dim: {FUSED_DIM}'

print(f'✅ Fused shape: {X_fused_train.shape}  (252 classical + 1280 deep = {FUSED_DIM} total)')

# StandardScaler — fit on train ONLY
print('Fitting StandardScaler on train split...')
scaler_fused = StandardScaler()
X_fused_train_sc = scaler_fused.fit_transform(X_fused_train)
X_fused_val_sc   = scaler_fused.transform(X_fused_val)
X_fused_test_sc  = scaler_fused.transform(X_fused_test)

# Save scaler for reproducibility
joblib.dump(scaler_fused, os.path.join(MODELS_DIR, 'scaler_fused_E4_E7.pkl'))
print('✅ Scaler saved → models/classifiers/scaler_fused_E4_E7.pkl')
print(f'   Train mean range: [{X_fused_train_sc.mean(axis=0).min():.4f}, '
      f'{X_fused_train_sc.mean(axis=0).max():.4f}] (should be near 0)')
print(f'   Train std  range: [{X_fused_train_sc.std(axis=0).min():.4f}, '
      f'{X_fused_train_sc.std(axis=0).max():.4f}] (should be near 1)')

# ── Cell 5: Shared evaluation helpers ────────────────────────────────────────

def evaluate_classifier(name, model_or_pred, X_test, y_test, X_val, y_val,
                        feat_dim, fit_time, inf_time_ms, extra_info=None):
    """
    Compute all classification metrics and return a results dict.
    Accepts either a fitted sklearn model or pre-computed prediction arrays.
    """
    if hasattr(model_or_pred, 'predict'):
        y_pred_test = model_or_pred.predict(X_test)
        y_pred_val  = model_or_pred.predict(X_val)
        try:
            y_prob_test = model_or_pred.predict_proba(X_test)
        except AttributeError:
            y_prob_test = None
    else:
        # pre-computed predictions passed in
        y_pred_test, y_pred_val = model_or_pred
        y_prob_test = None

    acc          = accuracy_score(y_test, y_pred_test)
    macro_f1     = f1_score(y_test, y_pred_test, average='macro')
    weighted_f1  = f1_score(y_test, y_pred_test, average='weighted')
    bal_acc      = balanced_accuracy_score(y_test, y_pred_test)
    mcc          = matthews_corrcoef(y_test, y_pred_test)
    val_macro_f1 = f1_score(y_val, y_pred_val, average='macro')
    per_class_f1 = f1_score(y_test, y_pred_test, average=None, labels=list(range(NUM_CLASSES)))

    auc = None
    if y_prob_test is not None:
        try:
            auc = roc_auc_score(y_test, y_prob_test, multi_class='ovr', average='macro')
        except Exception:
            pass

    result = {
        'experiment':        name,
        'feature_dim':       feat_dim,
        'accuracy':          round(acc, 4),
        'macro_f1':          round(macro_f1, 4),
        'weighted_f1':       round(weighted_f1, 4),
        'balanced_accuracy': round(bal_acc, 4),
        'mcc':               round(mcc, 4),
        'val_macro_f1':      round(val_macro_f1, 4),
        'overfitting_gap':   round(val_macro_f1 - macro_f1, 4),
        'auc_roc_macro':     round(auc, 4) if auc else 'N/A',
        'fit_time_s':        round(fit_time, 1),
        'inference_ms_per_crop': round(inf_time_ms, 2),
    }
    # Add per-class F1
    for i, cls in enumerate(CLASSES):
        result[f'f1_{cls}'] = round(per_class_f1[i], 4)

    if extra_info:
        result.update(extra_info)

    return result, y_pred_test, y_pred_val


def save_per_class_metrics(name, y_pred_test, y_test, out_dir):
    """Save per-class precision/recall/F1 CSV matching Aly's format."""
    report = classification_report(y_test, y_pred_test,
                                   target_names=CLASSES, output_dict=True)
    rows = []
    for cls in CLASSES:
        rows.append({
            'class':     cls,
            'precision': round(report[cls]['precision'], 4),
            'recall':    round(report[cls]['recall'], 4),
            'f1_score':  round(report[cls]['f1-score'], 4),
            'support':   int(report[cls]['support']),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, f'{name}_per_class_metrics.csv')
    df.to_csv(path, index=False)
    return df


def plot_confusion_matrix(name, y_pred_test, y_test, out_dir):
    """Save normalised confusion matrix figure."""
    cm = confusion_matrix(y_test, y_pred_test, normalize='true')
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap='Blues',
              values_format='.2f')
    ax.set_title(f'{name} — Normalised Confusion Matrix', fontsize=13, pad=12)
    plt.tight_layout()
    path = os.path.join(out_dir, f'{name}_confusion_matrix.png')
    fig.savefig(path, dpi=150)
    plt.show()
    return path


def append_to_classification_results(result_dict, metrics_dir):
    """Append this experiment's row to the shared classification_results.csv."""
    csv_path = os.path.join(metrics_dir, 'classification_results.csv')
    new_row = pd.DataFrame([result_dict])
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        # Remove old entry for this experiment if re-running
        existing = existing[existing['experiment'] != result_dict['experiment']]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(csv_path, index=False)
    print(f'   ✅ Saved to {csv_path}')


def save_predictions(name, y_pred_test, y_test, out_dir):
    """Save test-set predictions CSV."""
    df = pd.DataFrame({
        'true_label':  y_test,
        'true_class':  [CLASSES[i] for i in y_test],
        'pred_label':  y_pred_test,
        'pred_class':  [CLASSES[i] for i in y_pred_test],
        'correct':     (y_test == y_pred_test).astype(int),
    })
    path = os.path.join(out_dir, f'{name}_predictions.csv')
    df.to_csv(path, index=False)
    return path


# Track all results in this session
ALL_RESULTS = []

print('✅ Evaluation helpers ready')

# ── E4: Raw Early Fusion — Random Forest ─────────────────────────────────────
print('=' * 60)
print('E4 — Raw Early Fusion (1532-dim → Random Forest)')
print('=' * 60)

# Random Forest with balanced class weights (same config as E2 for fair comparison)
rf_e4 = RandomForestClassifier(
    n_estimators=200,
    max_features='sqrt',
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=SEED,
    n_jobs=-1,
    verbose=0
)

print(f'Training RF on {X_fused_train.shape} fused features (unscaled — RF is scale-invariant)...')
t_fit_start = time.time()
rf_e4.fit(X_fused_train, y_train)
fit_time_e4 = time.time() - t_fit_start
print(f'✅ Training complete in {fit_time_e4:.1f}s')

# Measure inference time on test set
t_inf_start = time.time()
_ = rf_e4.predict(X_fused_test)
inf_time_e4_ms = (time.time() - t_inf_start) / len(X_fused_test) * 1000

# Evaluate
result_e4, pred_e4_test, pred_e4_val = evaluate_classifier(
    'E4_raw_early_fusion', rf_e4,
    X_fused_test, y_test,
    X_fused_val,  y_val,
    feat_dim=FUSED_DIM,
    fit_time=fit_time_e4,
    inf_time_ms=inf_time_e4_ms,
    extra_info={'classifier': 'RandomForest(n=200)', 'compression': 'none'}
)

print(f'\n── E4 Results ──────────────────────────────────────────────')
print(f'  Feature dim    : {FUSED_DIM}')
print(f'  Accuracy       : {result_e4["accuracy"]:.4f}')
print(f'  Macro-F1 (TEST): {result_e4["macro_f1"]:.4f}  ← primary metric')
print(f'  Macro-F1 (VAL) : {result_e4["val_macro_f1"]:.4f}')
print(f'  Overfitting gap: {result_e4["overfitting_gap"]:.4f}')
print(f'  Weighted-F1    : {result_e4["weighted_f1"]:.4f}')
print(f'  Balanced Acc   : {result_e4["balanced_accuracy"]:.4f}')
print(f'  MCC            : {result_e4["mcc"]:.4f}')
print(f'  Fit time       : {fit_time_e4:.1f}s')
print(f'  Inference      : {inf_time_e4_ms:.2f} ms/crop')
print(f'\n  Per-class F1:')
for cls in CLASSES:
    print(f'    {cls:15s}: {result_e4["f1_"+cls]:.4f}')
print()
print('  Baseline comparison:')
print(f'    E2 Classical RF  : Macro-F1 = 0.6476  (252-dim)')
print(f'    E3 Deep SVM      : Macro-F1 = 0.7848  (1280-dim)')
print(f'    E4 Fused RF      : Macro-F1 = {result_e4["macro_f1"]:.4f}  ({FUSED_DIM}-dim)')
delta = result_e4['macro_f1'] - 0.7848
print(f'    Delta vs E3      : {delta:+.4f}')

# ── E4: Save outputs ──────────────────────────────────────────────────────────
print('Saving E4 outputs...')

# Model
model_path_e4 = os.path.join(MODELS_DIR, 'random_forest_E4.pkl')
joblib.dump(rf_e4, model_path_e4)
print(f'  Model     → {model_path_e4}')

# Per-class metrics
df_pc_e4 = save_per_class_metrics('E4', pred_e4_test, y_test, METRICS_DIR)
print(f'  Per-class → {METRICS_DIR}/E4_per_class_metrics.csv')

# Predictions
save_predictions('E4_early_fusion', pred_e4_test, y_test, PREDICTIONS_DIR)
print(f'  Preds     → {PREDICTIONS_DIR}/E4_early_fusion_predictions.csv')

# Confusion matrix
plot_confusion_matrix('E4_raw_early_fusion', pred_e4_test, y_test, FIGURES_DIR)

# Append to shared classification_results.csv
append_to_classification_results(result_e4, METRICS_DIR)

# Drive backup
joblib.dump(rf_e4, os.path.join(DRIVE_OUT, 'random_forest_E4.pkl'))
df_pc_e4.to_csv(os.path.join(DRIVE_OUT, 'E4_per_class_metrics.csv'), index=False)

ALL_RESULTS.append(result_e4)
print('\n✅ E4 complete')

# ── E5: PCA Fusion ────────────────────────────────────────────────────────────
print('=' * 60)
print('E5 — PCA Fusion (1532-dim → PCA(95%) → SVM RBF)')
print('=' * 60)

# PCA — fit on scaled TRAIN only
print('Fitting PCA (95% variance) on scaled train features...')
t0 = time.time()
pca_e5 = PCA(n_components=0.95, svd_solver='full', random_state=SEED)
X_pca_train = pca_e5.fit_transform(X_fused_train_sc)
X_pca_val   = pca_e5.transform(X_fused_val_sc)
X_pca_test  = pca_e5.transform(X_fused_test_sc)
pca_time = time.time() - t0

PCA_DIM = X_pca_train.shape[1]
var_explained = pca_e5.explained_variance_ratio_.sum()

print(f'✅ PCA complete in {pca_time:.1f}s')
print(f'   Components kept  : {PCA_DIM}  (from {FUSED_DIM})')
print(f'   Compression ratio: {FUSED_DIM/PCA_DIM:.1f}×')
print(f'   Variance retained: {var_explained:.4f} ({var_explained*100:.2f}%)')

# Save PCA model
joblib.dump(pca_e5, os.path.join(MODELS_DIR, 'pca_E5.pkl'))
print(f'   PCA model saved  → models/classifiers/pca_E5.pkl')

# Scree-like plot: cumulative variance
cumvar = np.cumsum(pca_e5.explained_variance_ratio_)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(1, len(cumvar)+1), cumvar, lw=2)
ax.axhline(0.95, color='red', ls='--', label='95% threshold')
ax.axvline(PCA_DIM, color='green', ls='--', label=f'{PCA_DIM} components')
ax.set_xlabel('Number of PCA components')
ax.set_ylabel('Cumulative explained variance')
ax.set_title('E5 — PCA Cumulative Explained Variance')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'E5_pca_variance_curve.png'), dpi=150)
plt.show()

# ── E5: SVM (RBF) on PCA features ────────────────────────────────────────────
# Same kernel as E3 for fair comparison; PCA features are already scaled
print(f'Training SVM (RBF) on {X_pca_train.shape} PCA features...')
print('(This may take 10–30 min on CPU, ~3–8 min with Colab GPU/TPU via sklearn)')

svm_e5 = SVC(
    kernel='rbf',
    C=10.0,
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=SEED
)

t_fit_start = time.time()
svm_e5.fit(X_pca_train, y_train)
fit_time_e5 = time.time() - t_fit_start
print(f'✅ SVM training complete in {fit_time_e5:.1f}s')

# Inference time
t_inf_start = time.time()
_ = svm_e5.predict(X_pca_test)
inf_time_e5_ms = (time.time() - t_inf_start) / len(X_pca_test) * 1000

# Evaluate
result_e5, pred_e5_test, pred_e5_val = evaluate_classifier(
    'E5_pca_fusion', svm_e5,
    X_pca_test, y_test,
    X_pca_val,  y_val,
    feat_dim=PCA_DIM,
    fit_time=fit_time_e5,
    inf_time_ms=inf_time_e5_ms,
    extra_info={
        'classifier': 'SVM(RBF,C=10)',
        'compression': f'PCA_95pct',
        'original_dim': FUSED_DIM,
        'pca_components': PCA_DIM,
        'variance_retained': round(float(var_explained), 4)
    }
)

print(f'\n── E5 Results ──────────────────────────────────────────────')
print(f'  Feature dim    : {PCA_DIM}  (from {FUSED_DIM}, {FUSED_DIM/PCA_DIM:.1f}× compression)')
print(f'  Variance kept  : {var_explained:.4f}')
print(f'  Accuracy       : {result_e5["accuracy"]:.4f}')
print(f'  Macro-F1 (TEST): {result_e5["macro_f1"]:.4f}  ← primary metric')
print(f'  Macro-F1 (VAL) : {result_e5["val_macro_f1"]:.4f}')
print(f'  Weighted-F1    : {result_e5["weighted_f1"]:.4f}')
print(f'  Balanced Acc   : {result_e5["balanced_accuracy"]:.4f}')
print(f'  Fit time       : {fit_time_e5:.1f}s')
print(f'  Inference      : {inf_time_e5_ms:.2f} ms/crop')
print(f'\n  Per-class F1:')
for cls in CLASSES:
    print(f'    {cls:15s}: {result_e5["f1_"+cls]:.4f}')
print()
print('  Baseline comparison:')
print(f'    E3 Deep SVM (1280-dim) : Macro-F1 = 0.7848')
print(f'    E4 Raw Fusion (1532)   : Macro-F1 = {result_e4["macro_f1"]:.4f}')
print(f'    E5 PCA Fusion ({PCA_DIM})   : Macro-F1 = {result_e5["macro_f1"]:.4f}')

# ── E5: Save outputs ──────────────────────────────────────────────────────────
print('Saving E5 outputs...')

joblib.dump(svm_e5, os.path.join(MODELS_DIR, 'svm_E5_pca.pkl'))
print(f'  SVM model → models/classifiers/svm_E5_pca.pkl')

df_pc_e5 = save_per_class_metrics('E5', pred_e5_test, y_test, METRICS_DIR)
save_predictions('E5_pca_fusion', pred_e5_test, y_test, PREDICTIONS_DIR)
plot_confusion_matrix('E5_pca_fusion', pred_e5_test, y_test, FIGURES_DIR)
append_to_classification_results(result_e5, METRICS_DIR)

joblib.dump(svm_e5, os.path.join(DRIVE_OUT, 'svm_E5_pca.pkl'))
joblib.dump(pca_e5, os.path.join(DRIVE_OUT, 'pca_E5.pkl'))
df_pc_e5.to_csv(os.path.join(DRIVE_OUT, 'E5_per_class_metrics.csv'), index=False)

ALL_RESULTS.append(result_e5)
print('\n✅ E5 complete')

# ── E6: Build Autoencoder ─────────────────────────────────────────────────────
print('=' * 60)
print('E6 — Autoencoder Fusion (1532-dim → 256-dim bottleneck → SVM)')
print('=' * 60)

AE_INPUT_DIM  = FUSED_DIM   # 1532
AE_HIDDEN_DIM = 512
AE_BOTTLE_DIM = 256

print('Training autoencoder on scaled fused train features...')
print('(Using sklearn MLPRegressor as shallow symmetric autoencoder)')

t_ae_start = time.time()
autoencoder_e6 = MLPRegressor(
    hidden_layer_sizes=(AE_HIDDEN_DIM, AE_BOTTLE_DIM, AE_HIDDEN_DIM),
    activation='relu',
    solver='adam',
    learning_rate_init=1e-3,
    max_iter=100,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=SEED,
    verbose=True
)

autoencoder_e6.fit(X_fused_train_sc, X_fused_train_sc)
ae_train_time = time.time() - t_ae_start
print(f'\n✅ Autoencoder training complete in {ae_train_time:.1f}s')

# Training curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(autoencoder_e6.loss_curve_, label='Train loss')
if hasattr(autoencoder_e6, 'validation_scores_') and autoencoder_e6.validation_scores_:
    ax.plot(autoencoder_e6.validation_scores_, label='Val score')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss/Score')
ax.set_title('E6 — Autoencoder Training Curve')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'E6_autoencoder_training_curve.png'), dpi=150)
plt.close()

# Save autoencoder weights
joblib.dump(autoencoder_e6, os.path.join(MODELS_DIR, 'autoencoder_E6.pkl'))
print('  Autoencoder saved → models/classifiers/autoencoder_E6.pkl')

# ── E6: Extract bottleneck features & train SVM ───────────────────────────────
print('Extracting 256-dim bottleneck features from encoder...')
t0 = time.time()

def encode_features(mlp, X):
    a = X.copy()
    for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        a = a @ W + b
        if i == 1:  # bottleneck is after 2nd layer
            break
        ACTIVATIONS['relu'](a)
    ACTIVATIONS['relu'](a)
    return a

X_ae_train = encode_features(autoencoder_e6, X_fused_train_sc)
X_ae_val   = encode_features(autoencoder_e6, X_fused_val_sc)
X_ae_test  = encode_features(autoencoder_e6, X_fused_test_sc)

print(f'✅ Bottleneck features extracted in {time.time()-t0:.1f}s')
print(f'   Bottleneck shape: {X_ae_train.shape}')

# Save bottleneck arrays for downstream use
np.save(os.path.join(FEATURES_DIR, 'ae_bottleneck_train_X.npy'), X_ae_train)
np.save(os.path.join(FEATURES_DIR, 'ae_bottleneck_val_X.npy'),   X_ae_val)
np.save(os.path.join(FEATURES_DIR, 'ae_bottleneck_test_X.npy'),  X_ae_test)
print('   Bottleneck arrays saved to data/processed/features/')

# Scale bottleneck features before SVM
scaler_ae = StandardScaler()
X_ae_train_sc = scaler_ae.fit_transform(X_ae_train)
X_ae_val_sc   = scaler_ae.transform(X_ae_val)
X_ae_test_sc  = scaler_ae.transform(X_ae_test)
joblib.dump(scaler_ae, os.path.join(MODELS_DIR, 'scaler_ae_E6.pkl'))

# Train SVM on bottleneck
print(f'\nTraining SVM (RBF) on {X_ae_train_sc.shape} bottleneck features...')
svm_e6 = SVC(
    kernel='rbf',
    C=10.0,
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=SEED
)
t_fit_start = time.time()
svm_e6.fit(X_ae_train_sc, y_train)
fit_time_e6 = ae_train_time + (time.time() - t_fit_start)
print(f'✅ SVM training complete')

t_inf_start = time.time()
_ = svm_e6.predict(X_ae_test_sc)
inf_time_e6_ms = (time.time() - t_inf_start) / len(X_ae_test_sc) * 1000

result_e6, pred_e6_test, pred_e6_val = evaluate_classifier(
    'E6_autoencoder_fusion', svm_e6,
    X_ae_test_sc, y_test,
    X_ae_val_sc,  y_val,
    feat_dim=AE_BOTTLE_DIM,
    fit_time=fit_time_e6,
    inf_time_ms=inf_time_e6_ms,
    extra_info={
        'classifier': 'SVM(RBF,C=10)+AE',
        'compression': f'AE_1532→256',
        'original_dim': FUSED_DIM,
        'ae_epochs': autoencoder_e6.n_iter_
    }
)

print(f'\n── E6 Results ──────────────────────────────────────────────')
print(f'  Feature dim    : {AE_BOTTLE_DIM}  (from {FUSED_DIM}, {FUSED_DIM/AE_BOTTLE_DIM:.1f}× compression)')
print(f'  Accuracy       : {result_e6["accuracy"]:.4f}')
print(f'  Macro-F1 (TEST): {result_e6["macro_f1"]:.4f}  ← primary metric')
print(f'  Macro-F1 (VAL) : {result_e6["val_macro_f1"]:.4f}')
print(f'  Weighted-F1    : {result_e6["weighted_f1"]:.4f}')
print(f'  Balanced Acc   : {result_e6["balanced_accuracy"]:.4f}')
print(f'  Fit time       : {fit_time_e6:.1f}s  (AE + SVM)')
print(f'  Inference      : {inf_time_e6_ms:.2f} ms/crop')
print(f'\n  Per-class F1:')
for cls in CLASSES:
    print(f'    {cls:15s}: {result_e6["f1_"+cls]:.4f}')
print()
print('  Compression comparison:')
print(f'    E5 PCA ({PCA_DIM}-dim)   : Macro-F1 = {result_e5["macro_f1"]:.4f}')
print(f'    E6 AE  (256-dim)   : Macro-F1 = {result_e6["macro_f1"]:.4f}')

# ── E6: Save outputs ──────────────────────────────────────────────────────────
print('Saving E6 outputs...')

joblib.dump(svm_e6, os.path.join(MODELS_DIR, 'svm_E6_ae.pkl'))
df_pc_e6 = save_per_class_metrics('E6', pred_e6_test, y_test, METRICS_DIR)
save_predictions('E6_autoencoder_fusion', pred_e6_test, y_test, PREDICTIONS_DIR)
plot_confusion_matrix('E6_autoencoder_fusion', pred_e6_test, y_test, FIGURES_DIR)
append_to_classification_results(result_e6, METRICS_DIR)

joblib.dump(svm_e6, os.path.join(DRIVE_OUT, 'svm_E6_ae.pkl'))
df_pc_e6.to_csv(os.path.join(DRIVE_OUT, 'E6_per_class_metrics.csv'), index=False)

ALL_RESULTS.append(result_e6)
print('\n✅ E6 complete')

# ── E7: Feature Selection Fusion ──────────────────────────────────────────────
print('=' * 60)
print('E7 — Feature Selection Fusion (top-200 features → Random Forest)')
print('=' * 60)

TOP_K = 200

# Step 1: Use E4's already-trained RF to get feature importances
# (RF importances are scale-invariant so we use unscaled X_fused)
print('Extracting feature importances from E4 RF (reusing trained model)...')
importances = rf_e4.feature_importances_

# Get top-200 indices sorted by importance
top_idx = np.argsort(importances)[::-1][:TOP_K]
top_idx_sorted = np.sort(top_idx)  # maintain feature order

# Identify which branch each top feature came from
n_classical = C_train_X.shape[1]   # 252
classical_selected = np.sum(top_idx < n_classical)
deep_selected      = TOP_K - classical_selected

print(f'✅ Top {TOP_K} features selected')
print(f'   From classical (idx 0–{n_classical-1}): {classical_selected} features ({100*classical_selected/TOP_K:.1f}%)')
print(f'   From deep      (idx {n_classical}–{FUSED_DIM-1}): {deep_selected} features ({100*deep_selected/TOP_K:.1f}%)')

# Plot feature importances
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Top 50 importances bar chart
top50_idx  = np.argsort(importances)[::-1][:50]
top50_vals = importances[top50_idx]
colors     = ['#2196F3' if i < n_classical else '#FF9800' for i in top50_idx]
axes[0].bar(range(50), top50_vals, color=colors)
axes[0].set_title('Top-50 Feature Importances (Blue=Classical, Orange=Deep)')
axes[0].set_xlabel('Feature rank')
axes[0].set_ylabel('Importance')

# Cumulative importance for top-200
top200_vals = importances[np.argsort(importances)[::-1][:200]]
axes[1].plot(np.arange(1, 201), np.cumsum(top200_vals) / importances.sum())
axes[1].axhline(0.5, color='red', ls='--', label='50% cumulative importance')
axes[1].set_xlabel('Number of features')
axes[1].set_ylabel('Cumulative importance fraction')
axes[1].set_title('Cumulative Feature Importance (Top 200)')
axes[1].legend()

plt.suptitle('E7 — Feature Selection Analysis', fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'E7_feature_importance.png'), dpi=150)
plt.show()

# Save selected feature indices
np.save(os.path.join(MODELS_DIR, 'E7_top200_feature_indices.npy'), top_idx_sorted)
print(f'   Indices saved → models/classifiers/E7_top200_feature_indices.npy')

# ── E7: Select top-200 features and retrain RF ────────────────────────────────
# Use unscaled features (RF is scale-invariant)
X_sel_train = X_fused_train[:, top_idx_sorted]
X_sel_val   = X_fused_val[:,   top_idx_sorted]
X_sel_test  = X_fused_test[:,  top_idx_sorted]

print(f'Training new RF on top-{TOP_K} selected features: {X_sel_train.shape}')

rf_e7 = RandomForestClassifier(
    n_estimators=200,
    max_features='sqrt',
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=SEED,
    n_jobs=-1,
    verbose=0
)

t_fit_start = time.time()
rf_e7.fit(X_sel_train, y_train)
fit_time_e7 = time.time() - t_fit_start
print(f'✅ Training complete in {fit_time_e7:.1f}s')

t_inf_start = time.time()
_ = rf_e7.predict(X_sel_test)
inf_time_e7_ms = (time.time() - t_inf_start) / len(X_sel_test) * 1000

result_e7, pred_e7_test, pred_e7_val = evaluate_classifier(
    'E7_feature_selection', rf_e7,
    X_sel_test, y_test,
    X_sel_val,  y_val,
    feat_dim=TOP_K,
    fit_time=fit_time_e7,
    inf_time_ms=inf_time_e7_ms,
    extra_info={
        'classifier': 'RandomForest(n=200)',
        'compression': f'feature_selection_top{TOP_K}',
        'original_dim': FUSED_DIM,
        'classical_features_selected': int(classical_selected),
        'deep_features_selected': int(deep_selected)
    }
)

print(f'\n── E7 Results ──────────────────────────────────────────────')
print(f'  Feature dim    : {TOP_K}  (from {FUSED_DIM}, {FUSED_DIM/TOP_K:.1f}× compression)')
print(f'  Classical kept : {classical_selected}/{n_classical} = {100*classical_selected/n_classical:.1f}%')
print(f'  Deep kept      : {deep_selected}/{D_train_X.shape[1]} = {100*deep_selected/D_train_X.shape[1]:.1f}%')
print(f'  Accuracy       : {result_e7["accuracy"]:.4f}')
print(f'  Macro-F1 (TEST): {result_e7["macro_f1"]:.4f}  ← primary metric')
print(f'  Macro-F1 (VAL) : {result_e7["val_macro_f1"]:.4f}')
print(f'  Weighted-F1    : {result_e7["weighted_f1"]:.4f}')
print(f'  Balanced Acc   : {result_e7["balanced_accuracy"]:.4f}')
print(f'  Fit time       : {fit_time_e7:.1f}s')
print(f'  Inference      : {inf_time_e7_ms:.2f} ms/crop')
print(f'\n  Per-class F1:')
for cls in CLASSES:
    print(f'    {cls:15s}: {result_e7["f1_"+cls]:.4f}')

# ── E7: Save outputs ──────────────────────────────────────────────────────────
print('Saving E7 outputs...')

joblib.dump(rf_e7, os.path.join(MODELS_DIR, 'random_forest_E7.pkl'))
df_pc_e7 = save_per_class_metrics('E7', pred_e7_test, y_test, METRICS_DIR)
save_predictions('E7_feature_selection', pred_e7_test, y_test, PREDICTIONS_DIR)
plot_confusion_matrix('E7_feature_selection', pred_e7_test, y_test, FIGURES_DIR)
append_to_classification_results(result_e7, METRICS_DIR)

joblib.dump(rf_e7, os.path.join(DRIVE_OUT, 'random_forest_E7.pkl'))
df_pc_e7.to_csv(os.path.join(DRIVE_OUT, 'E7_per_class_metrics.csv'), index=False)

ALL_RESULTS.append(result_e7)
print('\n✅ E7 complete')

# ── Cell 6: Full Summary Table ────────────────────────────────────────────────
print('=' * 70)
print('SUMMARY — E4 through E7 (Early Fusion Experiments)')
print('=' * 70)

# Include Aly's baselines for comparison
baselines = [
    {'experiment': 'E2_classical_RF  (Aly)',  'feature_dim': 252,   'macro_f1': 0.6476, 'accuracy': 0.7740, 'val_macro_f1': 0.6426},
    {'experiment': 'E3_deep_SVM      (Aly)',  'feature_dim': 1280,  'macro_f1': 0.7848, 'accuracy': 0.8522, 'val_macro_f1': 0.7826},
]

all_rows = baselines + ALL_RESULTS
df_summary = pd.DataFrame(all_rows)

# Print table
cols = ['experiment', 'feature_dim', 'accuracy', 'macro_f1', 'val_macro_f1', 'overfitting_gap',
        'weighted_f1', 'balanced_accuracy', 'fit_time_s', 'inference_ms_per_crop']
cols = [c for c in cols if c in df_summary.columns]
print(df_summary[cols].to_string(index=False))

# Best experiment
our_results = [r for r in ALL_RESULTS]
best = max(our_results, key=lambda x: x['macro_f1'])
print(f'\n🏆 Best Macro-F1 among E4–E7: {best["experiment"]} = {best["macro_f1"]:.4f}')

best_vs_e3 = best['macro_f1'] - 0.7848
print(f'   vs E3 baseline (0.7848): {best_vs_e3:+.4f}')
if best_vs_e3 > 0:
    print('   ✅ Fusion improves over deep-only baseline!')
else:
    print('   ⚠️  Fusion did not improve over deep-only baseline (still expected — see analysis below)')

# ── Cell 6b: Accuracy vs Feature Dimension Plot ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

experiments = ['E2\nClassical', 'E3\nDeep', 'E4\nRaw Fusion', 'E5\nPCA', 'E6\nAutoencoder', 'E7\nFeat Sel']
macro_f1s   = [
    0.6476, 0.7848,
    result_e4['macro_f1'], result_e5['macro_f1'],
    result_e6['macro_f1'], result_e7['macro_f1']
]
feat_dims   = [252, 1280, FUSED_DIM, PCA_DIM, AE_BOTTLE_DIM, TOP_K]
colors      = ['#9E9E9E', '#9E9E9E', '#2196F3', '#FF9800', '#4CAF50', '#9C27B0']

# Bar chart — Macro-F1
bars = axes[0].bar(experiments, macro_f1s, color=colors, edgecolor='black', linewidth=0.5)
axes[0].axhline(0.7848, color='red', ls='--', lw=1.5, label='E3 baseline (0.7848)')
axes[0].set_ylim(0.5, 1.0)
axes[0].set_ylabel('Macro-F1')
axes[0].set_title('Macro-F1 by Experiment')
axes[0].legend(fontsize=9)
for bar, val in zip(bars, macro_f1s):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=8)

# Scatter — Macro-F1 vs Feature Dim (accuracy-efficiency trade-off)
scatter_x = feat_dims
scatter_y = macro_f1s
for i, (x, y, label, color) in enumerate(zip(scatter_x, scatter_y, experiments, colors)):
    axes[1].scatter(x, y, color=color, s=120, zorder=5, edgecolor='black', linewidth=0.5)
    axes[1].annotate(label.replace('\n', ' '), (x, y),
                     xytext=(6, 3), textcoords='offset points', fontsize=7.5)
axes[1].set_xlabel('Feature Dimension')
axes[1].set_ylabel('Macro-F1')
axes[1].set_title('Accuracy–Efficiency Trade-off')
axes[1].axhline(0.7848, color='red', ls='--', lw=1, alpha=0.6, label='E3 baseline')
axes[1].legend(fontsize=9)

plt.suptitle('E4–E7 Early Fusion: Performance Comparison', fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'E4_E7_comparison.png'), dpi=150)
plt.show()
print('✅ Comparison figure saved → figures/classification/E4_E7_comparison.png')

# ── Cell 6c: Per-class F1 heatmap ─────────────────────────────────────────────
per_class_data = {}
for exp_name, y_pred in [
    ('E4_Raw_Fusion', pred_e4_test),
    ('E5_PCA',        pred_e5_test),
    ('E6_AutoEnc',    pred_e6_test),
    ('E7_FeatSel',    pred_e7_test),
]:
    per_class_data[exp_name] = f1_score(y_test, y_pred, average=None,
                                        labels=list(range(NUM_CLASSES)))

df_heatmap = pd.DataFrame(per_class_data, index=CLASSES).T

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='YlOrRd',
            vmin=0.5, vmax=1.0, ax=ax,
            linewidths=0.5, linecolor='white')
ax.set_title('E4–E7 Per-Class F1 Score Heatmap', fontsize=13, pad=10)
ax.set_xlabel('Waste Class')
ax.set_ylabel('Experiment')
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'E4_E7_perclass_f1_heatmap.png'), dpi=150)
plt.show()
print('✅ Per-class heatmap saved → figures/classification/E4_E7_perclass_f1_heatmap.png')

# ── Cell 6d: Final summary CSV + Drive backup ─────────────────────────────────
df_ahmed_summary = pd.DataFrame(ALL_RESULTS)
summary_path = os.path.join(METRICS_DIR, 'E4_E7_summary.csv')
df_ahmed_summary.to_csv(summary_path, index=False)
df_ahmed_summary.to_csv(os.path.join(DRIVE_OUT, 'E4_E7_summary.csv'), index=False)

# Also copy figures to Drive
import shutil
for fname in [
    'E4_raw_early_fusion_confusion_matrix.png',
    'E5_pca_variance_curve.png',
    'E5_pca_fusion_confusion_matrix.png',
    'E6_autoencoder_training_curve.png',
    'E6_autoencoder_fusion_confusion_matrix.png',
    'E7_feature_importance.png',
    'E7_feature_selection_confusion_matrix.png',
    'E4_E7_comparison.png',
    'E4_E7_perclass_f1_heatmap.png',
]:
    src = os.path.join(FIGURES_DIR, fname)
    dst = os.path.join(DRIVE_OUT, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)

print('✅ All outputs saved')
print(f'   Metrics  → {METRICS_DIR}')
print(f'   Models   → {MODELS_DIR}')
print(f'   Figures  → {FIGURES_DIR}')
print(f'   Drive    → {DRIVE_OUT}')

print('\n── Files generated for Bayo (E8-E10) & paper analysis ───────')
print('  No additional files needed from Ahmed for late/attention fusion.')
print('  Bayo uses: deep_train_X.npy, classical_train_clean_X.npy, random_forest_E2.pkl, svm_E3.pkl')
print('  All of these are Aly\'s outputs already available in the repo.')

# ── Cell 7: Structured interpretation ────────────────────────────────────────
print('\n' + '═'*70)
print('  E4–E7 INTERPRETATION (copy into report Section 12: Results)')
print('═'*70)

e4_f1 = result_e4['macro_f1']
e5_f1 = result_e5['macro_f1']
e6_f1 = result_e6['macro_f1']
e7_f1 = result_e7['macro_f1']
e3_f1 = 0.7848

print(f"""
EARLY FUSION RESULTS SUMMARY
─────────────────────────────────────────────────────
  Baseline    E2 Classical RF  : Macro-F1 = 0.6476  (252-dim)
  Baseline    E3 Deep SVM      : Macro-F1 = {e3_f1:.4f}  (1280-dim)
  ─────────────────────────────────────────────────────
  E4 Raw Fusion RF (1532-dim)  : Macro-F1 = {e4_f1:.4f}  delta vs E3 = {e4_f1-e3_f1:+.4f}
  E5 PCA Fusion SVM ({PCA_DIM}-dim) : Macro-F1 = {e5_f1:.4f}  delta vs E3 = {e5_f1-e3_f1:+.4f}
  E6 AE Fusion SVM (256-dim)   : Macro-F1 = {e6_f1:.4f}  delta vs E3 = {e6_f1-e3_f1:+.4f}
  E7 Feat Sel RF  (200-dim)    : Macro-F1 = {e7_f1:.4f}  delta vs E3 = {e7_f1-e3_f1:+.4f}

FEATURE BRANCH DOMINANCE (E7 analysis)
─────────────────────────────────────────────────────
  Classical features in top-200 : {result_e7.get('classical_features_selected', '?')}/{C_train_X.shape[1]}
  Deep features in top-200      : {result_e7.get('deep_features_selected', '?')}/{D_train_X.shape[1]}
  → This shows which modality is more informative for waste classification.

COMPRESSION TRADE-OFF
─────────────────────────────────────────────────────
  E5 PCA reduces 1532→{PCA_DIM} ({1532/PCA_DIM:.1f}× compression)
  E6 AE  reduces 1532→256  ({1532/256:.1f}× compression)
  E7 FS  reduces 1532→200  ({1532/200:.1f}× compression)
""")

print('Note: Paste exact numbers from above into your report results table.')

# ── Cell 8: Final verification ────────────────────────────────────────────────
csv_path = os.path.join(METRICS_DIR, 'classification_results.csv')
if os.path.exists(csv_path):
    df_all = pd.read_csv(csv_path)
    print('classification_results.csv current contents:')
    print(df_all[['experiment', 'feature_dim', 'accuracy', 'macro_f1', 'val_macro_f1']].to_string(index=False))
    
    ahmed_experiments = [r['experiment'] for r in ALL_RESULTS]
    for exp in ahmed_experiments:
        assert exp in df_all['experiment'].values, f'❌ {exp} missing from classification_results.csv!'
    print(f'\n✅ All {len(ahmed_experiments)} Ahmed experiments confirmed in shared CSV')
else:
    print('⚠️  classification_results.csv not found — check METRICS_DIR path')

# List all generated files
print('\n── Generated model files ────────────────────────────────────')
for fname in os.listdir(MODELS_DIR):
    if any(tag in fname for tag in ['E4', 'E5', 'E6', 'E7', 'scaler_fused', 'scaler_ae', 'pca_E5']):
        fpath = os.path.join(MODELS_DIR, fname)
        size_mb = os.path.getsize(fpath) / 1e6
        print(f'  {fname:<45} {size_mb:.1f} MB')

print('\n── Generated metric files ───────────────────────────────────')
for fname in os.listdir(METRICS_DIR):
    if any(tag in fname for tag in ['E4', 'E5', 'E6', 'E7']):
        print(f'  {fname}')

print('\n🎉 Notebook 05 complete — E4, E5, E6, E7 all done!')

