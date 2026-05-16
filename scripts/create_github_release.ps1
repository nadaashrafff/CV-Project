# create_github_release.ps1
# Run this AFTER pushing the repo to GitHub.
# Requires GitHub CLI: winget install --id GitHub.cli  &&  gh auth login
#
# Usage:
#   cd "c:\Users\zaki\Desktop\aly eui\Computer_Vision_Latest"
#   .\scripts\create_github_release.ps1

$REPO_ROOT = Split-Path $PSScriptRoot -Parent

$FILES = @(
    # Dataset zips
    "$REPO_ROOT\data\processed\dataset_split_70_15_15.zip",
    "$REPO_ROOT\data\processed\dataset_crops.zip",

    # Classical feature arrays (E2 / E4-E10)
    "$REPO_ROOT\data\processed\features\classical_train_clean_X.npy",
    "$REPO_ROOT\data\processed\features\classical_train_clean_y.npy",
    "$REPO_ROOT\data\processed\features\classical_val_X.npy",
    "$REPO_ROOT\data\processed\features\classical_val_y.npy",
    "$REPO_ROOT\data\processed\features\classical_test_X.npy",
    "$REPO_ROOT\data\processed\features\classical_test_y.npy",

    # Deep feature arrays (E3 / E4-E10)
    "$REPO_ROOT\data\processed\features\deep_train_X.npy",
    "$REPO_ROOT\data\processed\features\deep_train_y.npy",
    "$REPO_ROOT\data\processed\features\deep_val_X.npy",
    "$REPO_ROOT\data\processed\features\deep_val_y.npy",
    "$REPO_ROOT\data\processed\features\deep_test_X.npy",
    "$REPO_ROOT\data\processed\features\deep_test_y.npy",

    # YOLO backbone feature arrays (E1 / E11)
    "$REPO_ROOT\data\processed\features\yolo_train_X.npy",
    "$REPO_ROOT\data\processed\features\yolo_train_y.npy",
    "$REPO_ROOT\data\processed\features\yolo_val_X.npy",
    "$REPO_ROOT\data\processed\features\yolo_val_y.npy",
    "$REPO_ROOT\data\processed\features\yolo_test_X.npy",
    "$REPO_ROOT\data\processed\features\yolo_test_y.npy",

    # Trained models
    "$REPO_ROOT\models\classifiers\random_forest_E2.pkl",
    "$REPO_ROOT\models\classifiers\svm_E3.pkl",
    "$REPO_ROOT\models\yolo\yolov8n_E1_best.pt"
)

Write-Host "Checking files..." -ForegroundColor Cyan
$missing = @()
foreach ($f in $FILES) {
    if (Test-Path $f) {
        $mb = [math]::Round((Get-Item $f).Length / 1MB, 1)
        Write-Host "  OK  $([System.IO.Path]::GetFileName($f))  ($mb MB)"
    } else {
        Write-Host "  MISSING  $f" -ForegroundColor Red
        $missing += $f
    }
}

if ($missing.Count -gt 0) {
    Write-Host "`n$($missing.Count) file(s) missing — fix before creating release." -ForegroundColor Red
    exit 1
}

Write-Host "`nCreating GitHub Release v1.0 ..." -ForegroundColor Green
$fileArgs = $FILES | ForEach-Object { $_ }

gh release create v1.0 @fileArgs `
    --title "v1.0 — E1/E2/E3 Baselines Complete" `
    --notes "## Release v1.0

All baseline experiments (E1, E2, E3) complete. Feature arrays ready for E4-E11.

### Contents
- Dataset splits and crops (zip files)
- Pre-extracted feature arrays: classical (252-dim), deep (1280-dim), YOLO (256-dim)
- Trained models: Random Forest E2, SVM E3, YOLOv8n E1

### Results Summary
| Exp | Model | Macro-F1 / mAP@0.5 |
|-----|-------|---------------------|
| E1  | YOLOv8n | mAP50=0.456, mAP50-95=0.316 |
| E2  | RF + Classical (252-dim) | Macro-F1=0.648 |
| E3  | SVM + EfficientNetB0 (1280-dim) | Macro-F1=0.785 |

See README for full metrics and instructions."

Write-Host "`nRelease created. Teammates can now download with:" -ForegroundColor Green
Write-Host "  gh release download v1.0 --dir . --repo <REPO_URL>" -ForegroundColor Yellow
