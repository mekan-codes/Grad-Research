# Baseline Folder

This folder is for baseline experiments that do not use TinyPPG.

The main baseline is:

```text
Raw PPG CSV
-> window into fixed-length segments
-> preprocess PPG
-> train HR estimator
-> predict BPM
-> compare prediction to HR label
```

This is compared against the TinyPPG pipeline in `../HEART-RATE-ESTIMATION-MODEL`:

```text
Raw PPG CSV
-> window into fixed-length segments
-> preprocess PPG
-> frozen TinyPPG artifact detector
-> crop noisy samples
-> train HR estimator
-> predict BPM
-> compare prediction to HR label
```

## Dataset

The real prepared dataset is expected at:

```text
../HEART-RATE-ESTIMATION-MODEL/data/input/prepared
```

It currently contains subject CSVs such as `S1.csv` through `S15.csv`.

Each CSV is expected to have:

```text
time, ppg, hr, acc_x, acc_y, acc_z
```

The baseline uses `time`, `ppg`, and `hr`. Accelerometer columns are not used by the current HR model.

## Run A Small Baseline Smoke Test

From `C:\Users\user\Desktop\Grad Research\baseline-folder`:

```powershell
..\HEART-RATE-ESTIMATION-MODEL\.venv\Scripts\python.exe scripts\run_raw_loso.py --smoke-only --max-folds 1 --output-dir runs\raw_loso_smoke
```

This creates tiny synthetic subject CSVs and proves the baseline pipeline runs end to end.

## Run Baseline LOSO On Real Subjects

```powershell
..\HEART-RATE-ESTIMATION-MODEL\.venv\Scripts\python.exe scripts\run_raw_loso.py --output-dir runs\raw_loso
```

The script discovers prepared subjects automatically and runs:

```text
Fold S1: train on S2..S15, test on S1
Fold S2: train on S1,S3..S15, test on S2
...
```

## Outputs

Each fold writes:

```text
runs/raw_loso/fold_S1/
  config_used.yaml
  predictions.csv
  metrics/raw_metrics.json
  checkpoints/raw/
```

The experiment root writes:

```text
runs/raw_loso/
  fold_summary.csv
  aggregate_summary.json
  summary.md
```

## How To Compare With TinyPPG

Run the raw baseline from this folder, then run the full TinyPPG comparison from the main model folder:

```powershell
..\HEART-RATE-ESTIMATION-MODEL\.venv\Scripts\python.exe ..\HEART-RATE-ESTIMATION-MODEL\scripts\run_loso_experiment.py --config ..\HEART-RATE-ESTIMATION-MODEL\configs\loso.yaml --output-dir ..\HEART-RATE-ESTIMATION-MODEL\runs\loso
```

Use:

- raw baseline MAE/RMSE from `baseline-folder/runs/raw_loso/fold_summary.csv`
- TinyPPG comparison from `HEART-RATE-ESTIMATION-MODEL/runs/loso/fold_summary.csv`

TinyPPG is useful only if the TinyPPG-cropped path improves held-out HR MAE/RMSE without removing too much signal.
