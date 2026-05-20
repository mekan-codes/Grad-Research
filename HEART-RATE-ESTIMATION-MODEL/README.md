# Smartwatch PPG Heart-Rate Experiment Pipeline

This project compares heart-rate estimation from smartwatch PPG before and after motion-artifact filtering:

1. Raw PPG -> heart-rate estimation
2. Tiny-PPG artifact filtering -> cleaned PPG -> heart-rate estimation

Tiny-PPG is treated as a motion-artifact detector for PPG signals. It is not used as a heart-rate estimator. KID-PPG is treated as the preferred heart-rate estimator because its upstream project provides a pretrained PPG heart-rate inference model. If KID-PPG is not installed or cannot accept the current signal, this project falls back to a classical signal-processing estimator.

No video input and no rPPG-Toolbox components are used.

## Project Layout

```text
README.md
requirements.txt
.gitignore
src/
  __init__.py
  data_loader.py
  tiny_ppg_filter.py
  hr_estimator.py
  metrics.py
  utils.py
scripts/
  run_experiment.py
data/
  input/
  output/
checkpoints/
  README.md
experiments/
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the lightweight baseline dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional KID-PPG install:

```powershell
python -m pip install kid-ppg
```

KID-PPG currently brings TensorFlow-related version constraints, so it is intentionally not required by `requirements.txt`. When it is absent or incompatible, the runner prints:

```text
KID-PPG unavailable. Using classical signal-processing HR estimator.
```

Optional Tiny-PPG install material:

1. Put Tiny-PPG `model.py` in `checkpoints/tiny_ppg/model.py`, or pass `--tiny-ppg-dir path\to\Tiny-PPG`.
2. Put a trained Tiny-PPG PyTorch checkpoint in `checkpoints/tiny_ppg/`, or pass `--tiny-ppg-checkpoint path\to\checkpoint.pth`.
3. Install PyTorch separately if needed:

```powershell
python -m pip install torch
```

The Tiny-PPG GitHub repository provides model and training code, but it does not behave like a simple pip inference package in this pipeline. If usable local Tiny-PPG code and weights are not found, the runner prints:

```text
Tiny-PPG unavailable. Running raw PPG baseline only.
```

## Input Data

Put smartwatch CSV files in `data/input/`.

The runner detects columns by name:

- PPG: `ppg`, `PPG`, `signal`, `value`, `bvp`, `green`, `ir`, `red`
- Time: `time`, `timestamp`, `t`, `seconds`
- Ground-truth HR: `hr`, `HR`, `heart_rate`, `bpm`, `label`

If a time column is present, the sampling rate is estimated from it. Otherwise, the command-line `--fs` value is used.

## Run

```powershell
python scripts/run_experiment.py --input data/input/sample.csv --fs 100
```

Results are saved to:

```text
data/output/results.json
```

Useful optional arguments:

```powershell
python scripts/run_experiment.py `
  --input data/input/sample.csv `
  --fs 100 `
  --output data/output/results.json `
  --tiny-ppg-dir checkpoints/tiny_ppg `
  --tiny-ppg-checkpoint checkpoints/tiny_ppg/tiny_ppg.pth `
  --artifact-threshold 0.5
```

## What The Pipeline Does

1. Loads the CSV.
2. Detects the PPG column.
3. Detects sampling rate from time if possible, otherwise uses `--fs`.
4. Estimates HR from raw PPG with KID-PPG if possible.
5. Falls back to classical signal processing if KID-PPG is unavailable or incompatible.
6. Attempts Tiny-PPG artifact detection if local Tiny-PPG code and checkpoint are available.
7. Masks noisy samples as `NaN`.
8. Estimates HR again from the cleaned PPG.
9. If ground-truth HR exists, calculates:
   - `raw_mae`
   - `cleaned_mae`
   - `raw_rmse`
   - `cleaned_rmse`
   - `improvement_bpm`
   - `percent_signal_removed`
10. Writes JSON results to `data/output/results.json`.

## Fallback HR Estimator

The classical estimator is the guaranteed laptop-runnable path. For each analysis window it:

1. Removes or interpolates around NaNs.
2. Detrends the signal.
3. Normalizes the signal.
4. Bandpass filters around 0.7 to 3.0 Hz.
5. Estimates BPM using FFT power in the heart-rate band.
6. Uses time-domain peak detection as a consistency check.

## Interpreting Results

`raw.bpm` is the median heart-rate estimate from the unfiltered PPG. `cleaned.bpm` is the median estimate after Tiny-PPG artifact masking, when Tiny-PPG is available.

When a ground-truth HR column is present:

- Lower `raw_mae` and `raw_rmse` are better for the raw baseline.
- Lower `cleaned_mae` and `cleaned_rmse` are better for the Tiny-PPG cleaned path.
- Positive `improvement_bpm` means cleaning reduced MAE.
- `percent_signal_removed` reports how much PPG was masked by Tiny-PPG.

If no ground-truth HR column exists, error metrics are written as `null`, but HR estimates are still saved.

## Limitations

- KID-PPG expects 32 Hz PPG windows and its package has strict TensorFlow-era dependency constraints.
- Tiny-PPG requires local model code and a trained checkpoint. This project does not train Tiny-PPG by default.
- Masking noisy samples can reduce available signal. If too much signal is removed, cleaned HR estimates may be sparse or unavailable.
- The fallback estimator is useful for a baseline, but it is not a replacement for validated clinical-grade HR estimation.
- Input column detection is name-based. Rename columns or extend `src/data_loader.py` if your CSV uses unusual names.

## Upstream References

- KID-PPG: https://github.com/esl-epfl/KID-PPG
- Tiny-PPG: https://github.com/SZTU-wearable/Tiny-PPG

