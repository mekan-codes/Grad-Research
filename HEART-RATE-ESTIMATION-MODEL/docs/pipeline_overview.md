# TinyPPG + HR Estimation Pipeline

This repository now has two paths:

- The older baseline scripts remain in `src/data_loader.py`, `src/hr_estimator.py`, and `src/tiny_ppg_filter.py`.
- The trainable modular pipeline lives under `src/data`, `src/artifact`, `src/models`, `src/training`, and `src/inference`.

## Flow

Raw PPG window -> preprocessing -> frozen TinyPPG artifact detector -> artifact mask -> cropping -> variable-length clean PPG -> HR estimator -> bpm.

TinyPPG is loaded from `../Tiny-PPG-master` by default. The local model returns a segmentation tensor in `output["seg"]`; the adapter treats that as artifact probability and thresholds it into a noisy mask. If a future TinyPPG checkpoint returns a different output shape, `src/artifact/artifact_detector.py` raises an error instead of silently guessing.

## Training

`train_hr.py` trains only the HR estimator on prepared or synthetic windows.

`train_framework.py` runs TinyPPG and cropping during training, but the optimizer is built only from `framework.hr_model.parameters()`. TinyPPG parameters are frozen with:

```python
for param in tinyppg.parameters():
    param.requires_grad = False
tinyppg.eval()
```

## Cropping

The cropper uses `artifact_mask=True` to mean noisy. In default `crop` mode, noisy regions are removed and separate clean regions are concatenated. It returns the clean signal plus metadata:

- original length
- cropped length
- percent removed
- number of removed artifact segments
- all-clean/all-noisy flags

## Batching

Cropped windows have variable length. `src/data/collate.py` pads each batch to the longest clean signal and returns:

- `padded_ppg`
- `valid_mask`
- `hr_label`
- `metadata`

The HR models use `valid_mask` so padded samples do not affect pooling or attention.

## Commands

```bash
pip install -r requirements.txt
python scripts/run_artifact_crop_demo.py --input data/input/sample.csv
python scripts/train_hr_estimator.py --config configs/debug_cpu.yaml
python scripts/train_full_framework.py --config configs/debug_cpu.yaml
python scripts/evaluate_model.py --checkpoint checkpoints/full_framework/best_framework.pth
python -m pytest tests
```

On this Windows machine, the local interpreter may be `.venv\Scripts\python.exe` if `python` is not on PATH.

