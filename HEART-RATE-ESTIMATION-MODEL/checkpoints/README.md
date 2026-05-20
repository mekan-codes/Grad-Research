# Checkpoints

Place optional model assets here.

## KID-PPG

KID-PPG is normally installed from pip:

```powershell
python -m pip install kid-ppg
```

The upstream package includes its own pretrained weights. You usually do not need to put KID-PPG files in this folder.

## Tiny-PPG

Tiny-PPG is used only for motion-artifact detection. To enable it without training inside this project, provide:

```text
checkpoints/tiny_ppg/model.py
checkpoints/tiny_ppg/<trained-checkpoint>.pth
```

You can also pass paths explicitly:

```powershell
python scripts/run_experiment.py `
  --input data/input/sample.csv `
  --fs 100 `
  --tiny-ppg-dir checkpoints/tiny_ppg `
  --tiny-ppg-checkpoint checkpoints/tiny_ppg/tiny_ppg.pth
```

If Tiny-PPG files or weights are missing, the experiment still runs the raw PPG baseline.

