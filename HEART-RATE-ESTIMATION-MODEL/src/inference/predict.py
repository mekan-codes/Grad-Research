from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data_loader import load_ppg_csv
from src.data.preprocessing import config_from_mapping, preprocess_ppg_window
from src.models.full_framework import build_full_framework
from src.utils.checkpoint import load_checkpoint


def load_framework_for_inference(checkpoint_path: str | Path, device: str | torch.device = "cpu"):
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    if payload.get("model_type") != "full_framework":
        raise ValueError("predict_single expects a full_framework checkpoint")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint does not contain the training config needed to load TinyPPG")
    framework = build_full_framework(config, device=device)
    framework.hr_model.load_state_dict(payload["hr_model_state_dict"])
    framework.eval()
    return framework, config


def predict_csv(
    input_csv: str | Path,
    checkpoint_path: str | Path,
    fallback_fs: float = 64.0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    framework, config = load_framework_for_inference(checkpoint_path, device=device)
    loaded = load_ppg_csv(input_csv, fallback_fs=fallback_fs)
    data_cfg = config.get("data", {})
    fs = float(loaded.fs or data_cfg.get("sampling_rate_hz", fallback_fs))
    window_sec = float(data_cfg.get("window_sec", 30.0))
    n_samples = min(loaded.ppg.size, int(round(window_sec * fs)))
    if n_samples <= 0:
        raise ValueError("Input CSV contains no PPG samples")

    preprocess_config = config_from_mapping(config.get("preprocessing", {}), fs)
    ppg = preprocess_ppg_window(loaded.ppg[:n_samples], preprocess_config)
    tensor = torch.tensor(ppg, dtype=torch.float32, device=device).unsqueeze(0)
    valid_mask = torch.ones_like(tensor, dtype=torch.bool)

    with torch.no_grad():
        output = framework(tensor, valid_mask)

    predicted = float(output["predicted_hr"].detach().cpu().reshape(-1)[0])
    crop_meta = output["cropping_metadata"][0]
    artifact_mask = output["artifact_mask"].detach().cpu().numpy().reshape(-1)
    label = None
    absolute_error = None
    if loaded.ground_truth_hr is not None:
        hr = loaded.ground_truth_hr[:n_samples]
        finite = np.isfinite(hr)
        if finite.any():
            label = float(np.nanmedian(hr[finite]))
            absolute_error = abs(predicted - label)

    return {
        "predicted_hr_bpm": predicted,
        "label_hr_bpm": label,
        "absolute_error_bpm": absolute_error,
        "cropped_percent": crop_meta.get("percent_removed"),
        "artifact_summary": {
            "artifact_samples": int(artifact_mask.sum()),
            "total_samples": int(artifact_mask.size),
            "artifact_percent": float(artifact_mask.mean() * 100.0) if artifact_mask.size else 0.0,
            "number_of_removed_segments": crop_meta.get("number_of_removed_segments"),
        },
        "cropping_metadata": crop_meta,
    }

