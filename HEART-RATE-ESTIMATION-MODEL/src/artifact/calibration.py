from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from src.artifact.cropper import CropperConfig, crop_artifact_regions


def tensor_stats(tensor: torch.Tensor | np.ndarray) -> dict[str, Any]:
    arr = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    finite = arr[np.isfinite(arr)]
    stats: dict[str, Any] = {
        "shape": list(arr.shape),
        "finite_count": int(finite.size),
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
    }
    if finite.size:
        stats.update(
            {
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
            }
        )
    return stats


def probability_diagnostics(
    probability: torch.Tensor | np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    arr = probability.detach().cpu().numpy() if isinstance(probability, torch.Tensor) else np.asarray(probability)
    finite = arr[np.isfinite(arr)]
    diagnostics = tensor_stats(arr)
    diagnostics["threshold"] = float(threshold)
    diagnostics["percent_above_threshold"] = float(np.mean(finite >= threshold) * 100.0) if finite.size else 0.0
    diagnostics["may_need_sigmoid"] = bool(finite.size and (np.min(finite) < 0.0 or np.max(finite) > 1.0))
    diagnostics["possibly_inverted"] = bool(finite.size and np.mean(finite >= threshold) > 0.8)
    return diagnostics


def calibrate_artifact_thresholds(
    artifact_detector,
    samples: Iterable[dict[str, Any]],
    thresholds: list[float],
    cropper_config: CropperConfig,
    device: str | torch.device = "cpu",
    max_windows: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate crop statistics across thresholds without training."""

    device = torch.device(device)
    per_threshold: list[dict[str, Any]] = []
    buckets: dict[float, dict[str, list[float] | int]] = {
        float(threshold): {"percent_removed": [], "too_short": 0, "windows": 0}
        for threshold in thresholds
    }

    n_windows = 0
    for sample in samples:
        if max_windows is not None and n_windows >= int(max_windows):
            break
        ppg = sample["ppg"]
        tensor = ppg if isinstance(ppg, torch.Tensor) else torch.as_tensor(ppg, dtype=torch.float32)
        tensor = tensor.reshape(1, -1).to(device)
        valid_mask = torch.ones_like(tensor, dtype=torch.bool)
        with torch.no_grad():
            detection = artifact_detector(tensor, valid_mask)
        probability = detection.artifact_probability.squeeze(0).detach().cpu().numpy()
        signal = tensor.squeeze(0).detach().cpu().numpy()

        for threshold in thresholds:
            key = float(threshold)
            mask = probability >= key
            cropped = crop_artifact_regions(signal, mask, cropper_config)
            buckets[key]["percent_removed"].append(float(cropped.metadata.percent_removed))  # type: ignore[index]
            buckets[key]["too_short"] = int(buckets[key]["too_short"]) + int(
                cropped.metadata.cropped_length < cropper_config.min_output_samples
            )
            buckets[key]["windows"] = int(buckets[key]["windows"]) + 1
        n_windows += 1

    for threshold in thresholds:
        key = float(threshold)
        percents = np.asarray(buckets[key]["percent_removed"], dtype=float)  # type: ignore[arg-type]
        windows = int(buckets[key]["windows"])
        if windows == 0:
            row = {
                "threshold": key,
                "windows": 0,
                "mean_percent_removed": None,
                "median_percent_removed": None,
                "percent_windows_0_removed": None,
                "percent_windows_100_removed": None,
                "too_short_cropped_windows": 0,
                "too_short_cropped_window_percent": None,
                "validation_mae": None,
            }
        else:
            row = {
                "threshold": key,
                "windows": windows,
                "mean_percent_removed": float(np.mean(percents)),
                "median_percent_removed": float(np.median(percents)),
                "percent_windows_0_removed": float(np.mean(percents <= 0.0) * 100.0),
                "percent_windows_100_removed": float(np.mean(percents >= 100.0) * 100.0),
                "too_short_cropped_windows": int(buckets[key]["too_short"]),
                "too_short_cropped_window_percent": float(int(buckets[key]["too_short"]) / windows * 100.0),
                "validation_mae": None,
            }
        per_threshold.append(row)

    result = {
        "thresholds": per_threshold,
        "windows_evaluated": n_windows,
        "warnings": threshold_warnings(per_threshold),
    }
    if output_path is not None:
        save_calibration_result(result, output_path)
    return result


def threshold_warnings(
    rows: list[dict[str, Any]],
    mean_removed_above: float = 70.0,
    full_crop_above: float = 20.0,
    too_short_above: float = 20.0,
) -> list[str]:
    warnings: list[str] = []
    for row in rows:
        threshold = row["threshold"]
        mean_removed = row.get("mean_percent_removed")
        full_crop = row.get("percent_windows_100_removed")
        too_short = row.get("too_short_cropped_window_percent")
        if mean_removed is not None and mean_removed > mean_removed_above:
            warnings.append(f"threshold {threshold}: mean removed is high ({mean_removed:.1f}%)")
        if full_crop is not None and full_crop > full_crop_above:
            warnings.append(f"threshold {threshold}: many windows are 100% cropped ({full_crop:.1f}%)")
        if too_short is not None and too_short > too_short_above:
            warnings.append(f"threshold {threshold}: many cropped windows are too short ({too_short:.1f}%)")
    return warnings


def save_calibration_result(result: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path

