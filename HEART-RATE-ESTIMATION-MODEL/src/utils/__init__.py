"""Shared utility helpers.

This package intentionally mirrors the legacy ``src/utils.py`` helpers so the
older scripts keep working after adding the requested ``src/utils/`` package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_parent_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def finite_fraction(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.isfinite(arr).mean())


def interpolate_nans(values: np.ndarray, min_finite_fraction: float = 0.25) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("empty signal")

    finite = np.isfinite(arr)
    if finite.mean() < min_finite_fraction:
        raise ValueError("too few finite samples")

    if finite.all():
        return arr.astype(float, copy=True)

    idx = np.arange(arr.size)
    filled = np.interp(idx, idx[finite], arr[finite])
    return filled.astype(float, copy=False)


def zscore(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if not np.isfinite(std) or std < eps:
        return arr - mean
    return (arr - mean) / std


def to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return to_serializable(value.tolist())
    if isinstance(value, np.generic):
        return to_serializable(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2)
        handle.write("\n")
    return output_path

