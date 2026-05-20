from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal


@dataclass
class PreprocessConfig:
    sampling_rate_hz: float
    normalize: str = "zscore"
    detrend: bool = True
    bandpass_enabled: bool = True
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 5.0
    bandpass_order: int = 3


def config_from_mapping(config: dict[str, Any], sampling_rate_hz: float) -> PreprocessConfig:
    bandpass = config.get("bandpass", {}) if isinstance(config.get("bandpass"), dict) else {}
    return PreprocessConfig(
        sampling_rate_hz=float(sampling_rate_hz),
        normalize=str(config.get("normalize", "zscore")),
        detrend=bool(config.get("detrend", True)),
        bandpass_enabled=bool(bandpass.get("enabled", True)),
        bandpass_low_hz=float(bandpass.get("low_hz", 0.5)),
        bandpass_high_hz=float(bandpass.get("high_hz", 5.0)),
        bandpass_order=int(bandpass.get("order", 3)),
    )


def preprocess_ppg_window(values: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Prepare one PPG window for TinyPPG or HR estimation.

    The function avoids dataset-specific assumptions: the sampling rate comes
    from config, and the same path works for 20, 25, or 30 second windows.
    """

    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("Cannot preprocess an empty PPG window")

    x = interpolate_missing(x)
    if config.detrend and x.size >= 3:
        x = signal.detrend(x)
    if config.bandpass_enabled:
        x = bandpass_ppg(
            x,
            fs=config.sampling_rate_hz,
            low_hz=config.bandpass_low_hz,
            high_hz=config.bandpass_high_hz,
            order=config.bandpass_order,
        )
    x = normalize_signal(x, method=config.normalize)
    return x.astype(np.float32, copy=False)


def interpolate_missing(values: np.ndarray, min_finite_fraction: float = 0.25) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(x)
    if x.size == 0:
        raise ValueError("empty signal")
    if finite.mean() < min_finite_fraction:
        raise ValueError("too few finite PPG samples")
    if finite.all():
        return x.astype(float, copy=True)
    idx = np.arange(x.size)
    return np.interp(idx, idx[finite], x[finite]).astype(float, copy=False)


def normalize_signal(values: np.ndarray, method: str = "zscore", eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if method in {"none", "identity", ""}:
        return x
    if method == "zscore":
        mean = float(np.nanmean(x))
        std = float(np.nanstd(x))
        if not np.isfinite(std) or std < eps:
            return x - mean
        return (x - mean) / std
    if method == "minmax":
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
        scale = hi - lo
        if not np.isfinite(scale) or scale < eps:
            return x - lo
        return 2.0 * ((x - lo) / scale) - 1.0
    raise ValueError(f"Unknown normalization method: {method}")


def bandpass_ppg(
    values: np.ndarray,
    fs: float,
    low_hz: float = 0.5,
    high_hz: float = 5.0,
    order: int = 3,
) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.size < max(8, order * 3):
        return x
    nyquist = 0.5 * float(fs)
    high = min(float(high_hz), nyquist * 0.95)
    low = max(float(low_hz), 0.01)
    if high <= low:
        return x

    sos = signal.butter(order, (low, high), btype="bandpass", fs=fs, output="sos")
    try:
        return signal.sosfiltfilt(sos, x)
    except ValueError:
        return signal.sosfilt(sos, x)

