from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np
from scipy import fft, signal

from .utils import finite_fraction, interpolate_nans, zscore


KID_PPG_WARNING = "KID-PPG unavailable. Using classical signal-processing HR estimator."


@dataclass
class HRSeries:
    bpm_values: np.ndarray
    times_sec: np.ndarray
    method: str
    summary_bpm: float | None
    confidence: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    kid_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bpm": self.summary_bpm,
            "method": self.method,
            "confidence": self.confidence,
            "n_estimates": int(np.isfinite(self.bpm_values).sum()),
            "times_sec": self.times_sec,
            "bpm_series": self.bpm_values,
            "details": self.details,
            "kid_error": self.kid_error,
        }


def estimate_hr_series(
    ppg: np.ndarray,
    fs: float,
    prefer_kid: bool = True,
    window_sec: float = 8.0,
    step_sec: float = 2.0,
) -> HRSeries:
    kid_error = None
    if prefer_kid:
        try:
            return _estimate_with_kid_ppg(ppg, fs)
        except Exception as exc:  # KID is optional and can fail on dependency or shape issues.
            kid_error = str(exc)

    fallback = _estimate_classical_series(ppg, fs, window_sec=window_sec, step_sec=step_sec)
    fallback.kid_error = kid_error
    return fallback


def _estimate_with_kid_ppg(ppg: np.ndarray, fs: float) -> HRSeries:
    try:
        from kid_ppg.kid_ppg import KID_PPG
    except Exception as exc:
        raise RuntimeError(f"KID-PPG import failed: {exc}") from exc

    y = np.asarray(ppg, dtype=float)
    if y.size < max(10, int(fs * 10)):
        raise RuntimeError("KID-PPG needs at least about 10 seconds of PPG")

    filled = interpolate_nans(y, min_finite_fraction=0.5)
    finite_mask = np.isfinite(y).astype(float)

    target_fs = 32.0
    y_resampled = _resample_signal(filled, fs, target_fs)
    target_times = np.arange(y_resampled.size, dtype=float) / target_fs
    original_times = np.arange(y.size, dtype=float) / fs
    quality = np.interp(target_times, original_times, finite_mask)

    window_size = 256
    step_size = 64
    windows, centers, window_quality = _window_resampled_for_kid(
        y_resampled,
        quality,
        target_fs,
        window_size,
        step_size,
    )
    if windows.shape[0] < 2:
        raise RuntimeError("KID-PPG input is too short after resampling")

    pairs = np.concatenate([windows[:-1, :, None], windows[1:, :, None]], axis=-1)
    pair_quality = (window_quality[:-1] >= 0.8) & (window_quality[1:] >= 0.8)
    if pair_quality.sum() == 0:
        raise RuntimeError("KID-PPG windows contain too many masked samples")

    pairs = pairs[pair_quality].astype(np.float32)
    prediction_times = centers[1:][pair_quality]

    model = KID_PPG()
    bpm, std = model.predict(pairs)
    bpm = np.asarray(bpm, dtype=float)
    std = np.asarray(std, dtype=float)

    valid = np.isfinite(bpm) & (bpm >= 30.0) & (bpm <= 240.0)
    if valid.sum() == 0:
        raise RuntimeError("KID-PPG returned no finite plausible HR estimates")

    bpm = bpm[valid]
    std = std[valid] if std.size == valid.size else np.full_like(bpm, np.nan)
    prediction_times = prediction_times[valid]

    confidence = None
    if np.isfinite(std).any():
        confidence = float(1.0 / (1.0 + np.nanmedian(std)))

    return HRSeries(
        bpm_values=bpm,
        times_sec=prediction_times,
        method="kid_ppg",
        summary_bpm=float(np.nanmedian(bpm)),
        confidence=confidence,
        details={
            "target_fs_hz": target_fs,
            "window_sec": 8.0,
            "step_sec": 2.0,
            "median_prediction_std": float(np.nanmedian(std)) if np.isfinite(std).any() else None,
        },
    )


def _estimate_classical_series(
    ppg: np.ndarray,
    fs: float,
    window_sec: float,
    step_sec: float,
) -> HRSeries:
    y = np.asarray(ppg, dtype=float)
    starts = _window_starts(y.size, fs, window_sec, step_sec)

    estimates: list[float] = []
    times: list[float] = []
    details: list[dict[str, Any]] = []
    for start, stop in starts:
        window = y[start:stop]
        bpm, window_details = _estimate_classical_window(window, fs)
        estimates.append(bpm)
        times.append((start + stop) / (2.0 * fs))
        details.append(window_details)

    bpm_values = np.asarray(estimates, dtype=float)
    times_sec = np.asarray(times, dtype=float)
    valid = np.isfinite(bpm_values)
    summary = float(np.nanmedian(bpm_values[valid])) if valid.any() else None

    confidence = None
    if valid.sum() > 1:
        spread = float(np.nanstd(bpm_values[valid]))
        confidence = 1.0 / (1.0 + spread)
    elif valid.sum() == 1:
        confidence = 0.5

    return HRSeries(
        bpm_values=bpm_values,
        times_sec=times_sec,
        method="classical_fft_peak",
        summary_bpm=summary,
        confidence=confidence,
        details={
            "window_sec": window_sec,
            "step_sec": step_sec,
            "valid_windows": int(valid.sum()),
            "total_windows": int(bpm_values.size),
            "window_details": details,
        },
    )


def _estimate_classical_window(window: np.ndarray, fs: float) -> tuple[float, dict[str, Any]]:
    detail: dict[str, Any] = {"finite_fraction": finite_fraction(window)}
    min_samples = max(8, int(round(fs * 3.0)))
    if np.isfinite(window).sum() < min_samples:
        detail["reason"] = "too_few_finite_samples"
        return np.nan, detail

    try:
        x = interpolate_nans(window, min_finite_fraction=0.5)
    except ValueError as exc:
        detail["reason"] = str(exc)
        return np.nan, detail

    x = signal.detrend(x)
    x = zscore(x)

    try:
        filtered = _bandpass_ppg(x, fs)
    except ValueError as exc:
        detail["reason"] = str(exc)
        return np.nan, detail

    fft_bpm = _fft_bpm(filtered, fs)
    peak_bpm = _peak_bpm(filtered, fs)

    if np.isfinite(fft_bpm) and np.isfinite(peak_bpm) and abs(fft_bpm - peak_bpm) <= 12.0:
        bpm = float((fft_bpm + peak_bpm) / 2.0)
        source = "fft_peak_average"
    elif np.isfinite(fft_bpm):
        bpm = float(fft_bpm)
        source = "fft"
    elif np.isfinite(peak_bpm):
        bpm = float(peak_bpm)
        source = "peak"
    else:
        bpm = np.nan
        source = "unavailable"

    detail.update({"fft_bpm": fft_bpm, "peak_bpm": peak_bpm, "source": source})
    return bpm, detail


def _bandpass_ppg(values: np.ndarray, fs: float) -> np.ndarray:
    low = 0.7
    high = min(3.0, fs * 0.45)
    if high <= low:
        raise ValueError(f"sampling rate {fs:.3f} Hz is too low for 0.7-3.0 Hz HR band")

    sos = signal.butter(3, (low, high), btype="bandpass", fs=fs, output="sos")
    try:
        return signal.sosfiltfilt(sos, values)
    except ValueError:
        return signal.sosfilt(sos, values)


def _fft_bpm(values: np.ndarray, fs: float) -> float:
    n = values.size
    if n < 8:
        return np.nan

    n_fft = max(2048, int(2 ** np.ceil(np.log2(max(n * 4, 16)))))
    freqs = fft.rfftfreq(n_fft, d=1.0 / fs)
    spectrum = np.abs(fft.rfft(values, n=n_fft)) ** 2
    band = (freqs >= 0.7) & (freqs <= 3.0)
    if not band.any():
        return np.nan

    band_power = spectrum[band]
    band_freqs = freqs[band]
    peaks, _ = signal.find_peaks(band_power)
    if peaks.size:
        idx = peaks[np.argmax(band_power[peaks])]
    else:
        idx = int(np.argmax(band_power))
    return float(band_freqs[idx] * 60.0)


def _peak_bpm(values: np.ndarray, fs: float) -> float:
    min_distance = max(1, int(round(fs / 3.0)))
    prominence = max(0.1, float(np.nanstd(values) * 0.2))
    peaks, _ = signal.find_peaks(values, distance=min_distance, prominence=prominence)
    if peaks.size < 2:
        return np.nan

    intervals = np.diff(peaks) / fs
    intervals = intervals[(intervals > 0) & np.isfinite(intervals)]
    if intervals.size == 0:
        return np.nan

    bpm = 60.0 / float(np.median(intervals))
    if 42.0 <= bpm <= 180.0:
        return bpm
    return np.nan


def _window_starts(
    n_samples: int,
    fs: float,
    window_sec: float,
    step_sec: float,
) -> list[tuple[int, int]]:
    if n_samples <= 0:
        return []

    window_size = max(1, int(round(window_sec * fs)))
    step_size = max(1, int(round(step_sec * fs)))
    if n_samples < window_size:
        return [(0, n_samples)]

    starts = list(range(0, n_samples - window_size + 1, step_size))
    last_start = n_samples - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return [(start, start + window_size) for start in starts]


def _resample_signal(values: np.ndarray, fs: float, target_fs: float) -> np.ndarray:
    if abs(fs - target_fs) < 1e-6:
        return values.astype(float, copy=True)

    ratio = Fraction(target_fs / fs).limit_denominator(1000)
    return signal.resample_poly(values, ratio.numerator, ratio.denominator)


def _window_resampled_for_kid(
    values: np.ndarray,
    quality: np.ndarray,
    fs: float,
    window_size: int,
    step_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.size < window_size:
        return np.empty((0, window_size)), np.empty(0), np.empty(0)

    starts = list(range(0, values.size - window_size + 1, step_size))
    windows = []
    centers = []
    qualities = []
    for start in starts:
        stop = start + window_size
        windows.append(values[start:stop])
        centers.append((start + stop) / (2.0 * fs))
        qualities.append(float(np.mean(quality[start:stop])))

    return (
        np.asarray(windows, dtype=float),
        np.asarray(centers, dtype=float),
        np.asarray(qualities, dtype=float),
    )

