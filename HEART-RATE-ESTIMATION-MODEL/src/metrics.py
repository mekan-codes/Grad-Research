from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ErrorMetrics:
    mae: float | None
    rmse: float | None
    n: int


def compute_error_metrics(
    estimated_bpm: np.ndarray,
    estimate_times_sec: np.ndarray,
    ground_truth_hr: np.ndarray | None,
    ground_truth_times_sec: np.ndarray | None,
) -> ErrorMetrics:
    if ground_truth_hr is None:
        return ErrorMetrics(mae=None, rmse=None, n=0)

    estimates = np.asarray(estimated_bpm, dtype=float)
    estimate_times = np.asarray(estimate_times_sec, dtype=float)
    valid_estimates = np.isfinite(estimates) & np.isfinite(estimate_times)
    if valid_estimates.sum() == 0:
        return ErrorMetrics(mae=None, rmse=None, n=0)

    estimates = estimates[valid_estimates]
    estimate_times = estimate_times[valid_estimates]

    gt = np.asarray(ground_truth_hr, dtype=float)
    gt_valid = np.isfinite(gt)
    if gt_valid.sum() == 0:
        return ErrorMetrics(mae=None, rmse=None, n=0)

    if ground_truth_times_sec is not None and len(ground_truth_times_sec) == len(gt):
        gt_times = np.asarray(ground_truth_times_sec, dtype=float)
        gt_valid = gt_valid & np.isfinite(gt_times)
        if gt_valid.sum() == 0:
            return ErrorMetrics(mae=None, rmse=None, n=0)

        gt_times = gt_times[gt_valid]
        gt = gt[gt_valid]
        order = np.argsort(gt_times)
        gt_times = gt_times[order]
        gt = gt[order]
        reference = np.interp(estimate_times, gt_times, gt)
    else:
        reference = np.full_like(estimates, float(np.nanmedian(gt[gt_valid])), dtype=float)

    errors = estimates - reference
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        return ErrorMetrics(mae=None, rmse=None, n=0)

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    return ErrorMetrics(mae=mae, rmse=rmse, n=int(errors.size))


def build_metric_payload(
    raw_estimated_bpm: np.ndarray,
    raw_times_sec: np.ndarray,
    cleaned_estimated_bpm: np.ndarray | None,
    cleaned_times_sec: np.ndarray | None,
    ground_truth_hr: np.ndarray | None,
    ground_truth_times_sec: np.ndarray | None,
    percent_signal_removed: float | None,
) -> dict[str, float | int | None]:
    raw = compute_error_metrics(
        raw_estimated_bpm,
        raw_times_sec,
        ground_truth_hr,
        ground_truth_times_sec,
    )

    cleaned = ErrorMetrics(mae=None, rmse=None, n=0)
    if cleaned_estimated_bpm is not None and cleaned_times_sec is not None:
        cleaned = compute_error_metrics(
            cleaned_estimated_bpm,
            cleaned_times_sec,
            ground_truth_hr,
            ground_truth_times_sec,
        )

    improvement = None
    if raw.mae is not None and cleaned.mae is not None:
        improvement = raw.mae - cleaned.mae

    return {
        "raw_mae": raw.mae,
        "cleaned_mae": cleaned.mae,
        "raw_rmse": raw.rmse,
        "cleaned_rmse": cleaned.rmse,
        "improvement_bpm": improvement,
        "percent_signal_removed": percent_signal_removed,
        "raw_metric_windows": raw.n,
        "cleaned_metric_windows": cleaned.n,
    }


def percent_removed(mask: np.ndarray | None) -> float | None:
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return None
    return float(arr.mean() * 100.0)

