from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


PPG_COLUMN_NAMES = ("ppg", "PPG", "signal", "value", "bvp", "green", "ir", "red")
TIME_COLUMN_NAMES = ("time", "timestamp", "t", "seconds")
HR_COLUMN_NAMES = ("hr", "HR", "heart_rate", "bpm", "label")


@dataclass
class LoadedPPGData:
    path: Path
    dataframe: pd.DataFrame
    ppg: np.ndarray
    time_seconds: np.ndarray
    fs: float
    fs_source: str
    ppg_column: str
    time_column: str | None
    hr_column: str | None
    ground_truth_hr: np.ndarray | None
    ground_truth_time_seconds: np.ndarray | None
    accel_columns: list[str]


def load_ppg_csv(path: str | Path, fallback_fs: float | None = None) -> LoadedPPGData:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    if dataframe.empty:
        raise ValueError("Input CSV is empty")

    ppg_column = detect_column(dataframe.columns, PPG_COLUMN_NAMES)
    if ppg_column is None:
        expected = ", ".join(PPG_COLUMN_NAMES)
        raise ValueError(f"Could not detect a PPG column. Expected one of: {expected}")

    time_column = detect_column(dataframe.columns, TIME_COLUMN_NAMES)
    hr_column = detect_column(dataframe.columns, HR_COLUMN_NAMES)
    accel_columns = detect_accelerometer_columns(dataframe.columns)

    ppg = pd.to_numeric(dataframe[ppg_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(ppg).any():
        raise ValueError(f"PPG column '{ppg_column}' contains no numeric samples")

    parsed_time = _parse_time_column(dataframe, time_column, fallback_fs) if time_column else None
    fs, fs_source = _detect_sampling_rate(parsed_time, fallback_fs)
    if fs is None:
        raise ValueError("Could not infer sampling rate. Provide --fs.")

    if parsed_time is None:
        time_seconds = np.arange(ppg.size, dtype=float) / fs
        fs_source = "--fs"
    else:
        time_seconds = parsed_time

    ground_truth_hr = None
    ground_truth_time_seconds = None
    if hr_column is not None:
        ground_truth_hr = pd.to_numeric(dataframe[hr_column], errors="coerce").to_numpy(dtype=float)
        ground_truth_time_seconds = time_seconds if ground_truth_hr.size == time_seconds.size else None

    return LoadedPPGData(
        path=csv_path,
        dataframe=dataframe,
        ppg=ppg,
        time_seconds=time_seconds,
        fs=float(fs),
        fs_source=fs_source,
        ppg_column=ppg_column,
        time_column=time_column,
        hr_column=hr_column,
        ground_truth_hr=ground_truth_hr,
        ground_truth_time_seconds=ground_truth_time_seconds,
        accel_columns=accel_columns,
    )


def detect_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized_to_original = {_normalize_name(column): column for column in columns}
    for candidate in candidates:
        normalized = _normalize_name(candidate)
        if normalized in normalized_to_original:
            return normalized_to_original[normalized]
    return None


def detect_accelerometer_columns(columns: Iterable[str]) -> list[str]:
    normalized = {column: _normalize_name(column) for column in columns}
    accel_tokens = (
        "acc",
        "accel",
        "accelerometer",
        "acc_x",
        "acc_y",
        "acc_z",
        "x_acc",
        "y_acc",
        "z_acc",
        "accelerometer_x",
        "accelerometer_y",
        "accelerometer_z",
    )
    return [column for column, name in normalized.items() if name in accel_tokens]


def _normalize_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lower())
    return cleaned.strip("_")


def _parse_time_column(
    dataframe: pd.DataFrame,
    time_column: str | None,
    fallback_fs: float | None,
) -> np.ndarray | None:
    if time_column is None:
        return None

    raw = dataframe[time_column]
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().sum() >= max(3, int(0.5 * len(numeric))):
        return _numeric_time_to_seconds(numeric.to_numpy(dtype=float), fallback_fs)

    parsed = pd.to_datetime(raw, errors="coerce", utc=True)
    if parsed.notna().sum() < 3:
        return None

    first = parsed[parsed.notna()].iloc[0]
    seconds = (parsed - first).dt.total_seconds().to_numpy(dtype=float)
    return seconds


def _numeric_time_to_seconds(values: np.ndarray, fallback_fs: float | None) -> np.ndarray | None:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() < 3:
        return None

    diffs = np.diff(arr[finite])
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None

    median_diff = float(np.median(diffs))
    scale = _choose_numeric_time_scale(arr[finite], median_diff, fallback_fs)
    seconds = arr / scale
    first = seconds[finite][0]
    return seconds - first


def _choose_numeric_time_scale(
    finite_values: np.ndarray,
    median_diff: float,
    fallback_fs: float | None,
) -> float:
    scales = (1.0, 1_000.0, 1_000_000.0, 1_000_000_000.0)

    if fallback_fs is not None and fallback_fs > 0:
        candidates: list[tuple[float, float]] = []
        for scale in scales:
            candidate_fs = scale / median_diff
            if 0.1 <= candidate_fs <= 1000:
                score = abs(np.log(candidate_fs / fallback_fs))
                candidates.append((score, scale))
        if candidates:
            return min(candidates, key=lambda item: item[0])[1]

    median_abs_value = float(np.nanmedian(np.abs(finite_values)))
    if median_abs_value > 1e14:
        return 1_000_000_000.0
    if median_abs_value > 1e11:
        return 1_000.0
    if median_abs_value > 1e8:
        return 1.0
    if median_diff > 10_000:
        return 1_000_000.0
    if median_diff > 0.5:
        return 1_000.0
    return 1.0


def _detect_sampling_rate(
    time_seconds: np.ndarray | None,
    fallback_fs: float | None,
) -> tuple[float | None, str]:
    if time_seconds is not None:
        finite = np.isfinite(time_seconds)
        if finite.sum() >= 3:
            diffs = np.diff(time_seconds[finite])
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size:
                fs = 1.0 / float(np.median(diffs))
                if 0.1 <= fs <= 1000:
                    return fs, "time_column"

    if fallback_fs is not None and fallback_fs > 0:
        return float(fallback_fs), "--fs"

    return None, "unavailable"

