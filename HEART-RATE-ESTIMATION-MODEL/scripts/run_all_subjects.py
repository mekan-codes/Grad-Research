from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_ppg_csv
from src.hr_estimator import KID_PPG_WARNING, estimate_hr_series
from src.metrics import build_metric_payload
from src.tiny_ppg_filter import TINY_PPG_WARNING, apply_tiny_ppg_filter
from src.utils import write_json


INPUT_DIR = PROJECT_ROOT / "data" / "input" / "prepared"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
FS_HZ = 64.0
DEFAULT_ARTIFACT_THRESHOLD = 0.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PPG HR experiment on all prepared subject CSV files.",
    )
    parser.add_argument(
        "--artifact-threshold",
        type=float,
        default=DEFAULT_ARTIFACT_THRESHOLD,
        help="Tiny-PPG artifact probability threshold.",
    )
    parser.add_argument(
        "--artifact-output-mode",
        default="artifact_probability",
        choices=("artifact_probability", "clean_probability", "logits", "class_index"),
        help="How to interpret TinyPPG segmentation output.",
    )
    parser.add_argument(
        "--artifact-class-index",
        type=int,
        default=1,
        help="Class index to use when TinyPPG returns class channels.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subject_files = find_subject_files(INPUT_DIR)
    if not subject_files:
        print(f"No subject CSV files found in {INPUT_DIR}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for subject_file in subject_files:
        subject = subject_file.stem
        output_path = OUTPUT_DIR / f"results_{subject}.json"

        print(f"Processing {subject}...")
        result, notes = run_subject(
            subject_file,
            args.artifact_threshold,
            args.artifact_output_mode,
            args.artifact_class_index,
        )
        write_json(output_path, result)

        summary_rows.append(build_summary_row(subject, result, notes))
        print(f"Saved {output_path}")

    summary_path = OUTPUT_DIR / "summary_results.csv"
    summary = pd.DataFrame(summary_rows, columns=summary_columns())
    summary.to_csv(summary_path, index=False)

    print()
    print(f"Subjects processed: {len(summary_rows)}")
    print(f"Average raw MAE: {mean_or_nan(summary['raw_mae']):.2f} bpm")
    print(f"Average raw RMSE: {mean_or_nan(summary['raw_rmse']):.2f} bpm")
    print(f"Summary saved to {summary_path}")

    return 0


def find_subject_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("S*.csv"), key=subject_sort_key)
    return [path for path in files if path.is_file()]


def run_subject(
    input_path: Path,
    artifact_threshold: float,
    artifact_output_mode: str,
    artifact_class_index: int,
) -> tuple[dict, list[str]]:
    notes: list[str] = []
    loaded = load_ppg_csv(input_path, fallback_fs=FS_HZ)

    raw_hr = estimate_hr_series(loaded.ppg, loaded.fs, prefer_kid=True)
    if raw_hr.kid_error:
        print(KID_PPG_WARNING)
        notes.append("KID-PPG unavailable; used classical HR estimator")

    tiny_result = apply_tiny_ppg_filter(
        loaded.ppg,
        loaded.fs,
        threshold=artifact_threshold,
        artifact_output_mode=artifact_output_mode,
        artifact_class_index=artifact_class_index,
    )

    cleaned_hr = None
    if not tiny_result.available:
        print(TINY_PPG_WARNING)
        notes.append("Tiny-PPG unavailable; raw baseline only")
    else:
        cleaned_hr = estimate_hr_series(
            tiny_result.cleaned_ppg,
            loaded.fs,
            prefer_kid=(raw_hr.method == "kid_ppg"),
        )
        if cleaned_hr.kid_error:
            notes.append("KID-PPG unavailable for cleaned signal; used classical HR estimator")

    metrics = build_metric_payload(
        raw_estimated_bpm=raw_hr.bpm_values,
        raw_times_sec=raw_hr.times_sec,
        cleaned_estimated_bpm=cleaned_hr.bpm_values if cleaned_hr is not None else None,
        cleaned_times_sec=cleaned_hr.times_sec if cleaned_hr is not None else None,
        ground_truth_hr=loaded.ground_truth_hr,
        ground_truth_times_sec=loaded.ground_truth_time_seconds,
        percent_signal_removed=tiny_result.percent_signal_removed,
    )

    result = {
        "input": {
            "path": str(input_path),
            "rows": int(loaded.dataframe.shape[0]),
            "columns": list(loaded.dataframe.columns),
            "ppg_column": loaded.ppg_column,
            "time_column": loaded.time_column,
            "hr_column": loaded.hr_column,
            "accelerometer_columns": loaded.accel_columns,
            "sampling_rate_hz": loaded.fs,
            "sampling_rate_source": loaded.fs_source,
        },
        "raw": raw_hr.to_dict(),
        "tiny_ppg": tiny_result.to_dict(),
        "cleaned": cleaned_hr.to_dict() if cleaned_hr is not None else None,
        "metrics": metrics,
        "config": {
            "artifact_threshold": artifact_threshold,
            "artifact_output_mode": artifact_output_mode,
            "artifact_class_index": artifact_class_index,
        },
    }
    return result, notes


def build_summary_row(subject: str, result: dict, notes: list[str]) -> dict:
    raw = result["raw"]
    cleaned = result["cleaned"]
    metrics = result["metrics"]

    return {
        "subject": subject,
        "raw_estimated_bpm": raw.get("bpm"),
        "raw_mae": metrics.get("raw_mae"),
        "raw_rmse": metrics.get("raw_rmse"),
        "cleaned_estimated_bpm": cleaned.get("bpm") if cleaned else None,
        "cleaned_mae": metrics.get("cleaned_mae"),
        "cleaned_rmse": metrics.get("cleaned_rmse"),
        "percent_signal_removed": metrics.get("percent_signal_removed"),
        "method": raw.get("method"),
        "notes": "; ".join(notes),
    }


def summary_columns() -> list[str]:
    return [
        "subject",
        "raw_estimated_bpm",
        "raw_mae",
        "raw_rmse",
        "cleaned_estimated_bpm",
        "cleaned_mae",
        "cleaned_rmse",
        "percent_signal_removed",
        "method",
        "notes",
    ]


def mean_or_nan(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return float("nan")
    return float(np.nanmean(numeric))


def subject_sort_key(path: Path) -> tuple[int, str]:
    subject = path.stem.upper()
    try:
        return int(subject.lstrip("S")), subject
    except ValueError:
        return 9999, subject


if __name__ == "__main__":
    raise SystemExit(main())
