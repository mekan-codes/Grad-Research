from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_ppg_csv
from src.hr_estimator import KID_PPG_WARNING, estimate_hr_series
from src.metrics import build_metric_payload
from src.tiny_ppg_filter import TINY_PPG_WARNING, apply_tiny_ppg_filter
from src.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw PPG HR estimation with Tiny-PPG artifact-filtered HR estimation.",
    )
    parser.add_argument("--input", required=True, help="Path to smartwatch CSV input.")
    parser.add_argument(
        "--fs",
        type=float,
        default=100.0,
        help="Fallback sampling rate in Hz when no usable time column exists.",
    )
    parser.add_argument(
        "--output",
        default="data/output/results.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--tiny-ppg-dir",
        default=None,
        help="Optional path containing Tiny-PPG model.py.",
    )
    parser.add_argument(
        "--tiny-ppg-checkpoint",
        default=None,
        help="Optional path to a trained Tiny-PPG PyTorch checkpoint.",
    )
    parser.add_argument(
        "--artifact-threshold",
        type=float,
        default=0.5,
        help="Tiny-PPG artifact probability threshold.",
    )
    parser.add_argument(
        "--no-kid",
        action="store_true",
        help="Skip KID-PPG and use the classical signal-processing estimator.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    loaded = load_ppg_csv(args.input, fallback_fs=args.fs)

    raw_hr = estimate_hr_series(
        loaded.ppg,
        loaded.fs,
        prefer_kid=not args.no_kid,
    )
    kid_warning_printed = False
    if raw_hr.kid_error:
        print(KID_PPG_WARNING)
        kid_warning_printed = True

    tiny_result = apply_tiny_ppg_filter(
        loaded.ppg,
        loaded.fs,
        tiny_ppg_dir=args.tiny_ppg_dir,
        checkpoint_path=args.tiny_ppg_checkpoint,
        threshold=args.artifact_threshold,
    )

    cleaned_hr = None
    if not tiny_result.available:
        print(TINY_PPG_WARNING)
    else:
        cleaned_hr = estimate_hr_series(
            tiny_result.cleaned_ppg,
            loaded.fs,
            prefer_kid=(not args.no_kid and raw_hr.method == "kid_ppg"),
        )
        if cleaned_hr.kid_error and not kid_warning_printed:
            print(KID_PPG_WARNING)

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
            "path": str(Path(args.input)),
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
    }

    output_path = write_json(args.output, result)
    print(f"Results saved to {output_path}")

    if raw_hr.summary_bpm is not None:
        print(f"Raw HR estimate: {raw_hr.summary_bpm:.2f} bpm ({raw_hr.method})")
    if cleaned_hr is not None and cleaned_hr.summary_bpm is not None:
        print(f"Cleaned HR estimate: {cleaned_hr.summary_bpm:.2f} bpm ({cleaned_hr.method})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

