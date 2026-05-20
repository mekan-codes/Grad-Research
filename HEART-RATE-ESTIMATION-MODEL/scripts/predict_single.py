from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import predict_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict HR for one PPG CSV using a full framework checkpoint.")
    parser.add_argument("--input", required=True, help="CSV with time, ppg, optional acc, optional hr.")
    parser.add_argument("--checkpoint", required=True, help="Full framework checkpoint path.")
    parser.add_argument("--fs", type=float, default=64.0, help="Fallback sampling rate.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = predict_csv(args.input, args.checkpoint, fallback_fs=args.fs)
    print(f"Predicted HR: {result['predicted_hr_bpm']:.2f} bpm")
    if result["label_hr_bpm"] is not None:
        print(f"Reference HR: {result['label_hr_bpm']:.2f} bpm")
        print(f"Absolute error: {result['absolute_error_bpm']:.2f} bpm")
    print(f"Cropped percent: {result['cropped_percent']:.2f}%")
    summary = result["artifact_summary"]
    print(
        "Artifact summary: "
        f"{summary['artifact_samples']}/{summary['total_samples']} samples "
        f"({summary['artifact_percent']:.2f}%), "
        f"{summary['number_of_removed_segments']} removed segment(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

