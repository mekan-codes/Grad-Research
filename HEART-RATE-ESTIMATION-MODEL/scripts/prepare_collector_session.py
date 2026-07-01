from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import write_json
from src.utils.config import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one TinyPPGCollector session folder into a prepared subject CSV."
    )
    parser.add_argument("--session-dir", required=True, help="TinyPPGCollector session folder.")
    parser.add_argument(
        "--output-dir",
        default="data/input/collector_prepared",
        help="Directory for pipeline-ready S*.csv files.",
    )
    parser.add_argument(
        "--subject",
        required=True,
        help="Pipeline subject id to write, for example S1. Use one id per person for LOSO.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output CSV.")
    parser.add_argument("--hr-tolerance-ms", type=int, default=2000, help="Nearest HR label tolerance.")
    parser.add_argument("--imu-tolerance-ms", type=int, default=250, help="Nearest IMU tolerance.")
    parser.add_argument("--ecg-tolerance-ms", type=int, default=40, help="Nearest ECG tolerance.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        raise FileNotFoundError(f"Session folder not found: {session_dir}")

    subject = normalize_subject(args.subject)
    output_dir = resolve_project_path(args.output_dir)
    output_path = output_dir / f"{subject}.csv"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists, pass --overwrite to replace it: {output_path}")

    prepared, report = build_prepared_frame(
        session_dir=session_dir,
        hr_tolerance_ms=int(args.hr_tolerance_ms),
        imu_tolerance_ms=int(args.imu_tolerance_ms),
        ecg_tolerance_ms=int(args.ecg_tolerance_ms),
    )
    validate_prepared(prepared, report)

    output_dir.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    report.update(
        {
            "status": "ok",
            "subject": subject,
            "output_csv": str(output_path),
            "output_rows": int(prepared.shape[0]),
            "duration_sec": float(prepared["time"].max() - prepared["time"].min()) if not prepared.empty else 0.0,
        }
    )
    write_json(output_dir / f"{subject}_conversion_report.json", report)
    print(f"Wrote {output_path}")
    print(f"Rows: {prepared.shape[0]}; duration: {report['duration_sec']:.2f}s")
    return 0


def build_prepared_frame(
    session_dir: Path,
    hr_tolerance_ms: int,
    imu_tolerance_ms: int,
    ecg_tolerance_ms: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = read_metadata(session_dir)
    watch_ppg = read_csv(session_dir / "watch_ppg.csv")
    watch_imu = read_csv(session_dir / "watch_imu.csv")
    polar_hr = read_csv(session_dir / "polar_hr.csv")
    polar_ecg = read_csv(session_dir / "polar_ecg.csv")

    report: dict[str, Any] = {
        "session_dir": str(session_dir),
        "metadata": metadata,
        "input_rows": {
            "watch_ppg": int(watch_ppg.shape[0]),
            "watch_imu": int(watch_imu.shape[0]),
            "polar_hr": int(polar_hr.shape[0]),
            "polar_ecg": int(polar_ecg.shape[0]),
        },
        "warnings": [],
    }

    if watch_ppg.empty:
        return empty_prepared(), report

    ppg = watch_ppg.copy()
    ppg["timestamp_unix_ms"] = numeric(ppg.get("timestamp_unix_ms"))
    ppg["ppg"] = first_numeric_column(ppg, ("ppg_green", "ppg_ir", "ppg_red"))
    ppg = ppg[np.isfinite(ppg["timestamp_unix_ms"]) & np.isfinite(ppg["ppg"])].copy()
    if ppg.empty:
        report["warnings"].append("watch_ppg.csv has no finite PPG values.")
        return empty_prepared(), report

    ppg = ppg.sort_values("timestamp_unix_ms").reset_index(drop=True)
    first_ts = float(ppg["timestamp_unix_ms"].iloc[0])
    prepared = pd.DataFrame(
        {
            "timestamp_unix_ms": ppg["timestamp_unix_ms"].astype(float),
            "time": (ppg["timestamp_unix_ms"].astype(float) - first_ts) / 1000.0,
            "ppg": ppg["ppg"].astype(float),
        }
    )

    imu = select_columns(
        watch_imu,
        {
            "timestamp_unix_ms": "timestamp_unix_ms",
            "acc_x": "acc_x",
            "acc_y": "acc_y",
            "acc_z": "acc_z",
        },
    )
    prepared = nearest_join(prepared, imu, tolerance_ms=imu_tolerance_ms)

    hr = select_columns(polar_hr, {"timestamp_unix_ms": "timestamp_unix_ms", "hr": "hr_bpm"})
    prepared = nearest_join(prepared, hr, tolerance_ms=hr_tolerance_ms)

    ecg = select_columns(polar_ecg, {"timestamp_unix_ms": "timestamp_unix_ms", "ecg": "ecg_uv"})
    prepared = nearest_join(prepared, ecg, tolerance_ms=ecg_tolerance_ms)

    for column in ("hr", "acc_x", "acc_y", "acc_z", "ecg"):
        if column not in prepared.columns:
            prepared[column] = np.nan

    prepared = prepared[["time", "ppg", "hr", "acc_x", "acc_y", "acc_z", "ecg"]]
    report["finite_rows"] = {
        "ppg": int(np.isfinite(prepared["ppg"]).sum()),
        "hr": int(np.isfinite(prepared["hr"]).sum()),
        "acc_x": int(np.isfinite(prepared["acc_x"]).sum()),
        "ecg": int(np.isfinite(prepared["ecg"]).sum()),
    }
    report["estimated_ppg_hz"] = estimate_hz(prepared["time"].to_numpy(dtype=float))
    return prepared, report


def validate_prepared(frame: pd.DataFrame, report: dict[str, Any]) -> None:
    errors: list[str] = []
    if frame.empty:
        errors.append("No prepared rows were produced. The session likely has no watch PPG samples.")
    elif not np.isfinite(frame["ppg"]).any():
        errors.append("No finite PPG values were found.")
    elif not np.isfinite(frame["hr"]).any():
        errors.append("No finite HR labels were found. Connect Polar H10 HR/RR before training.")
    elif float(frame["time"].max() - frame["time"].min()) < 30.0:
        errors.append("Prepared recording is shorter than one 30 second training window.")

    if errors:
        report["status"] = "failed"
        report["errors"] = errors
        print(json.dumps(report, indent=2), file=sys.stderr)
        raise RuntimeError("Collector session is not trainable yet: " + " ".join(errors))


def read_metadata(session_dir: Path) -> dict[str, Any]:
    path = session_dir / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def empty_prepared() -> pd.DataFrame:
    return pd.DataFrame(columns=["time", "ppg", "hr", "acc_x", "acc_y", "acc_z", "ecg"])


def numeric(value: Any) -> pd.Series:
    return pd.to_numeric(value, errors="coerce")


def first_numeric_column(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in columns:
        if column in frame.columns:
            result = result.fillna(numeric(frame[column]))
    return result


def select_columns(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    if frame.empty or "timestamp_unix_ms" not in frame.columns:
        return pd.DataFrame(columns=list(mapping.values()))
    selected = pd.DataFrame()
    for source, target in mapping.items():
        selected[target] = numeric(frame[source]) if source in frame.columns else np.nan
    selected = selected[np.isfinite(selected["timestamp_unix_ms"])].copy()
    return selected.sort_values("timestamp_unix_ms").drop_duplicates("timestamp_unix_ms")


def nearest_join(base: pd.DataFrame, values: pd.DataFrame, tolerance_ms: int) -> pd.DataFrame:
    if values.empty:
        return base
    return pd.merge_asof(
        base.sort_values("timestamp_unix_ms"),
        values.sort_values("timestamp_unix_ms"),
        on="timestamp_unix_ms",
        direction="nearest",
        tolerance=float(tolerance_ms),
    )


def estimate_hz(time_sec: np.ndarray) -> float | None:
    if time_sec.size < 3:
        return None
    diffs = np.diff(time_sec)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None
    return float(1.0 / np.median(diffs))


def normalize_subject(value: str) -> str:
    subject = str(value).strip().upper().removesuffix(".CSV")
    if not subject:
        raise ValueError("Subject id cannot be empty")
    return subject


if __name__ == "__main__":
    raise SystemExit(main())
