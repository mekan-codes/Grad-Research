from __future__ import annotations

import argparse
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare PPG_FieldStudy Empatica E4 BVP/HR CSV files for the experiment runner.",
    )
    parser.add_argument("--zip", required=True, help="Path to archive.zip.")
    parser.add_argument(
        "--output-dir",
        default="data/input/prepared",
        help="Directory for prepared per-subject CSV files.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Optional subject id such as S1. When omitted, all subjects are prepared.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zip_path = Path(args.zip)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        subjects = discover_subjects(zf)
        if args.subject:
            subject = args.subject.upper()
            if subject not in subjects:
                raise ValueError(f"Subject {subject} not found. Available: {', '.join(subjects)}")
            subjects = [subject]

        written = []
        for subject in subjects:
            output_path = output_dir / f"{subject}.csv"
            prepare_subject(zf, subject, output_path)
            written.append(output_path)
            print(f"Wrote {output_path}")

    print(f"Prepared {len(written)} file(s).")
    return 0


def discover_subjects(zf: zipfile.ZipFile) -> list[str]:
    subjects = []
    for name in zf.namelist():
        parts = name.split("/")
        if len(parts) == 4 and parts[0] == "PPG_FieldStudy" and parts[3] == "BVP.csv":
            subjects.append(parts[1])
    return sorted(subjects, key=_subject_sort_key)


def prepare_subject(zf: zipfile.ZipFile, subject: str, output_path: Path) -> None:
    root = f"PPG_FieldStudy/{subject}/{subject}_E4"
    bvp = read_e4_csv(zf, f"{root}/BVP.csv")
    hr = read_e4_csv(zf, f"{root}/HR.csv")

    ppg_time = np.arange(bvp.values.shape[0], dtype=float) / bvp.fs
    absolute_ppg_time = bvp.start_time + ppg_time

    hr_time = hr.start_time + np.arange(hr.values.shape[0], dtype=float) / hr.fs
    hr_values = np.interp(absolute_ppg_time, hr_time, hr.values[:, 0], left=np.nan, right=np.nan)

    frame = pd.DataFrame(
        {
            "time": ppg_time,
            "ppg": bvp.values[:, 0],
            "hr": hr_values,
        }
    )

    acc_member = f"{root}/ACC.csv"
    if acc_member in zf.namelist():
        acc = read_e4_csv(zf, acc_member)
        acc_time = acc.start_time + np.arange(acc.values.shape[0], dtype=float) / acc.fs
        frame["acc_x"] = np.interp(absolute_ppg_time, acc_time, acc.values[:, 0], left=np.nan, right=np.nan)
        frame["acc_y"] = np.interp(absolute_ppg_time, acc_time, acc.values[:, 1], left=np.nan, right=np.nan)
        frame["acc_z"] = np.interp(absolute_ppg_time, acc_time, acc.values[:, 2], left=np.nan, right=np.nan)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


class E4Data:
    def __init__(self, start_time: float, fs: float, values: np.ndarray) -> None:
        self.start_time = start_time
        self.fs = fs
        self.values = values


def read_e4_csv(zf: zipfile.ZipFile, member: str) -> E4Data:
    with zf.open(member) as handle:
        frame = pd.read_csv(handle, header=None)

    if frame.shape[0] < 3:
        raise ValueError(f"{member} does not contain enough rows")

    start_time = float(frame.iloc[0, 0])
    fs = float(frame.iloc[1, 0])
    values = frame.iloc[2:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return E4Data(start_time=start_time, fs=fs, values=values)


def _subject_sort_key(subject: str) -> tuple[int, str]:
    try:
        return int(subject.lstrip("S")), subject
    except ValueError:
        return 9999, subject


if __name__ == "__main__":
    raise SystemExit(main())

