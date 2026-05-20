from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare local PPG-DaLiA .pkl files into time,ppg,hr CSV files.",
    )
    parser.add_argument("--input-root", default="data/raw/PPG-DaLiA", help="Root containing S*/S*.pkl files.")
    parser.add_argument("--output-dir", default="data/input/prepared", help="Output CSV directory.")
    parser.add_argument("--subject", default=None, help="Optional subject id such as S1.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    files = discover_subject_pickles(input_root, args.subject)
    if not files:
        print(f"No PPG-DaLiA pickle files found under {input_root}.")
        print("Expected local structure: data/raw/PPG-DaLiA/S1/S1.pkl, S2/S2.pkl, ...")
        print("This script does not download the dataset automatically.")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    for pkl_path in files:
        frame = prepare_one_pickle(pkl_path)
        output_path = output_dir / f"{pkl_path.stem.upper()}.csv"
        frame.to_csv(output_path, index=False)
        print(f"Wrote {output_path}")
    return 0


def discover_subject_pickles(input_root: Path, subject: str | None) -> list[Path]:
    if not input_root.exists():
        return []
    if subject:
        subject_id = subject.upper().removesuffix(".PKL")
        candidate = input_root / subject_id / f"{subject_id}.pkl"
        return [candidate] if candidate.exists() else []
    return sorted(input_root.glob("S*/S*.pkl"), key=lambda path: _subject_sort_key(path.stem))


def prepare_one_pickle(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")

    try:
        wrist = payload["signal"]["wrist"]
        bvp = np.asarray(wrist["BVP"], dtype=float).reshape(-1)
    except Exception as exc:
        raise ValueError(f"{path} does not look like a PPG-DaLiA pickle with signal.wrist.BVP") from exc

    fs_ppg = 64.0
    time = np.arange(bvp.size, dtype=float) / fs_ppg
    frame = pd.DataFrame({"time": time, "ppg": bvp})

    labels = np.asarray(payload.get("label", []), dtype=float).reshape(-1)
    if labels.size:
        label_time = np.linspace(0.0, time[-1], labels.size)
        frame["hr"] = np.interp(time, label_time, labels, left=np.nan, right=np.nan)
    else:
        frame["hr"] = np.nan

    acc = np.asarray(wrist.get("ACC", []), dtype=float)
    if acc.ndim == 2 and acc.shape[0] > 1:
        if acc.shape[1] == 3:
            acc_values = acc
        elif acc.shape[0] == 3:
            acc_values = acc.T
        else:
            acc_values = None
        if acc_values is not None:
            acc_time = np.linspace(0.0, time[-1], acc_values.shape[0])
            for idx, name in enumerate(("acc_x", "acc_y", "acc_z")):
                frame[name] = np.interp(time, acc_time, acc_values[:, idx], left=np.nan, right=np.nan)
    return frame


def _subject_sort_key(subject: str) -> tuple[int, str]:
    text = subject.upper()
    try:
        return int(text.lstrip("S")), text
    except ValueError:
        return 9999, text


if __name__ == "__main__":
    raise SystemExit(main())

