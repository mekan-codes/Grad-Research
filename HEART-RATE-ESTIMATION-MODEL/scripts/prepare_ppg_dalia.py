from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import shutil
import sys
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_TRAIN_SUBJECTS = [f"S{i}" for i in range(1, 11)]
DEFAULT_VAL_SUBJECTS = [f"S{i}" for i in range(11, 14)]
DEFAULT_TEST_SUBJECTS = [f"S{i}" for i in range(14, 16)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare local PPG-DaLiA .pkl files into subject-split CSV files.",
    )
    parser.add_argument("--data-root", "--input-root", default="data/raw/PPG_Dalia", help="Root containing S*/S*.pkl files.")
    parser.add_argument("--output-dir", default="data/prepared/ppg_dalia", help="Prepared output directory.")
    parser.add_argument("--train-subjects", default=",".join(DEFAULT_TRAIN_SUBJECTS), help="Comma-separated train subjects.")
    parser.add_argument("--val-subjects", default=",".join(DEFAULT_VAL_SUBJECTS), help="Comma-separated val subjects.")
    parser.add_argument("--test-subjects", default=",".join(DEFAULT_TEST_SUBJECTS), help="Comma-separated test subjects.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing prepared files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    splits = {
        "train": parse_subject_list(args.train_subjects),
        "val": parse_subject_list(args.val_subjects),
        "test": parse_subject_list(args.test_subjects),
    }
    try:
        metadata = prepare_ppg_dalia_dataset(
            data_root=Path(args.data_root),
            output_dir=Path(args.output_dir),
            splits=splits,
            force=bool(args.force),
        )
    except FileNotFoundError as exc:
        print(str(exc))
        print("This script does not download PPG-DaLiA automatically. Place the dataset locally first.")
        return 1
    print(f"Prepared {metadata['total_subjects']} subject file(s) under {args.output_dir}")
    return 0


def prepare_ppg_dalia_dataset(
    data_root: Path,
    output_dir: Path,
    splits: dict[str, list[str]] | None = None,
    force: bool = False,
) -> dict:
    splits = splits or {
        "train": DEFAULT_TRAIN_SUBJECTS,
        "val": DEFAULT_VAL_SUBJECTS,
        "test": DEFAULT_TEST_SUBJECTS,
    }
    if not data_root.exists():
        raise FileNotFoundError(f"PPG-DaLiA data root does not exist: {data_root}")

    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "splits": splits,
        "subjects": {},
        "total_subjects": 0,
    }
    missing: list[str] = []
    for split_name, subjects in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for subject in subjects:
            pkl_path = find_subject_pickle(data_root, subject)
            if pkl_path is None:
                missing.append(subject)
                continue
            output_path = split_dir / f"{subject.upper()}.csv"
            if not output_path.exists() or force:
                frame = prepare_one_pickle(pkl_path)
                frame.to_csv(output_path, index=False)
            metadata["subjects"][subject.upper()] = {
                "split": split_name,
                "source": str(pkl_path),
                "prepared_csv": str(output_path),
            }
            metadata["total_subjects"] += 1

    metadata["missing_subjects"] = missing
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if metadata["total_subjects"] == 0:
        raise FileNotFoundError(
            f"No PPG-DaLiA pickle files found under {data_root}. Expected paths like S1/S1.pkl."
        )
    if missing:
        print(f"Warning: missing subject(s): {', '.join(missing)}")
    return metadata


def find_subject_pickle(data_root: Path, subject: str) -> Path | None:
    subject_id = subject.upper().removesuffix(".PKL")
    candidates = [
        data_root / subject_id / f"{subject_id}.pkl",
        data_root / f"{subject_id}.pkl",
        data_root / subject_id.lower() / f"{subject_id}.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(data_root.glob(f"**/{subject_id}.pkl"))
    return matches[0] if matches else None


def parse_subject_list(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = list(value)
    return [str(part).strip().upper() for part in parts if str(part).strip()]


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


if __name__ == "__main__":
    raise SystemExit(main())

