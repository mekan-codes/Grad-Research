from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
import torch

from scripts.run_loso_experiment import (
    LosoFold,
    build_fold_config,
    build_prediction_rows,
    create_smoke_subject_csvs,
    discover_subjects,
    make_loso_folds,
)
from src.artifact.artifact_detector import adapt_tinyppg_output
from src.data.ppg_dalia_dataset import PPGDaLiAWindowDataset, SyntheticPPGWindowDataset
from src.training.train_hr import build_train_val_datasets


def test_discover_subjects_sorts_numerically(tmp_path: Path) -> None:
    for name in ("S10.csv", "S2.csv", "S1.csv"):
        (tmp_path / name).write_text("time,ppg,hr\n0,0,60\n1,1,60\n2,0,60\n", encoding="utf-8")

    assert discover_subjects(tmp_path) == ["S1", "S2", "S10"]


def test_loso_folds_hold_out_one_subject() -> None:
    folds = make_loso_folds(["S1", "S2", "S3"])

    assert [fold.test_subject for fold in folds] == ["S1", "S2", "S3"]
    assert folds[0].train_subjects == ["S2", "S3"]
    assert folds[1].train_subjects == ["S1", "S3"]
    assert folds[2].train_subjects == ["S1", "S2"]


def test_train_val_split_never_contains_held_out_subject(tmp_path: Path) -> None:
    create_smoke_subject_csvs(tmp_path)
    config = {
        "seed": 7,
        "device": "cpu",
        "data": {
            "dataset": "ppg_dalia",
            "prepared_dir": str(tmp_path),
            "sampling_rate_hz": 64,
            "window_sec": 30,
            "step_sec": 15,
            "val_fraction": 0.5,
        },
        "preprocessing": {
            "normalize": "zscore",
            "detrend": False,
            "bandpass": {"enabled": False},
        },
        "training": {"batch_size": 2, "num_workers": 0},
    }
    fold_config = build_fold_config(config, LosoFold(test_subject="S2", train_subjects=["S1", "S3"]), tmp_path / "fold_S2")

    train_dataset, val_dataset = build_train_val_datasets(fold_config)

    assert _subjects_in_dataset(train_dataset) <= {"S1", "S3"}
    assert _subjects_in_dataset(val_dataset) <= {"S1", "S3"}
    assert "S2" not in _subjects_in_dataset(train_dataset)
    assert "S2" not in _subjects_in_dataset(val_dataset)


def test_requested_subjects_fail_when_csv_is_missing(tmp_path: Path) -> None:
    create_smoke_subject_csvs(tmp_path)
    (tmp_path / "S2.csv").unlink()

    with pytest.raises(FileNotFoundError, match="S2"):
        PPGDaLiAWindowDataset(
            data_dir=tmp_path,
            sampling_rate_hz=64,
            window_sec=30,
            subjects=["S1", "S2"],
        )


def test_synthetic_dataset_is_deterministic_per_index() -> None:
    dataset = SyntheticPPGWindowDataset(num_samples=3, sampling_rate_hz=8, window_sec=2, seed=123)

    first = dataset[0]["ppg"]
    second = dataset[0]["ppg"]

    assert torch.equal(first, second)


def test_tinyppg_adapter_handles_time_last_class_layout() -> None:
    logits = torch.tensor(
        [
            [[0.0, 2.0], [3.0, 0.0], [0.0, 2.0]],
            [[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]],
        ]
    )

    probability = adapt_tinyppg_output(
        logits,
        target_length=3,
        artifact_output_mode="logits",
        artifact_class_index=1,
    )

    assert probability.shape == (2, 3)
    assert probability[0, 0] > probability[0, 1]


def test_prediction_rows_put_raw_and_tinyppg_on_same_windows() -> None:
    tiny_output = {
        "artifact_mask": torch.tensor([[True, False, False], [False, False, True]]),
        "cropping_metadata": [
            {
                "original_length": 3,
                "cropped_length": 2,
                "percent_removed": 33.333333,
                "number_of_removed_segments": 1,
                "all_clean": False,
                "all_noisy": False,
            },
            {
                "original_length": 3,
                "cropped_length": 2,
                "percent_removed": 33.333333,
                "number_of_removed_segments": 1,
                "all_clean": False,
                "all_noisy": False,
            },
        ],
    }

    rows = build_prediction_rows(
        metadata=[
            {"subject": "S1", "source": "S1.csv", "start": 0, "stop": 3},
            {"subject": "S1", "source": "S1.csv", "start": 3, "stop": 6},
        ],
        valid_lengths=[3, 3],
        true_hr=np.array([60.0, 70.0]),
        raw_prediction=np.array([62.0, 65.0]),
        tinyppg_prediction=np.array([61.0, 72.0]),
        tiny_output=tiny_output,
        min_output_samples=1,
    )

    assert [(row["start"], row["stop"]) for row in rows] == [(0, 3), (3, 6)]
    assert [row["raw_prediction"] for row in rows] == [62.0, 65.0]
    assert [row["tinyppg_prediction"] for row in rows] == [61.0, 72.0]
    assert rows[0]["absolute_error_improvement_raw_minus_tinyppg"] == 1.0


def test_loso_smoke_runner(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    checkpoint = root.parent / "Tiny-PPG-master" / "Save_Model" / "model_parameter-2023-5-31-1.pkl"
    if not checkpoint.exists():
        pytest.skip("Local TinyPPG checkpoint is not available")

    output_dir = tmp_path / "loso_smoke"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_loso_experiment.py",
            "--config",
            "configs/loso.yaml",
            "--smoke-only",
            "--max-folds",
            "1",
            "--output-dir",
            str(output_dir),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (output_dir / "fold_summary.csv").exists()
    assert (output_dir / "aggregate_summary.json").exists()
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "fold_S1" / "metrics" / "comparison.json").exists()
    assert (output_dir / "fold_S1" / "predictions.csv").exists()


def _subjects_in_dataset(dataset: Any) -> set[str]:
    subjects = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        subjects.add(str(sample["metadata"]["subject"]))
    return subjects
