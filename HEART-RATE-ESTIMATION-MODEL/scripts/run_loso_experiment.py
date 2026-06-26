from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact.tinyppg_wrapper import count_trainable_parameters
from src.data.collate import collate_variable_length_ppg
from src.data.ppg_dalia_dataset import PPGDaLiAWindowDataset
from src.models.full_framework import build_full_framework, build_hr_model
from src.training.train_framework import train_full_framework
from src.training.train_hr import train_hr_estimator
from src.utils import write_json
from src.utils.checkpoint import load_checkpoint
from src.utils.config import apply_cli_overrides, load_config, normalize_config_paths, save_config
from src.utils.environment import collect_environment_info, resolve_device, validate_training_environment
from src.utils.seed import set_seed


@dataclass(frozen=True)
class LosoFold:
    test_subject: str
    train_subjects: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leave-one-subject-out TinyPPG HR experiments.")
    parser.add_argument("--config", default="configs/loso.yaml", help="LOSO config path.")
    parser.add_argument("--subjects", default="auto", help="'auto' or comma-separated subjects such as S1,S2,S3.")
    parser.add_argument("--folds", default="auto", help="'auto' or comma-separated held-out subjects.")
    parser.add_argument("--output-dir", default="runs/loso", help="Experiment output directory.")
    parser.add_argument("--smoke-only", action="store_true", help="Run a tiny CPU-safe LOSO smoke experiment.")
    parser.add_argument("--max-folds", type=int, default=None, help="Optional maximum number of folds to run.")
    parser.add_argument("--resume", nargs="?", const="latest", default=None, help="Resume each fold from latest or a checkpoint path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_config = normalize_config_paths(
        apply_cli_overrides(
            load_config(args.config),
            output_dir=args.output_dir,
            resume=args.resume,
        )
    )
    output_dir = Path(base_config.get("paths", {}).get("output_dir") or args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "environment.json", collect_environment_info())

    if args.smoke_only:
        base_config = make_smoke_config(base_config, output_dir)

    data_dir = prepared_data_dir(base_config)
    discovered = discover_subjects(data_dir)
    subjects = resolve_subject_selection(args.subjects, discovered)
    folds = make_loso_folds(subjects, resolve_fold_selection(args.folds, subjects))
    if args.max_folds is not None:
        folds = folds[: max(0, int(args.max_folds))]
    if not folds:
        raise ValueError("No LOSO folds selected.")

    report = validate_training_environment(
        base_config,
        data_root=base_config.get("data", {}).get("data_root"),
        output_dir=output_dir,
        require_dataset=True,
        require_tinyppg=True,
    )
    write_json(output_dir / "readiness_report.json", {"ok": report.ok, "info": report.info, "errors": report.errors, "warnings": report.warnings})
    if not report.ok:
        for error in report.errors:
            print(f"ERROR: {error}")
        return 2
    for warning in report.warnings:
        print(f"WARNING: {warning}")

    set_seed(int(base_config.get("seed", 42)))
    summary_rows: list[dict[str, Any]] = []
    for index, fold in enumerate(folds, start=1):
        print(f"[{index}/{len(folds)}] Running fold {fold.test_subject}")
        fold_dir = output_dir / f"fold_{fold.test_subject}"
        try:
            row = run_fold(base_config, fold, fold_dir, resume=args.resume)
        except Exception as exc:
            row = {
                "fold": fold.test_subject,
                "test_subject": fold.test_subject,
                "train_subjects": ",".join(fold.train_subjects),
                "status": "failed",
                "error": str(exc),
            }
            fold_dir.mkdir(parents=True, exist_ok=True)
            write_json(fold_dir / "fold_error.json", row)
            print(f"Fold {fold.test_subject} failed: {exc}")
        summary_rows.append(row)

    summary_path = output_dir / "fold_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    aggregate = aggregate_summary(summary_rows)
    write_json(output_dir / "aggregate_summary.json", aggregate)
    write_summary_md(output_dir / "summary.md", aggregate, summary_rows, report.warnings)
    print(f"LOSO experiment complete: {output_dir / 'summary.md'}")
    return 0 if aggregate.get("completed_folds", 0) > 0 else 1


def run_fold(base_config: dict[str, Any], fold: LosoFold, fold_dir: Path, resume: str | bool | None = None) -> dict[str, Any]:
    fold_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = fold_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    config = build_fold_config(base_config, fold, fold_dir, resume=resume)
    save_config(config, fold_dir / "config_used.yaml")

    print(f"  Training raw HR baseline for fold {fold.test_subject}")
    raw_result = train_hr_estimator(config, resume=resume)
    print(f"  Training TinyPPG-cropped HR model for fold {fold.test_subject}")
    tinyppg_result = train_full_framework(config, resume=resume)
    print(f"  Evaluating both models on held-out subject {fold.test_subject}")
    test_dataset = build_subject_dataset(config, fold.test_subject, max_windows_key="max_test_windows")
    evaluation = evaluate_checkpoints_on_same_windows(
        raw_checkpoint=raw_result["checkpoint_path"],
        tinyppg_checkpoint=tinyppg_result["checkpoint_path"],
        config=config,
        test_dataset=test_dataset,
        predictions_path=fold_dir / "predictions.csv",
    )

    raw_metrics = {**evaluation["raw"], "checkpoint_path": raw_result["checkpoint_path"]}
    tiny_metrics = {**evaluation["tinyppg_cropped"], "checkpoint_path": tinyppg_result["checkpoint_path"]}
    comparison = {
        **evaluation["comparison"],
        "fold": fold.test_subject,
        "test_subject": fold.test_subject,
        "train_subjects": fold.train_subjects,
        "raw_checkpoint": raw_result["checkpoint_path"],
        "tinyppg_checkpoint": tinyppg_result["checkpoint_path"],
    }
    write_json(metrics_dir / "raw_metrics.json", raw_metrics)
    write_json(metrics_dir / "tinyppg_cropped_metrics.json", tiny_metrics)
    write_json(metrics_dir / "comparison.json", comparison)

    return {
        "fold": fold.test_subject,
        "test_subject": fold.test_subject,
        "train_subjects": ",".join(fold.train_subjects),
        "status": "completed",
        "raw_mae": raw_metrics.get("mae"),
        "raw_rmse": raw_metrics.get("rmse"),
        "raw_bias": raw_metrics.get("bias"),
        "tinyppg_mae": tiny_metrics.get("mae"),
        "tinyppg_rmse": tiny_metrics.get("rmse"),
        "tinyppg_bias": tiny_metrics.get("bias"),
        "mae_improvement_raw_minus_tinyppg": comparison.get("mae_improvement_raw_minus_tinyppg"),
        "rmse_improvement_raw_minus_tinyppg": comparison.get("rmse_improvement_raw_minus_tinyppg"),
        "test_windows": raw_metrics.get("n"),
        "mean_percent_signal_removed": comparison.get("mean_percent_signal_removed"),
        "too_short_window_rate": comparison.get("too_short_window_rate"),
        "all_noisy_window_rate": comparison.get("all_noisy_window_rate"),
        "tinyppg_trainable_parameters": comparison.get("tinyppg_trainable_parameters"),
        "raw_checkpoint": raw_result["checkpoint_path"],
        "tinyppg_checkpoint": tinyppg_result["checkpoint_path"],
    }


def build_fold_config(
    base_config: dict[str, Any],
    fold: LosoFold,
    fold_dir: Path,
    resume: str | bool | None = None,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    data_cfg = config.setdefault("data", {})
    data_cfg["train_subjects"] = list(fold.train_subjects)
    data_cfg["val_subjects"] = []
    data_cfg["test_subjects"] = [fold.test_subject]

    paths = config.setdefault("paths", {})
    paths["output_dir"] = str(fold_dir)
    paths["checkpoint_dir"] = str(fold_dir / "checkpoints")
    paths["log_dir"] = str(fold_dir / "logs")
    config.setdefault("project", {})["output_dir"] = str(fold_dir)

    training = config.setdefault("training", {})
    training["hr_checkpoint_dir"] = str(fold_dir / "checkpoints" / "raw")
    training["framework_checkpoint_dir"] = str(fold_dir / "checkpoints" / "tinyppg_cropped")
    if resume is not None:
        config["resume"] = resume
        training["resume"] = resume
    return config


def evaluate_checkpoints_on_same_windows(
    raw_checkpoint: str | Path,
    tinyppg_checkpoint: str | Path,
    config: dict[str, Any],
    test_dataset,
    predictions_path: str | Path | None = None,
) -> dict[str, Any]:
    device = resolve_device(str(config.get("device", "cpu")))
    raw_model = load_raw_model(raw_checkpoint, config, device)
    framework = load_tinyppg_framework(tinyppg_checkpoint, config, device)
    raw_model.eval()
    framework.eval()

    loader = DataLoader(
        test_dataset,
        batch_size=int(config.get("evaluation", {}).get("batch_size", config.get("training", {}).get("batch_size", 16))),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_variable_length_ppg,
    )
    max_batches = config.get("evaluation", {}).get("max_test_batches", config.get("data", {}).get("max_test_batches"))
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            ppg = batch["padded_ppg"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            target = batch["hr_label"].to(device)
            raw_prediction = raw_model(ppg, valid_mask)
            tiny_output = framework(ppg, valid_mask)
            tiny_prediction = tiny_output["predicted_hr"]
            valid_lengths = [int(value) for value in batch["valid_mask"].sum(dim=1).tolist()]
            rows.extend(
                build_prediction_rows(
                    metadata=batch["metadata"],
                    valid_lengths=valid_lengths,
                    true_hr=target.detach().cpu().numpy(),
                    raw_prediction=raw_prediction.detach().cpu().numpy(),
                    tinyppg_prediction=tiny_prediction.detach().cpu().numpy(),
                    tiny_output=tiny_output,
                    min_output_samples=int(config.get("cropper", {}).get("min_output_samples", 1)),
                )
            )

    if predictions_path is not None:
        path = Path(predictions_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    raw_metrics = prediction_metrics(rows, "raw_prediction")
    tiny_metrics = prediction_metrics(rows, "tinyppg_prediction")
    comparison = cleanup_and_improvement_metrics(rows, raw_metrics, tiny_metrics)
    tinyppg = getattr(framework.artifact_detector, "tinyppg", framework.artifact_detector)
    comparison["tinyppg_trainable_parameters"] = count_trainable_parameters(tinyppg)
    comparison["tinyppg_frozen"] = comparison["tinyppg_trainable_parameters"] == 0
    return {
        "raw": raw_metrics,
        "tinyppg_cropped": tiny_metrics,
        "comparison": comparison,
        "predictions": rows,
    }


def build_prediction_rows(
    metadata: list[dict[str, Any]],
    valid_lengths: list[int],
    true_hr: np.ndarray,
    raw_prediction: np.ndarray,
    tinyppg_prediction: np.ndarray,
    tiny_output: dict[str, Any],
    min_output_samples: int = 1,
) -> list[dict[str, Any]]:
    artifact_mask = tiny_output["artifact_mask"].detach().cpu().numpy().astype(bool)
    crop_metadata = list(tiny_output.get("cropping_metadata", []))
    rows = []
    for idx, meta in enumerate(metadata):
        length = int(valid_lengths[idx])
        mask = artifact_mask[idx, :length]
        crop = crop_metadata[idx] if idx < len(crop_metadata) else {}
        subject = str(meta.get("subject") or Path(str(meta.get("source", ""))).stem or "")
        raw_value = float(raw_prediction[idx])
        tiny_value = float(tinyppg_prediction[idx])
        truth = float(true_hr[idx])
        raw_error = raw_value - truth
        tiny_error = tiny_value - truth
        cropped_length = int(crop.get("cropped_length", 0))
        too_short = bool(crop.get("kept_short_clean_segment", False)) or cropped_length < int(min_output_samples)
        rows.append(
            {
                "subject": subject,
                "source": meta.get("source"),
                "start": int(meta.get("start", 0)),
                "stop": int(meta.get("stop", 0)),
                "true_hr": truth,
                "raw_prediction": raw_value,
                "tinyppg_prediction": tiny_value,
                "raw_error": raw_error,
                "tinyppg_error": tiny_error,
                "raw_absolute_error": abs(raw_error),
                "tinyppg_absolute_error": abs(tiny_error),
                "absolute_error_improvement_raw_minus_tinyppg": abs(raw_error) - abs(tiny_error),
                "artifact_percent": float(mask.mean() * 100.0) if mask.size else 0.0,
                "original_length": int(crop.get("original_length", length)),
                "cropped_length": cropped_length,
                "percent_signal_removed": _safe_optional_float(crop.get("percent_removed")),
                "removed_artifact_segments": int(crop.get("number_of_removed_segments", 0)),
                "all_clean": bool(crop.get("all_clean", False)),
                "all_noisy": bool(crop.get("all_noisy", False)),
                "cropped_too_short": too_short,
            }
        )
    return rows


def load_raw_model(checkpoint_path: str | Path, config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    payload = load_checkpoint(checkpoint_path, map_location=device)
    model = build_hr_model(payload.get("model_config", config.get("model", {}))).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model


def load_tinyppg_framework(checkpoint_path: str | Path, config: dict[str, Any], device: torch.device):
    payload = load_checkpoint(checkpoint_path, map_location=device)
    framework = build_full_framework(config, device=device)
    framework.hr_model.load_state_dict(payload["hr_model_state_dict"])
    framework.assert_tinyppg_frozen()
    return framework


def build_subject_dataset(config: dict[str, Any], subject: str, max_windows_key: str = "max_windows") -> PPGDaLiAWindowDataset:
    data_cfg = config.get("data", {})
    return PPGDaLiAWindowDataset(
        data_dir=prepared_data_dir(config),
        sampling_rate_hz=float(data_cfg.get("sampling_rate_hz", 64.0)),
        window_sec=float(data_cfg.get("window_sec", 30.0)),
        step_sec=data_cfg.get("step_sec"),
        preprocess=config.get("preprocessing", {}),
        max_windows=data_cfg.get(max_windows_key),
        subjects=[subject],
    )


def discover_subjects(data_dir: str | Path) -> list[str]:
    root = Path(data_dir)
    if not root.exists():
        return []
    subjects = {path.stem.upper() for path in root.glob("S*.csv") if path.is_file()}
    return sorted(subjects, key=subject_sort_key)


def resolve_subject_selection(value: str | Iterable[str], discovered: list[str]) -> list[str]:
    if isinstance(value, str) and value.strip().lower() == "auto":
        subjects = list(discovered)
    else:
        subjects = parse_subject_list(value)
    missing = [subject for subject in subjects if subject not in set(discovered)]
    if missing:
        raise ValueError(f"Subject(s) not found in prepared data: {', '.join(missing)}")
    if len(subjects) < 2:
        raise ValueError("LOSO needs at least two subjects.")
    return sorted(subjects, key=subject_sort_key)


def resolve_fold_selection(value: str | Iterable[str], subjects: list[str]) -> list[str]:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return list(subjects)
    folds = parse_subject_list(value)
    missing = [subject for subject in folds if subject not in set(subjects)]
    if missing:
        raise ValueError(f"Fold subject(s) were not selected for the experiment: {', '.join(missing)}")
    return sorted(folds, key=subject_sort_key)


def make_loso_folds(subjects: list[str], fold_subjects: list[str] | None = None) -> list[LosoFold]:
    selected_folds = fold_subjects or subjects
    folds = []
    for test_subject in selected_folds:
        train_subjects = [subject for subject in subjects if subject != test_subject]
        if not train_subjects:
            raise ValueError(f"Fold {test_subject} has no training subjects.")
        folds.append(LosoFold(test_subject=test_subject, train_subjects=train_subjects))
    return folds


def parse_subject_list(value: str | Iterable[str]) -> list[str]:
    parts = value.split(",") if isinstance(value, str) else list(value)
    return [str(part).strip().upper().removesuffix(".CSV") for part in parts if str(part).strip()]


def subject_sort_key(subject: str) -> tuple[int, str]:
    text = str(subject).upper()
    try:
        return int(text.lstrip("S")), text
    except ValueError:
        return 9999, text


def prepared_data_dir(config: dict[str, Any]) -> Path:
    data_cfg = config.get("data", {})
    return Path(data_cfg.get("prepared_dir") or data_cfg.get("data_dir") or "data/input/prepared")


def prediction_metrics(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, float | int | None]:
    true_values = np.asarray([row["true_hr"] for row in rows], dtype=float)
    predicted = np.asarray([row[prediction_key] for row in rows], dtype=float)
    valid = np.isfinite(true_values) & np.isfinite(predicted)
    if valid.sum() == 0:
        return {"mae": None, "rmse": None, "bias": None, "mean_error": None, "n": 0}
    errors = predicted[valid] - true_values[valid]
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(math.sqrt(np.mean(np.square(errors)))),
        "bias": float(np.mean(errors)),
        "mean_error": float(np.mean(errors)),
        "n": int(valid.sum()),
    }


def cleanup_and_improvement_metrics(
    rows: list[dict[str, Any]],
    raw_metrics: dict[str, Any],
    tiny_metrics: dict[str, Any],
) -> dict[str, Any]:
    mean_removed = _mean([row.get("percent_signal_removed") for row in rows])
    artifact_percent = _mean([row.get("artifact_percent") for row in rows])
    return {
        "mae_improvement_raw_minus_tinyppg": _delta(raw_metrics.get("mae"), tiny_metrics.get("mae")),
        "rmse_improvement_raw_minus_tinyppg": _delta(raw_metrics.get("rmse"), tiny_metrics.get("rmse")),
        "bias_delta_raw_minus_tinyppg": _delta(raw_metrics.get("bias"), tiny_metrics.get("bias")),
        "mean_percent_signal_removed": mean_removed,
        "mean_artifact_percent": artifact_percent,
        "too_short_window_rate": _rate(rows, "cropped_too_short"),
        "all_noisy_window_rate": _rate(rows, "all_noisy"),
        "all_clean_window_rate": _rate(rows, "all_clean"),
        "test_windows": len(rows),
    }


def aggregate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    failed = [row for row in rows if row.get("status") != "completed"]
    return {
        "total_folds": len(rows),
        "completed_folds": len(completed),
        "failed_folds": len(failed),
        "failed_subjects": [row.get("test_subject") for row in failed],
        "raw_mae": numeric_summary(completed, "raw_mae"),
        "tinyppg_mae": numeric_summary(completed, "tinyppg_mae"),
        "raw_rmse": numeric_summary(completed, "raw_rmse"),
        "tinyppg_rmse": numeric_summary(completed, "tinyppg_rmse"),
        "mae_improvement_raw_minus_tinyppg": numeric_summary(completed, "mae_improvement_raw_minus_tinyppg"),
        "rmse_improvement_raw_minus_tinyppg": numeric_summary(completed, "rmse_improvement_raw_minus_tinyppg"),
        "mean_percent_signal_removed": numeric_summary(completed, "mean_percent_signal_removed"),
        "too_short_window_rate": numeric_summary(completed, "too_short_window_rate"),
        "all_noisy_window_rate": numeric_summary(completed, "all_noisy_window_rate"),
        "tinyppg_frozen_all_folds": bool(completed) and all(int(row.get("tinyppg_trainable_parameters", -1)) == 0 for row in completed),
    }


def numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    values = [float(row[key]) for row in rows if _is_finite_number(row.get(key))]
    if not values:
        return {"mean": None, "std": None, "median": None, "n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "n": int(arr.size),
    }


def make_smoke_config(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    config = deepcopy(config)
    smoke_dir = output_dir / "smoke_subjects"
    fs = float(config.get("data", {}).get("sampling_rate_hz", 64.0))
    window_sec = float(config.get("data", {}).get("window_sec", 30.0))
    create_smoke_subject_csvs(smoke_dir, fs=fs, duration_sec=max(32.0, window_sec + 2.0))
    data_cfg = config.setdefault("data", {})
    data_cfg["prepared_dir"] = str(smoke_dir)
    data_cfg["data_dir"] = str(smoke_dir)
    data_cfg["train_subjects"] = []
    data_cfg["val_subjects"] = []
    data_cfg["test_subjects"] = []
    data_cfg["max_windows"] = 4
    data_cfg["max_val_windows"] = 2
    data_cfg["max_test_windows"] = 2
    data_cfg["max_test_batches"] = 1
    data_cfg["num_workers"] = 0

    training = config.setdefault("training", {})
    training["epochs"] = 1
    training["max_epochs"] = 1
    training["batch_size"] = 2
    training["num_workers"] = 0
    training["max_train_batches"] = 1
    training["max_val_batches"] = 1
    training["use_amp"] = False

    evaluation = config.setdefault("evaluation", {})
    evaluation["batch_size"] = 2
    evaluation["max_test_batches"] = 1

    config["device"] = "cpu"
    config["use_amp"] = False
    config.setdefault("cropper", {})["empty_policy"] = "keep_original"
    config.setdefault("model", {})["name"] = "hr_estimator"
    return normalize_config_paths(config)


def create_smoke_subject_csvs(output_dir: Path, fs: float = 64.0, duration_sec: float = 32.0) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_count = int(round(fs * duration_sec))
    time = np.arange(sample_count, dtype=float) / fs
    for idx, subject in enumerate(("S1", "S2", "S3"), start=1):
        hr = 60.0 + idx * 5.0
        freq = hr / 60.0
        ppg = np.sin(2.0 * np.pi * freq * time) + 0.1 * np.sin(2.0 * np.pi * 0.25 * time)
        frame = pd.DataFrame(
            {
                "time": time,
                "ppg": ppg,
                "hr": np.full_like(time, hr, dtype=float),
                "acc_x": np.zeros_like(time),
                "acc_y": np.zeros_like(time),
                "acc_z": np.zeros_like(time),
            }
        )
        frame.to_csv(output_dir / f"{subject}.csv", index=False)


def write_summary_md(path: Path, aggregate: dict[str, Any], rows: list[dict[str, Any]], warnings: list[str]) -> None:
    lines = [
        "# LOSO TinyPPG HR Experiment Summary",
        "",
        f"- Folds completed: {aggregate.get('completed_folds')} / {aggregate.get('total_folds')}",
        f"- Failed subjects: {aggregate.get('failed_subjects')}",
        f"- Raw MAE mean: {_format_summary_value(aggregate.get('raw_mae'))}",
        f"- TinyPPG-cropped MAE mean: {_format_summary_value(aggregate.get('tinyppg_mae'))}",
        f"- MAE improvement raw minus TinyPPG: {_format_summary_value(aggregate.get('mae_improvement_raw_minus_tinyppg'))}",
        f"- Mean percent signal removed: {_format_summary_value(aggregate.get('mean_percent_signal_removed'))}",
        f"- TinyPPG frozen in all completed folds: {aggregate.get('tinyppg_frozen_all_folds')}",
        "",
        "## Fold Results",
        "",
    ]
    for row in rows:
        if row.get("status") != "completed":
            lines.append(f"- {row.get('test_subject')}: failed ({row.get('error')})")
            continue
        lines.append(
            "- {subject}: raw MAE {raw}, TinyPPG MAE {tiny}, improvement {delta}, removed {removed}".format(
                subject=row.get("test_subject"),
                raw=_format_number(row.get("raw_mae")),
                tiny=_format_number(row.get("tinyppg_mae")),
                delta=_format_number(row.get("mae_improvement_raw_minus_tinyppg")),
                removed=_format_number(row.get("mean_percent_signal_removed")),
            )
        )
    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- None recorded.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_summary_value(summary: Any) -> str:
    if not isinstance(summary, dict):
        return str(summary)
    return _format_number(summary.get("mean"))


def _format_number(value: Any) -> str:
    return "n/a" if not _is_finite_number(value) else f"{float(value):.3f}"


def _mean(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if _is_finite_number(value)]
    if not finite:
        return None
    return float(np.mean(np.asarray(finite, dtype=float)))


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(np.mean([bool(row.get(key, False)) for row in rows]) * 100.0)


def _delta(left: Any, right: Any) -> float | None:
    if not _is_finite_number(left) or not _is_finite_number(right):
        return None
    return float(left) - float(right)


def _safe_optional_float(value: Any) -> float | None:
    if not _is_finite_number(value):
        return None
    return float(value)


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
