from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


BASELINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BASELINE_ROOT.parent
PROJECT_ROOT = WORKSPACE_ROOT / "HEART-RATE-ESTIMATION-MODEL"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_loso_experiment import (  # noqa: E402
    LosoFold,
    build_fold_config,
    build_subject_dataset,
    create_smoke_subject_csvs,
    discover_subjects,
    make_loso_folds,
    parse_subject_list,
    resolve_fold_selection,
    resolve_subject_selection,
)
from src.data.collate import collate_variable_length_ppg  # noqa: E402
from src.models.full_framework import build_hr_model  # noqa: E402
from src.training.train_hr import train_hr_estimator  # noqa: E402
from src.utils import write_json  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402
from src.utils.config import apply_cli_overrides, load_config, normalize_config_paths, save_config  # noqa: E402
from src.utils.environment import collect_environment_info, resolve_device, validate_training_environment  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw PPG leave-one-subject-out baseline training.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "loso.yaml"), help="Base config path.")
    parser.add_argument("--subjects", default="auto", help="'auto' or comma-separated subjects.")
    parser.add_argument("--folds", default="auto", help="'auto' or comma-separated held-out subjects.")
    parser.add_argument("--output-dir", default=str(BASELINE_ROOT / "runs" / "raw_loso"), help="Output directory.")
    parser.add_argument("--smoke-only", action="store_true", help="Run a tiny synthetic smoke baseline.")
    parser.add_argument("--max-folds", type=int, default=None, help="Optional maximum number of folds.")
    parser.add_argument("--resume", nargs="?", const="latest", default=None, help="Resume from latest or checkpoint.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = normalize_config_paths(
        apply_cli_overrides(
            load_config(args.config),
            output_dir=args.output_dir,
            resume=args.resume,
        )
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "environment.json", collect_environment_info())

    if args.smoke_only:
        config = make_smoke_config(config, output_dir)

    data_dir = Path(config.get("data", {}).get("prepared_dir") or config.get("data", {}).get("data_dir"))
    discovered = discover_subjects(data_dir)
    subjects = resolve_subject_selection(args.subjects, discovered)
    folds = make_loso_folds(subjects, resolve_fold_selection(args.folds, subjects))
    if args.max_folds is not None:
        folds = folds[: max(0, int(args.max_folds))]

    report = validate_training_environment(
        config,
        data_root=config.get("data", {}).get("data_root"),
        output_dir=output_dir,
        require_dataset=True,
        require_tinyppg=False,
    )
    write_json(output_dir / "readiness_report.json", {"ok": report.ok, "info": report.info, "errors": report.errors, "warnings": report.warnings})
    if not report.ok:
        for error in report.errors:
            print(f"ERROR: {error}")
        return 2

    set_seed(int(config.get("seed", 42)))
    rows = []
    for index, fold in enumerate(folds, start=1):
        print(f"[{index}/{len(folds)}] Running raw baseline fold {fold.test_subject}")
        fold_dir = output_dir / f"fold_{fold.test_subject}"
        try:
            rows.append(run_fold(config, fold, fold_dir, resume=args.resume))
        except Exception as exc:
            error_row = {
                "fold": fold.test_subject,
                "test_subject": fold.test_subject,
                "train_subjects": ",".join(fold.train_subjects),
                "status": "failed",
                "error": str(exc),
            }
            fold_dir.mkdir(parents=True, exist_ok=True)
            write_json(fold_dir / "fold_error.json", error_row)
            rows.append(error_row)

    pd.DataFrame(rows).to_csv(output_dir / "fold_summary.csv", index=False)
    aggregate = aggregate_summary(rows)
    write_json(output_dir / "aggregate_summary.json", aggregate)
    write_summary(output_dir / "summary.md", aggregate, rows)
    print(f"Raw baseline complete: {output_dir / 'summary.md'}")
    return 0 if aggregate["completed_folds"] > 0 else 1


def run_fold(config: dict[str, Any], fold: LosoFold, fold_dir: Path, resume: str | bool | None = None) -> dict[str, Any]:
    fold_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = fold_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fold_config = build_fold_config(config, fold, fold_dir, resume=resume)
    save_config(fold_config, fold_dir / "config_used.yaml")

    print(f"  Training raw HR baseline for fold {fold.test_subject}")
    train_result = train_hr_estimator(fold_config, resume=resume)
    print(f"  Evaluating raw HR baseline on held-out subject {fold.test_subject}")
    test_dataset = build_subject_dataset(fold_config, fold.test_subject, max_windows_key="max_test_windows")
    predictions = evaluate_raw_checkpoint(
        checkpoint_path=train_result["checkpoint_path"],
        config=fold_config,
        test_dataset=test_dataset,
        predictions_path=fold_dir / "predictions.csv",
    )
    metrics = raw_metrics(predictions)
    metrics["checkpoint_path"] = train_result["checkpoint_path"]
    write_json(metrics_dir / "raw_metrics.json", metrics)
    return {
        "fold": fold.test_subject,
        "test_subject": fold.test_subject,
        "train_subjects": ",".join(fold.train_subjects),
        "status": "completed",
        "raw_mae": metrics["mae"],
        "raw_rmse": metrics["rmse"],
        "raw_bias": metrics["bias"],
        "test_windows": metrics["n"],
        "raw_checkpoint": train_result["checkpoint_path"],
    }


def evaluate_raw_checkpoint(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    test_dataset,
    predictions_path: str | Path,
) -> list[dict[str, Any]]:
    device = resolve_device(str(config.get("device", "cpu")))
    payload = load_checkpoint(checkpoint_path, map_location=device)
    model = build_hr_model(payload.get("model_config", config.get("model", {}))).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

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
            target = batch["hr_label"].detach().cpu().numpy()
            prediction = model(ppg, valid_mask).detach().cpu().numpy()
            for item, truth, pred in zip(batch["metadata"], target, prediction):
                error = float(pred) - float(truth)
                rows.append(
                    {
                        "subject": item.get("subject"),
                        "source": item.get("source"),
                        "start": int(item.get("start", 0)),
                        "stop": int(item.get("stop", 0)),
                        "true_hr": float(truth),
                        "raw_prediction": float(pred),
                        "raw_error": error,
                        "raw_absolute_error": abs(error),
                    }
                )

    path = Path(predictions_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return rows


def raw_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    true_values = np.asarray([row["true_hr"] for row in rows], dtype=float)
    predicted = np.asarray([row["raw_prediction"] for row in rows], dtype=float)
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


def aggregate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    return {
        "total_folds": len(rows),
        "completed_folds": len(completed),
        "failed_folds": len(rows) - len(completed),
        "raw_mae": summarize(completed, "raw_mae"),
        "raw_rmse": summarize(completed, "raw_rmse"),
        "raw_bias": summarize(completed, "raw_bias"),
    }


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    values = [float(row[key]) for row in rows if is_number(row.get(key))]
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
    create_smoke_subject_csvs(smoke_dir)
    data = config.setdefault("data", {})
    data["prepared_dir"] = str(smoke_dir)
    data["data_dir"] = str(smoke_dir)
    data["train_subjects"] = []
    data["val_subjects"] = []
    data["test_subjects"] = []
    data["max_windows"] = 4
    data["max_val_windows"] = 2
    data["max_test_windows"] = 2
    data["max_test_batches"] = 1
    data["num_workers"] = 0

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
    return normalize_config_paths(config)


def write_summary(path: Path, aggregate: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Raw PPG LOSO Baseline Summary",
        "",
        f"- Folds completed: {aggregate['completed_folds']} / {aggregate['total_folds']}",
        f"- Raw MAE mean: {format_mean(aggregate['raw_mae'])}",
        f"- Raw RMSE mean: {format_mean(aggregate['raw_rmse'])}",
        "",
        "## Fold Results",
        "",
    ]
    for row in rows:
        if row.get("status") != "completed":
            lines.append(f"- {row.get('test_subject')}: failed ({row.get('error')})")
        else:
            lines.append(f"- {row['test_subject']}: MAE {format_number(row['raw_mae'])}, RMSE {format_number(row['raw_rmse'])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_mean(summary: dict[str, Any]) -> str:
    return format_number(summary.get("mean"))


def format_number(value: Any) -> str:
    return "n/a" if not is_number(value) else f"{float(value):.3f}"


def is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
