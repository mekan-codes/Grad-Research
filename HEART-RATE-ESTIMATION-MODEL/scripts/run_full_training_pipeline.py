from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_ppg_dalia import prepare_ppg_dalia_dataset
from src.artifact.calibration import calibrate_artifact_thresholds
from src.artifact.cropper import CropperConfig
from src.artifact.tinyppg_wrapper import count_trainable_parameters
from src.data.ppg_dalia_dataset import SyntheticPPGWindowDataset
from src.models.full_framework import build_full_framework, build_hr_model
from src.training.evaluate import evaluate_checkpoint
from src.training.train_framework import train_full_framework
from src.training.train_hr import build_train_val_datasets, train_hr_estimator
from src.utils.config import apply_cli_overrides, load_config, normalize_config_paths, save_config
from src.utils.environment import collect_environment_info, resolve_device, validate_training_environment
from src.utils.logging import configure_run_logger
from src.utils.seed import set_seed
from src.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full TinyPPG + HR training pipeline.")
    parser.add_argument("--config", default="configs/train_workspace.yaml", help="Training config path.")
    parser.add_argument("--data-root", default=None, help="Override raw PPG-DaLiA root.")
    parser.add_argument("--output-dir", default=None, help="Run output directory.")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip PPG-DaLiA preparation.")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline HR estimator training.")
    parser.add_argument("--skip-full-framework", action="store_true", help="Skip TinyPPG-crop framework training.")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip TinyPPG threshold calibration.")
    parser.add_argument("--smoke-only", action="store_true", help="Run a tiny CPU-safe synthetic smoke pipeline.")
    parser.add_argument("--resume", nargs="?", const="latest", default=None, help="Resume from latest or a checkpoint path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = normalize_config_paths(
        apply_cli_overrides(
            load_config(args.config),
            data_root=args.data_root,
            output_dir=args.output_dir,
            resume=args.resume,
        )
    )
    if args.smoke_only:
        config = _force_smoke_overrides(config)

    output_dir = Path(config.get("paths", {}).get("output_dir") or config.get("project", {}).get("output_dir") or "runs/default")
    dirs = _make_run_dirs(output_dir)
    logger = configure_run_logger(dirs["logs"])
    write_json(dirs["metrics"] / "environment.json", collect_environment_info())

    logger.info("Starting full training pipeline")
    logger.info("Command: %s", " ".join(sys.argv))
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "auto")))
    config["device"] = str(device)
    if device.type != "cuda":
        config["use_amp"] = False
        config.setdefault("training", {})["use_amp"] = False
    logger.info("Using device: %s", device)
    save_config(config, output_dir / "config_used.yaml")

    report = validate_training_environment(
        config,
        data_root=config.get("data", {}).get("data_root"),
        output_dir=output_dir,
        require_dataset=not args.smoke_only and str(config.get("data", {}).get("dataset", "")).lower() != "synthetic",
        require_tinyppg=True,
    )
    write_json(dirs["metrics"] / "readiness_report.json", {"ok": report.ok, "info": report.info, "errors": report.errors, "warnings": report.warnings})
    for warning in report.warnings:
        logger.warning(warning)
    if not report.ok:
        for error in report.errors:
            logger.error(error)
        return 2

    if not args.skip_prepare and not args.smoke_only:
        _prepare_if_needed(config, logger)
    elif args.skip_prepare:
        logger.info("Skipping dataset preparation")

    dataset_summary = _dataset_summary(config, logger)
    write_json(dirs["metrics"] / "dataset_summary.json", dataset_summary)

    smoke_result = _smoke_forward_pass(config, device, logger)
    write_json(dirs["metrics"] / "smoke_forward.json", smoke_result)
    logger.info("TinyPPG trainable parameters: %s", smoke_result["tinyppg_trainable_parameters"])
    logger.info("HR model trainable parameters: %s", smoke_result["hr_model_trainable_parameters"])

    calibration = None
    if not args.skip_calibration:
        calibration = _run_calibration(config, device, dirs["metrics"], logger)
    else:
        logger.info("Skipping artifact threshold calibration")

    baseline_result = None
    baseline_metrics = None
    if not args.skip_baseline:
        logger.info("Training baseline HR estimator")
        baseline_result = train_hr_estimator(config, resume=args.resume, logger=logger)
        baseline_metrics = evaluate_checkpoint(baseline_result["checkpoint_path"], config=config)
        write_json(dirs["metrics"] / "baseline_metrics.json", {**baseline_metrics, **baseline_result})
    else:
        logger.info("Skipping baseline training")

    full_result = None
    full_metrics = None
    if not args.skip_full_framework:
        logger.info("Training full TinyPPG-crop framework")
        full_result = train_full_framework(config, resume=args.resume, logger=logger)
        full_metrics = evaluate_checkpoint(full_result["checkpoint_path"], config=config)
        write_json(dirs["metrics"] / "full_framework_metrics.json", {**full_metrics, **full_result})
    else:
        logger.info("Skipping full framework training")

    comparison = _build_comparison(baseline_metrics, full_metrics, baseline_result, full_result, calibration)
    write_json(dirs["metrics"] / "comparison.json", comparison)
    logger.info("Final comparison: %s", comparison)
    _write_summary(output_dir / "summary.md", comparison, report.warnings, calibration)
    logger.info("Pipeline complete. Summary: %s", output_dir / "summary.md")
    return 0


def _force_smoke_overrides(config: dict[str, Any]) -> dict[str, Any]:
    config = apply_cli_overrides(config)
    config["device"] = "cpu"
    config["use_amp"] = False
    data = config.setdefault("data", {})
    data["dataset"] = "synthetic"
    data["synthetic_samples"] = min(int(data.get("synthetic_samples", 8)), 8)
    data["num_workers"] = 0
    training = config.setdefault("training", {})
    training["epochs"] = 1
    training["max_epochs"] = 1
    training["batch_size"] = min(int(training.get("batch_size", 2)), 2)
    training["num_workers"] = 0
    training["use_amp"] = False
    training["max_train_batches"] = 1
    training["max_val_batches"] = 1
    cropper = config.setdefault("cropper", {})
    cropper["empty_policy"] = "keep_original"
    cropper["min_output_samples"] = max(1, int(cropper.get("min_output_samples", 8)))
    return config


def _make_run_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "logs": output_dir / "logs",
        "checkpoints": output_dir / "checkpoints",
        "metrics": output_dir / "metrics",
        "plots": output_dir / "plots",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _prepare_if_needed(config: dict[str, Any], logger: Any) -> None:
    data_cfg = config.get("data", {})
    prepared_dir = _configured_prepared_dir(data_cfg)
    if _has_prepared_csvs(prepared_dir):
        logger.info("Prepared data already exists at %s", prepared_dir)
        return

    data_dir = data_cfg.get("data_dir")
    if data_dir is not None and _has_prepared_csvs(Path(data_dir)):
        logger.info("Using prepared CSV data at %s", data_dir)
        data_cfg["prepared_dir"] = str(data_dir)
        return

    raw_root = Path(data_cfg.get("data_root", "data/raw/PPG_Dalia"))
    if not raw_root.exists():
        raise FileNotFoundError(
            "No prepared CSV data was found and the raw PPG-DaLiA root does not exist. "
            f"Checked prepared data at {prepared_dir} and raw data at {raw_root}."
        )

    logger.info("Preparing PPG-DaLiA from %s", raw_root)
    prepare_ppg_dalia_dataset(
        data_root=raw_root,
        output_dir=prepared_dir,
        splits={
            "train": list(data_cfg.get("train_subjects", [])),
            "val": list(data_cfg.get("val_subjects", [])),
            "test": list(data_cfg.get("test_subjects", [])),
        },
        force=False,
    )


def _configured_prepared_dir(data_cfg: dict[str, Any]) -> Path:
    return Path(data_cfg.get("prepared_dir") or data_cfg.get("data_dir") or "data/prepared/ppg_dalia")


def _has_prepared_csvs(path: Path) -> bool:
    if not path.exists():
        return False
    if any(path.glob("*.csv")):
        return True
    return any(any((path / split).glob("*.csv")) for split in ("train", "val", "test"))


def _dataset_summary(config: dict[str, Any], logger: Any) -> dict[str, Any]:
    try:
        train_dataset, val_dataset = build_train_val_datasets(config)
    except Exception as exc:
        logger.warning("Could not build dataset summary before training: %s", exc)
        return {"available": False, "reason": str(exc)}
    summary = {
        "available": True,
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "dataset": config.get("data", {}).get("dataset"),
        "window_sec": config.get("data", {}).get("window_sec"),
        "sampling_rate_hz": config.get("data", {}).get("sampling_rate_hz"),
    }
    logger.info("Dataset summary: %s", summary)
    return summary


def _smoke_forward_pass(config: dict[str, Any], device: torch.device, logger: Any) -> dict[str, Any]:
    logger.info("Running smoke forward pass and TinyPPG freeze check")
    framework = build_full_framework(config, device=device)
    framework.assert_tinyppg_frozen()
    dataset = SyntheticPPGWindowDataset(
        num_samples=2,
        sampling_rate_hz=float(config.get("data", {}).get("sampling_rate_hz", 64)),
        window_sec=float(config.get("data", {}).get("window_sec", 30)),
        seed=int(config.get("seed", 42)),
    )
    batch = torch.stack([dataset[0]["ppg"], dataset[1]["ppg"]]).to(device)
    valid_mask = torch.ones_like(batch, dtype=torch.bool)
    with torch.no_grad():
        output = framework(batch, valid_mask)
    tinyppg = getattr(framework.artifact_detector, "tinyppg", framework.artifact_detector)
    return {
        "predicted_shape": list(output["predicted_hr"].shape),
        "cropped_shape": list(output["cropped_signal"].shape),
        "tinyppg_trainable_parameters": count_trainable_parameters(tinyppg),
        "hr_model_parameters": _count_parameters(framework.hr_model),
        "hr_model_trainable_parameters": _count_parameters(framework.hr_model, trainable_only=True),
    }


def _run_calibration(config: dict[str, Any], device: torch.device, metrics_dir: Path, logger: Any) -> dict[str, Any]:
    logger.info("Running TinyPPG artifact threshold calibration")
    framework = build_full_framework(config, device=device)
    _, val_dataset = build_train_val_datasets(config)
    calib_cfg = config.get("artifact_calibration", {})
    crop_cfg = config.get("cropper", {})
    cropper_config = CropperConfig(
        mode=str(crop_cfg.get("mode", "crop")),
        min_clean_samples=int(crop_cfg.get("min_clean_samples", 1)),
        min_output_samples=int(crop_cfg.get("min_output_samples", 1)),
        empty_policy=str(crop_cfg.get("empty_policy", "empty")),
        mask_fill_value=float(crop_cfg.get("mask_fill_value", 0.0)),
    )
    result = calibrate_artifact_thresholds(
        artifact_detector=framework.artifact_detector,
        samples=(val_dataset[idx] for idx in range(len(val_dataset))),
        thresholds=[float(value) for value in calib_cfg.get("thresholds", [0.5])],
        cropper_config=cropper_config,
        device=device,
        max_windows=calib_cfg.get("max_windows"),
        output_path=metrics_dir / "artifact_threshold_sweep.json",
        warn_mean_removed_above=float(calib_cfg.get("warn_mean_removed_above", 70.0)),
        warn_full_crop_above=float(calib_cfg.get("warn_full_crop_above", 20.0)),
        warn_too_short_above=float(calib_cfg.get("warn_too_short_above", 20.0)),
    )
    for warning in result.get("warnings", []):
        logger.warning(warning)
    return result


def _build_comparison(
    baseline_metrics: dict[str, float] | None,
    full_metrics: dict[str, float] | None,
    baseline_result: dict[str, Any] | None,
    full_result: dict[str, Any] | None,
    calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "baseline": baseline_metrics,
        "full_framework": full_metrics,
        "baseline_checkpoint": baseline_result.get("checkpoint_path") if baseline_result else None,
        "full_framework_checkpoint": full_result.get("checkpoint_path") if full_result else None,
        "artifact_calibration_warnings": calibration.get("warnings", []) if calibration else [],
        "mae_delta_baseline_minus_full": None,
        "rmse_delta_baseline_minus_full": None,
    }
    if baseline_metrics and full_metrics:
        comparison["mae_delta_baseline_minus_full"] = baseline_metrics.get("mae", 0.0) - full_metrics.get("mae", 0.0)
        comparison["rmse_delta_baseline_minus_full"] = baseline_metrics.get("rmse", 0.0) - full_metrics.get("rmse", 0.0)
    return comparison


def _write_summary(
    path: Path,
    comparison: dict[str, Any],
    environment_warnings: list[str],
    calibration: dict[str, Any] | None,
) -> None:
    lines = [
        "# PPG HR Training Run Summary",
        "",
        "This summary reports pipeline execution only. It is not a claim of real model quality unless real PPG-DaLiA training was run.",
        "",
        "## Results",
        "",
        f"- Baseline metrics: `{comparison.get('baseline')}`",
        f"- Full framework metrics: `{comparison.get('full_framework')}`",
        f"- MAE delta baseline minus full: `{comparison.get('mae_delta_baseline_minus_full')}`",
        f"- RMSE delta baseline minus full: `{comparison.get('rmse_delta_baseline_minus_full')}`",
        "",
        "## Checkpoints",
        "",
        f"- Baseline: `{comparison.get('baseline_checkpoint')}`",
        f"- Full framework: `{comparison.get('full_framework_checkpoint')}`",
        "",
        "## Artifact Calibration",
        "",
    ]
    if calibration:
        lines.append(f"- Windows evaluated: `{calibration.get('windows_evaluated')}`")
        lines.append(f"- Warnings: `{calibration.get('warnings', [])}`")
    else:
        lines.append("- Calibration was skipped.")
    lines.extend(["", "## Warnings", ""])
    warnings = environment_warnings + list(comparison.get("artifact_calibration_warnings", []))
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- None recorded.")
    lines.extend(
        [
            "",
            "## Next Steps",
            "",
            "- Confirm TinyPPG `artifact_output_mode` and threshold on labeled/visualized windows.",
            "- Run the workspace config on the GPU machine with the real PPG-DaLiA dataset.",
            "- Compare `metrics/baseline_metrics.json`, `metrics/full_framework_metrics.json`, and `metrics/comparison.json`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(param.numel() for param in module.parameters() if (param.requires_grad or not trainable_only))


if __name__ == "__main__":
    raise SystemExit(main())
