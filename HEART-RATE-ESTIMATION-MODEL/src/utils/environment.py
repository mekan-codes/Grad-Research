from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch


REQUIRED_PACKAGES = ("numpy", "pandas", "scipy", "torch", "yaml")
PLOTTING_PACKAGES = ("matplotlib",)


@dataclass
class EnvironmentReport:
    ok: bool
    info: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def resolve_device(requested: str | None = "auto") -> torch.device:
    value = (requested or "auto").lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(value)


def collect_environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": None,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "git_commit": get_git_commit(),
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
    }
    if torch.cuda.is_available():
        try:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            info["cuda_device_name"] = f"unavailable: {exc}"
    return info


def validate_training_environment(
    config: dict[str, Any],
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    require_dataset: bool = True,
    require_tinyppg: bool = True,
    require_plotting: bool = False,
) -> EnvironmentReport:
    info = collect_environment_info()
    errors: list[str] = []
    warnings: list[str] = []

    required = list(REQUIRED_PACKAGES)
    if require_plotting:
        required.extend(PLOTTING_PACKAGES)
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    info["required_packages"] = {name: importlib.util.find_spec(name) is not None for name in required}
    if missing:
        errors.append(f"Missing required package(s): {', '.join(missing)}")

    tiny_cfg = config.get("tinyppg", {})
    checkpoint = _resolve_path(tiny_cfg.get("checkpoint_path"))
    info["tinyppg_checkpoint"] = str(checkpoint) if checkpoint else None
    info["tinyppg_checkpoint_exists"] = bool(checkpoint and checkpoint.exists())
    if require_tinyppg and not info["tinyppg_checkpoint_exists"]:
        errors.append(f"TinyPPG checkpoint not found: {tiny_cfg.get('checkpoint_path')}")

    data_cfg = config.get("data", {})
    raw_root = data_root if data_root is not None else data_cfg.get("data_root")
    root = Path(str(raw_root)) if raw_root not in {None, ""} else None
    info["dataset_root"] = str(root) if root is not None else None
    info["dataset_root_exists"] = root.exists() if root is not None else False
    prepared_root = _resolve_path(data_cfg.get("prepared_dir") or data_cfg.get("data_dir"))
    prepared_csvs_exist = _has_prepared_csvs(prepared_root)
    info["prepared_data_root"] = str(prepared_root) if prepared_root is not None else None
    info["prepared_data_exists"] = bool(prepared_root and prepared_root.exists())
    info["prepared_csvs_exist"] = prepared_csvs_exist
    dataset = str(config.get("data", {}).get("dataset", "")).lower()
    if require_dataset and dataset != "synthetic":
        raw_exists = bool(root and root.exists())
        if not raw_exists and not prepared_csvs_exist:
            errors.append(
                "Dataset data was not found. "
                f"Raw root: {root}; prepared CSV root: {prepared_root}"
            )
        elif not raw_exists and prepared_csvs_exist:
            warnings.append(f"Raw dataset root is unavailable; using prepared CSVs at {prepared_root}.")

    out = Path(output_dir or config.get("paths", {}).get("output_dir") or config.get("project", {}).get("output_dir") or "runs/default")
    info["output_dir"] = str(out)
    writable = _check_writable(out)
    info["output_dir_writable"] = writable
    if not writable:
        errors.append(f"Output directory is not writable: {out}")

    if str(config.get("device", "auto")).lower().startswith("cuda") and not torch.cuda.is_available():
        warnings.append("CUDA was requested but is not available; training will fall back to CPU.")
    if config.get("use_amp", False) and not torch.cuda.is_available():
        warnings.append("AMP requested, but CUDA is unavailable; AMP will be disabled.")

    return EnvironmentReport(ok=not errors, info=info, errors=errors, warnings=warnings)


def print_environment_report(report: EnvironmentReport) -> None:
    print("Training environment")
    print("--------------------")
    for key, value in report.info.items():
        print(f"{key}: {value}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    print("status:", "OK" if report.ok else "NOT READY")


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def _resolve_path(value: Any) -> Path | None:
    if value in {None, ""}:
        return None
    raw = Path(str(value))
    if raw.is_absolute():
        return raw
    project_root = Path(__file__).resolve().parents[2]
    repo_root = project_root.parent
    for root in (Path.cwd(), project_root, repo_root):
        candidate = root / raw
        if candidate.exists():
            return candidate
    return project_root / raw


def _has_prepared_csvs(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    if any(path.glob("*.csv")):
        return True
    return any(any((path / split).glob("*.csv")) for split in ("train", "val", "test"))


def _check_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False
