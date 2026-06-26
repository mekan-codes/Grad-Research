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

from src.artifact.artifact_detector import adapt_tinyppg_output
from src.artifact.tinyppg_wrapper import load_tinyppg
from src.data.ppg_dalia_dataset import PPGDaLiAWindowDataset
from src.utils.config import load_config, normalize_config_paths, resolve_project_path


MODES = ("artifact_probability", "clean_probability", "logits", "class_index")
THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect TinyPPG raw output semantics on prepared PPG-DaLiA windows.")
    parser.add_argument("--config", default="configs/train_workspace.yaml", help="Training config path.")
    parser.add_argument("--input", default="data/input/prepared/S1.csv", help="Prepared subject CSV to inspect.")
    parser.add_argument("--num-windows", type=int, default=3, help="Number of windows to inspect.")
    parser.add_argument("--output", default="runs/tinyppg_output_diagnostic.json", help="Diagnostic JSON output path.")
    parser.add_argument("--device", default="cpu", help="Device for TinyPPG inference.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = normalize_config_paths(load_config(args.config))
    csv_path = resolve_project_path(args.input)
    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    windows, metadata = load_windows(csv_path, config, args.num_windows)
    valid_mask = torch.ones_like(windows, dtype=torch.bool)
    tinyppg_input = normalize_batch(windows, valid_mask).unsqueeze(1)

    tiny_cfg = config.get("tinyppg", {})
    loaded = load_tinyppg(
        model_dir=tiny_cfg.get("model_dir"),
        checkpoint_path=tiny_cfg.get("checkpoint_path"),
        device=args.device,
        strict=bool(tiny_cfg.get("strict", True)),
        require_checkpoint=bool(tiny_cfg.get("require_checkpoint", True)),
    )
    tinyppg_input = tinyppg_input.to(args.device)
    with torch.no_grad():
        raw_output = loaded.model(tinyppg_input)

    raw_summary = summarize_output(raw_output)
    adapted_results = inspect_adapted_modes(raw_output, target_length=windows.shape[-1])
    results = {
        "config": str(resolve_project_path(args.config, prefer_existing=True)),
        "input_csv": str(csv_path),
        "output_json": str(output_path),
        "device": args.device,
        "model_path": str(loaded.model_path),
        "checkpoint_path": str(loaded.checkpoint_path) if loaded.checkpoint_path else None,
        "num_windows": int(windows.shape[0]),
        "window_length": int(windows.shape[-1]),
        "window_metadata": metadata,
        "raw_output": raw_summary,
        "adapted_modes": adapted_results,
    }
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print_report(results)
    return 0


def load_windows(csv_path: Path, config: dict[str, Any], num_windows: int) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Prepared CSV not found: {csv_path}")
    data_cfg = config.get("data", {})
    dataset = PPGDaLiAWindowDataset(
        data_dir=csv_path.parent,
        sampling_rate_hz=float(data_cfg.get("sampling_rate_hz", 64.0)),
        window_sec=float(data_cfg.get("window_sec", 30.0)),
        step_sec=data_cfg.get("step_sec"),
        preprocess=config.get("preprocessing", {}),
        max_windows=max(1, int(num_windows)),
        subjects=[csv_path.stem],
    )
    samples = [dataset[idx] for idx in range(min(len(dataset), max(1, int(num_windows))))]
    windows = torch.stack([sample["ppg"].float() for sample in samples], dim=0)
    metadata = [dict(sample.get("metadata", {}), hr_label=float(sample["hr_label"])) for sample in samples]
    return windows, metadata


def normalize_batch(x: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    valid = valid_mask.float()
    denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (x * valid).sum(dim=1, keepdim=True) / denom
    centered = (x - mean) * valid
    var = (centered.square() * valid).sum(dim=1, keepdim=True) / denom
    return centered / torch.sqrt(var + eps)


def summarize_output(output: Any) -> dict[str, Any]:
    tensors = collect_tensors(output)
    return {
        "type": type(output).__name__,
        "keys": list(output.keys()) if isinstance(output, dict) else None,
        "tensors": {name: tensor_summary(tensor) for name, tensor in tensors.items()},
    }


def collect_tensors(value: Any, prefix: str = "output") -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    if isinstance(value, torch.Tensor):
        tensors[prefix] = value.detach().cpu()
    elif isinstance(value, dict):
        for key, item in value.items():
            tensors.update(collect_tensors(item, f"{prefix}.{key}"))
    elif isinstance(value, (tuple, list)):
        for idx, item in enumerate(value):
            tensors.update(collect_tensors(item, f"{prefix}.{idx}"))
    return tensors


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return {"shape": list(tensor.shape), "min": None, "max": None, "mean": None}
    return {
        "shape": list(tensor.shape),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
    }


def inspect_adapted_modes(raw_output: Any, target_length: int) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for mode in MODES:
        try:
            adapted = adapt_tinyppg_output(
                raw_output,
                target_length=target_length,
                artifact_output_mode=mode,
                artifact_class_index=1,
            ).detach().cpu()
        except Exception as exc:
            results[mode] = {"error": f"{type(exc).__name__}: {exc}"}
            continue

        threshold_rows = []
        for threshold in THRESHOLDS:
            removed_by_window = (adapted >= threshold).float().mean(dim=1) * 100.0
            threshold_rows.append(
                {
                    "threshold": float(threshold),
                    "percent_removed": float(removed_by_window.mean()),
                    "percent_removed_by_window": [float(value) for value in removed_by_window],
                }
            )
        results[mode] = {
            "shape": list(adapted.shape),
            "stats": tensor_summary(adapted),
            "first_20_values": [float(value) for value in adapted[0, :20]],
            "thresholds": threshold_rows,
        }
    return results


def print_report(results: dict[str, Any]) -> None:
    print(f"Saved diagnostic JSON: {results['output_json']}")
    print(f"Input CSV: {results['input_csv']}")
    print(f"Windows inspected: {results['num_windows']} x {results['window_length']}")
    raw = results["raw_output"]
    print(f"Raw TinyPPG output type: {raw['type']}")
    if raw.get("keys") is not None:
        print(f"Raw TinyPPG output keys: {raw['keys']}")
    print("Raw output tensors:")
    for name, stats in raw["tensors"].items():
        print(
            f"  {name}: shape={stats['shape']} "
            f"min={format_number(stats['min'])} max={format_number(stats['max'])} mean={format_number(stats['mean'])}"
        )

    for mode, mode_result in results["adapted_modes"].items():
        print(f"\nMode: {mode}")
        if "error" in mode_result:
            print(f"  error: {mode_result['error']}")
            continue
        print(f"  adapted shape: {mode_result['shape']}")
        print(f"  first 20 values: {format_values(mode_result['first_20_values'])}")
        print("  percent removed:")
        for row in mode_result["thresholds"]:
            print(f"    threshold {row['threshold']:.1f}: {row['percent_removed']:.2f}%")


def format_number(value: float | None) -> str:
    return "None" if value is None else f"{value:.6f}"


def format_values(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


if __name__ == "__main__":
    raise SystemExit(main())
