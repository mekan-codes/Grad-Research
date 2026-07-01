from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact.artifact_detector import (
    adapt_tinyppg_output,
    extract_segmentation_tensor,
)
from src.artifact.calibration import probability_diagnostics, tensor_stats
from src.artifact.cropper import CropperConfig, crop_artifact_regions
from src.artifact.tinyppg_wrapper import load_tinyppg
from src.data.preprocessing import config_from_mapping, preprocess_ppg_window
from src.data_loader import load_ppg_csv
from src.utils.config import apply_cli_overrides, load_config, normalize_config_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect TinyPPG raw/adapted artifact output on one window.")
    parser.add_argument("--config", default="configs/smoke_test.yaml", help="Config path.")
    parser.add_argument("--input", default=None, help="Optional CSV input window.")
    parser.add_argument("--output-dir", default="runs/tinyppg_diagnostic", help="Run output directory.")
    parser.add_argument("--run-name", default=None, help="Optional run subdirectory name.")
    parser.add_argument("--fs", type=float, default=64.0, help="Fallback sampling rate.")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plot generation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.run_name:
        output_dir = output_dir / args.run_name
    config = normalize_config_paths(apply_cli_overrides(load_config(args.config), output_dir=str(output_dir)))
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    ppg, fs, source = _load_or_synthetic(args.input, args.fs, config)
    preprocess_config = config_from_mapping(config.get("preprocessing", {}), fs)
    clean_input = preprocess_ppg_window(ppg, preprocess_config)

    tiny_cfg = config.get("tinyppg", {})
    loaded = load_tinyppg(
        model_dir=tiny_cfg.get("model_dir"),
        checkpoint_path=tiny_cfg.get("checkpoint_path"),
        device="cpu",
        strict=bool(tiny_cfg.get("strict", True)),
        require_checkpoint=bool(tiny_cfg.get("require_checkpoint", True)),
    )
    tensor = torch.tensor(clean_input, dtype=torch.float32).reshape(1, 1, -1)
    with torch.no_grad():
        raw_output = loaded.model(tensor)
    raw_tensor = extract_segmentation_tensor(raw_output)
    probability = adapt_tinyppg_output(
        raw_output,
        target_length=clean_input.size,
        artifact_output_mode=str(tiny_cfg.get("artifact_output_mode", "artifact_probability")),
        artifact_class_index=int(tiny_cfg.get("artifact_class_index", 1)),
    ).squeeze(0)
    threshold = float(tiny_cfg.get("threshold", 0.5))
    artifact_mask = probability.cpu().numpy() >= threshold
    crop_cfg = config.get("cropper", {})
    cropped = crop_artifact_regions(
        clean_input,
        artifact_mask,
        CropperConfig(
            mode=str(crop_cfg.get("mode", "crop")),
            min_clean_samples=int(crop_cfg.get("min_clean_samples", 1)),
            min_output_samples=int(crop_cfg.get("min_output_samples", 1)),
            empty_policy=str(crop_cfg.get("empty_policy", "empty")),
        ),
    )

    stats = {
        "source": source,
        "artifact_output_mode": tiny_cfg.get("artifact_output_mode", "artifact_probability"),
        "artifact_class_index": tiny_cfg.get("artifact_class_index", 1),
        "threshold": threshold,
        "raw_output_stats": tensor_stats(raw_tensor),
        "artifact_probability_stats": probability_diagnostics(probability, threshold),
        "percent_cropped": cropped.metadata.percent_removed,
        "cropping_metadata": cropped.metadata.to_dict(),
        "notes": [
            "If raw output is outside [0, 1], artifact_output_mode may need to be logits.",
            "If percent_above_threshold is near 100%, output may be inverted or threshold may be too low.",
            "If output represents clean probability, set artifact_output_mode: clean_probability.",
        ],
    }
    stats_path = metrics_dir / "tinyppg_output_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    if not args.no_plot:
        plot_path = plots_dir / "tinyppg_artifact_diagnostic.png"
        _plot_diagnostic(clean_input, probability.cpu().numpy(), artifact_mask, fs, plot_path)
        print(f"Saved plot: {plot_path}")
    print(f"Saved stats: {stats_path}")
    print(f"Percent above threshold: {stats['artifact_probability_stats']['percent_above_threshold']:.2f}%")
    print(f"Percent cropped: {stats['percent_cropped']:.2f}%")
    return 0


def _load_or_synthetic(input_path: str | None, fallback_fs: float, config: dict) -> tuple[np.ndarray, float, str]:
    window_sec = float(config.get("data", {}).get("window_sec", 30.0))
    if input_path and Path(input_path).exists():
        loaded = load_ppg_csv(input_path, fallback_fs=fallback_fs)
        n = min(loaded.ppg.size, int(round(loaded.fs * window_sec)))
        return loaded.ppg[:n], float(loaded.fs), str(input_path)
    fs = float(config.get("data", {}).get("sampling_rate_hz", fallback_fs))
    n = int(round(fs * window_sec))
    t = np.arange(n, dtype=float) / fs
    rng = np.random.default_rng(13)
    ppg = np.sin(2.0 * np.pi * 1.2 * t) + 0.08 * rng.normal(size=n)
    return ppg.astype(float), fs, "synthetic"


def _plot_diagnostic(
    ppg: np.ndarray,
    probability: np.ndarray,
    artifact_mask: np.ndarray,
    fs: float,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    time = np.arange(ppg.size) / fs
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    axes[0].plot(time, ppg, linewidth=1.0)
    for start, stop in _segments(artifact_mask):
        axes[0].axvspan(start / fs, stop / fs, color="#d62728", alpha=0.25)
    axes[0].set_title("PPG with thresholded artifact regions")
    axes[1].plot(time, probability, linewidth=1.0, color="#9467bd")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Adapted TinyPPG artifact probability")
    axes[1].set_xlabel("time (s)")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, mask.size))
    return segments


if __name__ == "__main__":
    raise SystemExit(main())
