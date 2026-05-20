from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact.artifact_detector import TinyPPGArtifactDetector
from src.artifact.cropper import ArtifactCropper, CropperConfig
from src.artifact.tinyppg_wrapper import load_tinyppg
from src.data.preprocessing import config_from_mapping, preprocess_ppg_window
from src.data_loader import load_ppg_csv
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TinyPPG artifact detection and cropping demo.")
    parser.add_argument("--input", default="data/input/sample.csv", help="Input PPG CSV.")
    parser.add_argument("--config", default="configs/debug_cpu.yaml", help="Config with TinyPPG paths.")
    parser.add_argument("--output-dir", default="data/output", help="Directory for plot output.")
    parser.add_argument("--fs", type=float, default=64.0, help="Fallback sampling rate.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
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
    detector = TinyPPGArtifactDetector(
        loaded.model,
        threshold=float(tiny_cfg.get("threshold", 0.5)),
        artifact_class_index=int(tiny_cfg.get("artifact_class_index", 1)),
    )
    tensor = torch.tensor(clean_input, dtype=torch.float32).unsqueeze(0)
    detection = detector(tensor)

    crop_cfg = config.get("cropper", {})
    cropper = ArtifactCropper(
        CropperConfig(
            mode=str(crop_cfg.get("mode", "crop")),
            min_clean_samples=int(crop_cfg.get("min_clean_samples", 1)),
            mask_fill_value=float(crop_cfg.get("mask_fill_value", 0.0)),
        )
    )
    mask = detection.artifact_mask.squeeze(0).cpu().numpy()
    cropped = cropper(clean_input, mask)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "artifact_crop_demo.png"
    _plot_demo(clean_input, mask, cropped.signal, fs, source, output_path)
    print(f"Saved plot: {output_path}")
    meta = cropped.metadata
    print(
        f"Cropped {meta.percent_removed:.2f}% "
        f"across {meta.number_of_removed_segments} artifact segment(s)"
    )
    return 0


def _load_or_synthetic(input_path: str, fallback_fs: float, config: dict) -> tuple[np.ndarray, float, str]:
    path = Path(input_path)
    window_sec = float(config.get("data", {}).get("window_sec", 30.0))
    if path.exists():
        loaded = load_ppg_csv(path, fallback_fs=fallback_fs)
        n = min(loaded.ppg.size, int(round(loaded.fs * window_sec)))
        return loaded.ppg[:n], float(loaded.fs), str(path)

    fs = float(fallback_fs)
    n = int(round(fs * window_sec))
    t = np.arange(n, dtype=float) / fs
    ppg = np.sin(2.0 * np.pi * 1.25 * t) + 0.25 * np.sin(2.0 * np.pi * 2.5 * t)
    start = n // 3
    stop = min(n, start + n // 6)
    rng = np.random.default_rng(11)
    ppg[start:stop] += rng.normal(0.0, 2.0, size=stop - start)
    print(f"Input not found: {path}. Using a synthetic demo window.")
    return ppg.astype(float), fs, "synthetic demo"


def _plot_demo(
    raw: np.ndarray,
    artifact_mask: np.ndarray,
    cropped: np.ndarray,
    fs: float,
    source: str,
    output_path: Path,
) -> None:
    time = np.arange(raw.size) / fs
    cropped_time = np.arange(cropped.size) / fs
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    axes[0].plot(time, raw, color="#1f77b4", linewidth=1.0, label="raw PPG")
    for start, stop in _mask_segments(artifact_mask):
        axes[0].axvspan(start / fs, stop / fs, color="#d62728", alpha=0.25)
    axes[0].set_title(f"Raw PPG with TinyPPG artifact regions ({source})")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("normalized PPG")
    axes[1].plot(cropped_time, cropped, color="#2ca02c", linewidth=1.0)
    axes[1].set_title("Cropped clean PPG signal")
    axes[1].set_xlabel("cropped time (s)")
    axes[1].set_ylabel("normalized PPG")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _mask_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = None
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

