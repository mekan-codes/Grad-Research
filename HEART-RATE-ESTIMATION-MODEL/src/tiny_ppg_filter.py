from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import percent_removed
from .utils import interpolate_nans, zscore


TINY_PPG_WARNING = "Tiny-PPG unavailable. Running raw PPG baseline only."


@dataclass
class TinyPPGResult:
    available: bool
    cleaned_ppg: np.ndarray | None
    artifact_mask: np.ndarray | None
    artifact_probability: np.ndarray | None
    percent_signal_removed: float | None
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "percent_signal_removed": self.percent_signal_removed,
            "details": self.details,
        }


def apply_tiny_ppg_filter(
    ppg: np.ndarray,
    fs: float,
    tiny_ppg_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    threshold: float = 0.5,
) -> TinyPPGResult:
    try:
        import torch
        import torch.nn.functional as torch_functional
    except Exception as exc:
        return _unavailable(f"PyTorch import failed: {exc}")

    model_path = _find_model_path(tiny_ppg_dir)
    if model_path is None:
        return _unavailable("Tiny-PPG model.py was not found")

    checkpoint = _find_checkpoint(checkpoint_path, model_path.parent)
    if checkpoint is None:
        return _unavailable("Tiny-PPG checkpoint was not found")

    try:
        module = _load_model_module(model_path)
        model = module.Model()
        state = torch.load(str(checkpoint), map_location="cpu")
        state_dict = _extract_state_dict(state)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    except Exception as exc:
        return _unavailable(f"Tiny-PPG model setup failed: {exc}")

    y = np.asarray(ppg, dtype=float)
    if y.size == 0:
        return _unavailable("empty PPG signal")

    try:
        filled = interpolate_nans(y, min_finite_fraction=0.25)
    except ValueError as exc:
        return _unavailable(f"PPG signal cannot be prepared: {exc}")

    probabilities = np.zeros(y.size, dtype=float)
    counts = np.zeros(y.size, dtype=float)
    segments = _segment_ranges(y.size, fs)

    try:
        with torch.no_grad():
            for start, stop in segments:
                segment = zscore(filled[start:stop])
                tensor = torch.from_numpy(segment.astype(np.float32)[None, None, :])
                output = model(tensor)
                if isinstance(output, dict):
                    output = output.get("seg")
                if output is None:
                    raise RuntimeError("Tiny-PPG model did not return a segmentation output")

                if output.shape[-1] != segment.size:
                    output = torch_functional.interpolate(
                        output,
                        size=segment.size,
                        mode="linear",
                        align_corners=False,
                    )

                probs = output.detach().cpu().numpy().reshape(-1)
                probabilities[start:stop] += probs[: stop - start]
                counts[start:stop] += 1.0
    except Exception as exc:
        return _unavailable(f"Tiny-PPG inference failed: {exc}")

    valid_counts = counts > 0
    probabilities[valid_counts] = probabilities[valid_counts] / counts[valid_counts]
    probabilities[~valid_counts] = 0.0

    artifact_mask = probabilities >= threshold
    cleaned = y.astype(float, copy=True)
    cleaned[artifact_mask] = np.nan

    return TinyPPGResult(
        available=True,
        cleaned_ppg=cleaned,
        artifact_mask=artifact_mask,
        artifact_probability=probabilities,
        percent_signal_removed=percent_removed(artifact_mask),
        details={
            "model_path": str(model_path),
            "checkpoint_path": str(checkpoint),
            "threshold": threshold,
            "segments": len(segments),
        },
    )


def _unavailable(reason: str) -> TinyPPGResult:
    return TinyPPGResult(
        available=False,
        cleaned_ppg=None,
        artifact_mask=None,
        artifact_probability=None,
        percent_signal_removed=None,
        reason=reason,
    )


def _find_model_path(tiny_ppg_dir: str | Path | None) -> Path | None:
    candidates: list[Path] = []
    if tiny_ppg_dir is not None:
        candidates.append(Path(tiny_ppg_dir) / "model.py")

    candidates.extend(
        [
            Path("checkpoints") / "tiny_ppg" / "model.py",
            Path("checkpoints") / "Tiny-PPG" / "model.py",
            Path("Tiny-PPG") / "model.py",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_checkpoint(checkpoint_path: str | Path | None, model_dir: Path) -> Path | None:
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        return path if path.exists() else None

    search_roots = [model_dir, Path("checkpoints") / "tiny_ppg", Path("checkpoints")]
    patterns = ("*.pt", "*.pth", "*.ckpt", "*.bin", "*.pkl")
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[0]
    return None


def _load_model_module(model_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("tiny_ppg_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {model_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_state_dict(state: Any) -> dict[str, Any]:
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            nested = state.get(key)
            if isinstance(nested, dict):
                state = nested
                break

    if not isinstance(state, dict):
        raise RuntimeError("checkpoint does not contain a PyTorch state dict")

    return {
        key.replace("module.", "", 1): value
        for key, value in state.items()
        if hasattr(value, "shape")
    }


def _segment_ranges(n_samples: int, fs: float) -> list[tuple[int, int]]:
    if n_samples <= 0:
        return []

    window = int(round(max(256.0, min(4096.0, fs * 30.0))))
    window = min(window, n_samples)
    step = max(1, window // 2)

    if n_samples <= window:
        return [(0, n_samples)]

    starts = list(range(0, n_samples - window + 1, step))
    last_start = n_samples - window
    if starts[-1] != last_start:
        starts.append(last_start)
    return [(start, start + window) for start in starts]
