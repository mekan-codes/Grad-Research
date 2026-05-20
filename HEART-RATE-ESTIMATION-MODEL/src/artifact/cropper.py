from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import torch


@dataclass
class CropMetadata:
    original_length: int
    cropped_length: int
    percent_removed: float
    number_of_removed_segments: int
    mode: str
    all_clean: bool = False
    all_noisy: bool = False
    kept_short_clean_segment: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CroppedSignal:
    signal: np.ndarray
    metadata: CropMetadata


@dataclass
class CropperConfig:
    mode: str = "crop"
    min_clean_samples: int = 1
    mask_fill_value: float = 0.0


class ArtifactCropper:
    """Remove noisy PPG samples using an artifact mask.

    ``artifact_mask=True`` means a sample is noisy. The default ``crop`` mode
    removes those samples and concatenates the remaining clean regions.
    """

    def __init__(self, config: CropperConfig | None = None) -> None:
        self.config = config or CropperConfig()

    def __call__(self, ppg: np.ndarray | torch.Tensor, artifact_mask: np.ndarray | torch.Tensor) -> CroppedSignal:
        return crop_artifact_regions(ppg, artifact_mask, self.config)


def crop_artifact_regions(
    ppg: np.ndarray | torch.Tensor,
    artifact_mask: np.ndarray | torch.Tensor,
    config: CropperConfig | None = None,
) -> CroppedSignal:
    cfg = config or CropperConfig()
    values = _to_numpy(ppg).astype(np.float32, copy=False).reshape(-1)
    mask = _to_numpy(artifact_mask).astype(bool, copy=False).reshape(-1)
    if values.shape[0] != mask.shape[0]:
        raise ValueError(f"PPG length {values.shape[0]} and artifact mask length {mask.shape[0]} do not match")

    original_length = int(values.size)
    removed_segments = _count_true_segments(mask)
    if original_length == 0:
        metadata = CropMetadata(0, 0, 0.0, 0, cfg.mode)
        return CroppedSignal(values.copy(), metadata)

    percent = float(mask.mean() * 100.0)
    if cfg.mode == "mask":
        output = values.copy()
        output[mask] = float(cfg.mask_fill_value)
        metadata = CropMetadata(
            original_length=original_length,
            cropped_length=int(output.size),
            percent_removed=percent,
            number_of_removed_segments=removed_segments,
            mode="mask",
            all_clean=not mask.any(),
            all_noisy=bool(mask.all()),
        )
        return CroppedSignal(output, metadata)
    if cfg.mode != "crop":
        raise ValueError(f"Unknown cropper mode: {cfg.mode}")

    if not mask.any():
        metadata = CropMetadata(
            original_length=original_length,
            cropped_length=original_length,
            percent_removed=0.0,
            number_of_removed_segments=0,
            mode="crop",
            all_clean=True,
        )
        return CroppedSignal(values.copy(), metadata)

    if mask.all():
        metadata = CropMetadata(
            original_length=original_length,
            cropped_length=0,
            percent_removed=100.0,
            number_of_removed_segments=1,
            mode="crop",
            all_noisy=True,
        )
        return CroppedSignal(np.empty(0, dtype=np.float32), metadata)

    clean_segments = _segments(mask, value=False)
    min_len = max(1, int(cfg.min_clean_samples))
    kept = [(start, stop) for start, stop in clean_segments if stop - start >= min_len]
    kept_short = False
    if not kept and clean_segments:
        kept = [max(clean_segments, key=lambda item: item[1] - item[0])]
        kept_short = True

    if kept:
        cropped = np.concatenate([values[start:stop] for start, stop in kept]).astype(np.float32, copy=False)
    else:
        cropped = np.empty(0, dtype=np.float32)

    metadata = CropMetadata(
        original_length=original_length,
        cropped_length=int(cropped.size),
        percent_removed=float(round(100.0 * (1.0 - (cropped.size / original_length)), 6)),
        number_of_removed_segments=removed_segments,
        mode="crop",
        kept_short_clean_segment=kept_short,
    )
    return CroppedSignal(cropped, metadata)


def _to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _count_true_segments(mask: np.ndarray) -> int:
    return len(_segments(mask, value=True))


def _segments(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return []
    target = arr == value
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_target in enumerate(target):
        if is_target and start is None:
            start = idx
        elif not is_target and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, arr.size))
    return segments
