from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


SUPPORTED_WINDOW_SECONDS = (20.0, 25.0, 30.0)


@dataclass(frozen=True)
class WindowConfig:
    sampling_rate_hz: float
    window_sec: float = 30.0
    step_sec: float | None = None
    include_partial: bool = False

    @property
    def window_samples(self) -> int:
        return seconds_to_samples(self.window_sec, self.sampling_rate_hz)

    @property
    def step_samples(self) -> int:
        step = self.step_sec if self.step_sec is not None else self.window_sec
        return seconds_to_samples(step, self.sampling_rate_hz)


def seconds_to_samples(seconds: float, sampling_rate_hz: float) -> int:
    if seconds <= 0:
        raise ValueError("Window duration must be positive")
    if sampling_rate_hz <= 0:
        raise ValueError("Sampling rate must be positive")
    return max(1, int(round(float(seconds) * float(sampling_rate_hz))))


def sliding_window_ranges(
    n_samples: int,
    sampling_rate_hz: float,
    window_sec: float,
    step_sec: float | None = None,
    include_partial: bool = False,
) -> list[tuple[int, int]]:
    """Return ``(start, stop)`` sample ranges for a PPG signal."""

    if n_samples <= 0:
        return []
    window = seconds_to_samples(window_sec, sampling_rate_hz)
    step = seconds_to_samples(step_sec if step_sec is not None else window_sec, sampling_rate_hz)

    if n_samples < window:
        return [(0, n_samples)] if include_partial else []

    starts = list(range(0, n_samples - window + 1, step))
    if include_partial and starts:
        last_stop = starts[-1] + window
        if last_stop < n_samples:
            starts.append(max(0, n_samples - window))
    return [(start, min(start + window, n_samples)) for start in starts]


def make_windows(values: np.ndarray, ranges: Iterable[tuple[int, int]]) -> list[np.ndarray]:
    arr = np.asarray(values)
    return [arr[start:stop] for start, stop in ranges]


def median_label_for_range(labels: np.ndarray, start: int, stop: int) -> float | None:
    y = np.asarray(labels, dtype=float)[start:stop]
    finite = np.isfinite(y)
    if not finite.any():
        return None
    return float(np.nanmedian(y[finite]))

