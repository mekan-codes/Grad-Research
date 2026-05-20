from __future__ import annotations

import numpy as np

from src.data.windowing import seconds_to_samples, sliding_window_ranges


def test_supported_window_durations_convert_to_samples() -> None:
    assert seconds_to_samples(30, 64) == 1920
    assert seconds_to_samples(25, 64) == 1600
    assert seconds_to_samples(20, 64) == 1280


def test_sliding_window_ranges() -> None:
    ranges = sliding_window_ranges(3200, sampling_rate_hz=64, window_sec=20, step_sec=10)
    assert ranges[0] == (0, 1280)
    assert ranges[1] == (640, 1920)
    assert ranges[-1] == (1920, 3200)


def test_partial_window_can_be_included() -> None:
    values = np.arange(100)
    ranges = sliding_window_ranges(values.size, sampling_rate_hz=64, window_sec=30, include_partial=True)
    assert ranges == [(0, 100)]

