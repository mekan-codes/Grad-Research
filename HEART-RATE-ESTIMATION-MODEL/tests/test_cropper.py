from __future__ import annotations

import numpy as np

from src.artifact.cropper import CropperConfig, crop_artifact_regions


def test_cropper_removes_noisy_regions_by_default() -> None:
    ppg = np.arange(10, dtype=np.float32)
    mask = np.array([False, False, True, True, False, False, False, True, False, False])

    result = crop_artifact_regions(ppg, mask, CropperConfig(mode="crop", min_clean_samples=1))

    assert result.signal.tolist() == [0, 1, 4, 5, 6, 8, 9]
    assert result.metadata.original_length == 10
    assert result.metadata.cropped_length == 7
    assert result.metadata.number_of_removed_segments == 2
    assert result.metadata.percent_removed == 30.0


def test_cropper_handles_all_clean_and_all_noisy() -> None:
    ppg = np.arange(5, dtype=np.float32)

    clean = crop_artifact_regions(ppg, np.zeros(5, dtype=bool))
    assert clean.signal.tolist() == ppg.tolist()
    assert clean.metadata.all_clean is True

    noisy = crop_artifact_regions(ppg, np.ones(5, dtype=bool))
    assert noisy.signal.size == 0
    assert noisy.metadata.all_noisy is True
    assert noisy.metadata.percent_removed == 100.0


def test_cropper_keeps_longest_short_segment_if_needed() -> None:
    ppg = np.arange(8, dtype=np.float32)
    mask = np.array([True, False, True, True, False, False, True, True])

    result = crop_artifact_regions(ppg, mask, CropperConfig(mode="crop", min_clean_samples=5))

    assert result.signal.tolist() == [4, 5]
    assert result.metadata.kept_short_clean_segment is True

