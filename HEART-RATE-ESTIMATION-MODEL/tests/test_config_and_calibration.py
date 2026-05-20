from __future__ import annotations

import torch

from src.artifact.artifact_detector import ArtifactDetectionResult
from src.artifact.calibration import calibrate_artifact_thresholds
from src.artifact.cropper import CropperConfig, crop_artifact_regions
from src.utils.config import load_config


class LinearProbabilityDetector:
    def __call__(self, ppg: torch.Tensor, valid_mask: torch.Tensor | None = None) -> ArtifactDetectionResult:
        batch, length = ppg.shape
        probability = torch.linspace(0.0, 1.0, length).repeat(batch, 1)
        artifact_mask = probability >= 0.5
        return ArtifactDetectionResult(artifact_mask, probability, {})


def test_workspace_and_smoke_configs_load() -> None:
    workspace = load_config("configs/train_workspace.yaml")
    smoke = load_config("configs/smoke_test.yaml")

    assert workspace["data"]["window_seconds_options"] == [30, 25, 20]
    assert workspace["tinyppg"]["artifact_output_mode"] in {
        "artifact_probability",
        "clean_probability",
        "logits",
        "class_index",
    }
    assert smoke["data"]["dataset"] == "synthetic"


def test_threshold_calibration_returns_json_like_output() -> None:
    samples = [{"ppg": torch.zeros(16), "hr_label": 70.0} for _ in range(3)]

    result = calibrate_artifact_thresholds(
        artifact_detector=LinearProbabilityDetector(),
        samples=samples,
        thresholds=[0.25, 0.5, 0.75],
        cropper_config=CropperConfig(mode="crop", min_output_samples=1),
        max_windows=2,
    )

    assert result["windows_evaluated"] == 2
    assert len(result["thresholds"]) == 3
    assert all("mean_percent_removed" in row for row in result["thresholds"])


def test_cropper_empty_fallback_keeps_valid_tensor_when_requested() -> None:
    result = crop_artifact_regions(
        torch.arange(5, dtype=torch.float32),
        torch.ones(5, dtype=torch.bool),
        CropperConfig(mode="crop", empty_policy="keep_original", min_output_samples=1),
    )

    assert result.signal.tolist() == [0, 1, 2, 3, 4]
    assert result.metadata.all_noisy is True

