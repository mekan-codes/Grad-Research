from __future__ import annotations

import torch
from torch import nn

from src.artifact.artifact_detector import ArtifactDetectionResult
from src.artifact.cropper import ArtifactCropper, CropperConfig
from src.models.full_framework import FullPPGHRFramework
from src.models.hr_estimator import HREstimator


class DummyArtifactDetector(nn.Module):
    def forward(self, ppg: torch.Tensor, valid_mask: torch.Tensor | None = None) -> ArtifactDetectionResult:
        if ppg.ndim == 1:
            ppg = ppg.unsqueeze(0)
        mask = torch.zeros_like(ppg, dtype=torch.bool)
        mask[:, 10:20] = True
        probability = mask.float()
        if valid_mask is not None:
            mask = mask & valid_mask.bool()
            probability = probability.masked_fill(~valid_mask.bool(), 0.0)
        return ArtifactDetectionResult(mask, probability, {"source": "dummy"})


def test_full_framework_forward_pass_runs_on_dummy_data() -> None:
    framework = FullPPGHRFramework(
        artifact_detector=DummyArtifactDetector(),
        cropper=ArtifactCropper(CropperConfig(mode="crop", min_clean_samples=1)),
        hr_model=HREstimator(hidden_channels=32, dropout=0.0),
    )
    ppg = torch.randn(2, 128)
    valid_mask = torch.ones(2, 128, dtype=torch.bool)

    output = framework(ppg, valid_mask)

    assert output["predicted_hr"].shape == (2,)
    assert output["artifact_mask"].shape == (2, 128)
    assert output["cropped_signal"].shape[0] == 2
    assert output["cropping_metadata"][0]["cropped_length"] == 118

