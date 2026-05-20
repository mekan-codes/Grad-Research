from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.artifact.artifact_detector import ArtifactDetectionResult, TinyPPGArtifactDetector
from src.artifact.cropper import ArtifactCropper, CropperConfig
from src.artifact.tinyppg_wrapper import assert_no_trainable_tinyppg_parameters, load_tinyppg
from src.data.collate import pad_ppg_sequences
from src.models.hr_estimator import HREstimator
from src.models.robust_hr_estimator import RobustHREstimator


class FullPPGHRFramework(nn.Module):
    """Sequential TinyPPG artifact cropper plus trainable HR estimator."""

    def __init__(
        self,
        artifact_detector: nn.Module,
        cropper: ArtifactCropper,
        hr_model: nn.Module,
    ) -> None:
        super().__init__()
        self.artifact_detector = artifact_detector
        self.cropper = cropper
        self.hr_model = hr_model
        self.freeze_tinyppg()

    def freeze_tinyppg(self) -> None:
        tinyppg = getattr(self.artifact_detector, "tinyppg", self.artifact_detector)
        for param in tinyppg.parameters():
            param.requires_grad = False
        tinyppg.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.freeze_tinyppg()
        return self

    def assert_tinyppg_frozen(self) -> None:
        tinyppg = getattr(self.artifact_detector, "tinyppg", self.artifact_detector)
        assert_no_trainable_tinyppg_parameters(tinyppg)

    def forward(self, padded_ppg: torch.Tensor, valid_mask: torch.Tensor | None = None) -> dict[str, Any]:
        device = padded_ppg.device
        if padded_ppg.ndim == 1:
            padded_ppg = padded_ppg.unsqueeze(0)
        elif padded_ppg.ndim == 3 and padded_ppg.shape[1] == 1:
            padded_ppg = padded_ppg[:, 0, :]
        elif padded_ppg.ndim != 2:
            raise ValueError(f"Expected padded_ppg shape [B,T], [B,1,T], or [T]; got {tuple(padded_ppg.shape)}")
        if valid_mask is None:
            valid_mask = torch.ones_like(padded_ppg, dtype=torch.bool)
        elif valid_mask.ndim == 1:
            valid_mask = valid_mask.unsqueeze(0)
        elif valid_mask.ndim == 3 and valid_mask.shape[1] == 1:
            valid_mask = valid_mask[:, 0, :]
        detection = self.artifact_detector(padded_ppg, valid_mask)
        if not isinstance(detection, ArtifactDetectionResult):
            detection = _coerce_detection_result(detection)

        cropped_sequences: list[torch.Tensor] = []
        metadata: list[dict[str, Any]] = []
        batch_size = padded_ppg.shape[0]
        ppg_2d = padded_ppg
        valid_2d = valid_mask

        for row in range(batch_size):
            length = int(valid_2d[row].sum().item())
            values = ppg_2d[row, :length]
            artifact_mask = detection.artifact_mask[row, :length]
            cropped = self.cropper(values, artifact_mask)
            cropped_tensor = torch.as_tensor(cropped.signal, dtype=torch.float32, device=device)
            cropped_sequences.append(cropped_tensor)
            metadata.append(cropped.metadata.to_dict())

        cropped_padded, cropped_valid_mask = pad_ppg_sequences(cropped_sequences)
        cropped_padded = cropped_padded.to(device)
        cropped_valid_mask = cropped_valid_mask.to(device)
        predicted_hr = self.hr_model(cropped_padded, cropped_valid_mask)
        return {
            "predicted_hr": predicted_hr,
            "artifact_mask": detection.artifact_mask,
            "artifact_probability": detection.artifact_probability,
            "cropped_signal": cropped_padded,
            "cropped_valid_mask": cropped_valid_mask,
            "cropping_metadata": metadata,
        }


FullFramework = FullPPGHRFramework


def build_hr_model(config: dict[str, Any]) -> nn.Module:
    name = str(config.get("name", "hr_estimator"))
    hidden = int(config.get("hidden_channels", 64))
    dropout = float(config.get("dropout", 0.1))
    if name == "robust_hr_estimator":
        return RobustHREstimator(hidden_channels=hidden, dropout=dropout)
    if name == "hr_estimator":
        return HREstimator(hidden_channels=hidden, dropout=dropout)
    raise ValueError(f"Unknown HR model name: {name}")


def build_full_framework(config: dict[str, Any], device: str | torch.device = "cpu") -> FullPPGHRFramework:
    tiny_cfg = config.get("tinyppg", {})
    loaded = load_tinyppg(
        model_dir=tiny_cfg.get("model_dir"),
        checkpoint_path=tiny_cfg.get("checkpoint_path"),
        device=device,
        strict=bool(tiny_cfg.get("strict", True)),
        require_checkpoint=bool(tiny_cfg.get("require_checkpoint", True)),
    )
    detector = TinyPPGArtifactDetector(
        loaded.model,
        threshold=float(tiny_cfg.get("threshold", 0.5)),
        artifact_output_mode=str(tiny_cfg.get("artifact_output_mode", "artifact_probability")),
        artifact_class_index=int(tiny_cfg.get("artifact_class_index", 1)),
    )
    crop_cfg = config.get("cropper", {})
    cropper = ArtifactCropper(
        CropperConfig(
            mode=str(crop_cfg.get("mode", "crop")),
            min_clean_samples=int(crop_cfg.get("min_clean_samples", 1)),
            mask_fill_value=float(crop_cfg.get("mask_fill_value", 0.0)),
            min_output_samples=int(crop_cfg.get("min_output_samples", 1)),
            empty_policy=str(crop_cfg.get("empty_policy", "empty")),
        )
    )
    hr_model = build_hr_model(config.get("model", {})).to(device)
    framework = FullPPGHRFramework(detector, cropper, hr_model).to(device)
    framework.assert_tinyppg_frozen()
    return framework


def _coerce_detection_result(value: Any) -> ArtifactDetectionResult:
    if isinstance(value, dict):
        mask = value.get("artifact_mask")
        probability = value.get("artifact_probability")
        if mask is None or probability is None:
            raise ValueError("Detection dict must contain artifact_mask and artifact_probability")
        return ArtifactDetectionResult(mask, probability, value.get("metadata", {}))
    raise TypeError(f"Unsupported artifact detector output: {type(value)!r}")
