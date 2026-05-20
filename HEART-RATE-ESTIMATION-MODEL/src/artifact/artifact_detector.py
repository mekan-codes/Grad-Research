from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.artifact.tinyppg_wrapper import freeze_module


class TinyPPGOutputError(RuntimeError):
    """Raised when TinyPPG output cannot be converted into an artifact mask."""


@dataclass
class ArtifactDetectionResult:
    artifact_mask: torch.Tensor
    artifact_probability: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


class TinyPPGArtifactDetector(nn.Module):
    """Frozen TinyPPG adapter that returns a noisy/artifact mask.

    Mask convention: ``True`` means noisy/artifact and should be cropped.
    """

    def __init__(
        self,
        tinyppg: nn.Module,
        threshold: float = 0.5,
        artifact_output_mode: str = "artifact_probability",
        artifact_class_index: int = 1,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        self.tinyppg = freeze_module(tinyppg)
        self.threshold = float(threshold)
        self.artifact_output_mode = str(artifact_output_mode)
        self.artifact_class_index = int(artifact_class_index)
        self.normalize_input = bool(normalize_input)

    def forward(
        self,
        ppg: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> ArtifactDetectionResult:
        x = _as_batch_time(ppg).float()
        mask = _as_valid_mask(valid_mask, x)
        if self.normalize_input:
            x = _normalize_batch(x, mask)
        model_input = x.unsqueeze(1)

        with torch.no_grad():
            output = self.tinyppg(model_input)
            probability = adapt_tinyppg_output(
                output,
                target_length=x.shape[-1],
                artifact_output_mode=self.artifact_output_mode,
                artifact_class_index=self.artifact_class_index,
            )

        probability = probability.to(device=x.device, dtype=torch.float32)
        probability = probability.masked_fill(~mask, 0.0)
        artifact_mask = probability >= self.threshold
        artifact_mask = artifact_mask & mask
        return ArtifactDetectionResult(
            artifact_mask=artifact_mask,
            artifact_probability=probability,
            metadata={
                "threshold": self.threshold,
                "mask_true_means": "artifact/noisy",
                "artifact_output_mode": self.artifact_output_mode,
                "adapter": "TinyPPG segmentation tensor adapted according to config",
            },
        )

    def detect_numpy(self, ppg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor = torch.as_tensor(ppg, dtype=torch.float32)
        result = self.forward(tensor)
        return (
            result.artifact_mask.squeeze(0).cpu().numpy().astype(bool),
            result.artifact_probability.squeeze(0).cpu().numpy(),
        )


def adapt_tinyppg_output(
    output: Any,
    target_length: int,
    artifact_output_mode: str = "artifact_probability",
    artifact_class_index: int = 1,
) -> torch.Tensor:
    """Convert TinyPPG output to ``[batch, time]`` artifact probabilities.

    The caller must choose the output interpretation with
    ``artifact_output_mode``:

    - ``artifact_probability``: tensor is already P(artifact).
    - ``clean_probability``: tensor is P(clean), so it is inverted.
    - ``logits``: one-channel logits use sigmoid; two-channel logits use softmax.
    - ``class_index``: select ``artifact_class_index`` from a class dimension.

    TODO: confirm TinyPPG class/output semantics for any checkpoint other than
    the local ``output["seg"]`` checkpoint before trusting real training metrics.
    """

    tensor = _extract_segmentation_tensor(output)
    tensor, already_probability = _normalize_shape(tensor, artifact_output_mode, artifact_class_index)

    if tensor.shape[-1] != target_length:
        tensor = F.interpolate(
            tensor.unsqueeze(1),
            size=target_length,
            mode="linear",
            align_corners=False,
        ).squeeze(1)

    probability = _apply_output_mode(tensor, artifact_output_mode, already_probability=already_probability)
    return torch.nan_to_num(probability, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _normalize_shape(tensor: torch.Tensor, mode: str, artifact_class_index: int) -> tuple[torch.Tensor, bool]:
    if tensor.ndim == 1:
        return tensor.unsqueeze(0), False
    if tensor.ndim == 2:
        return tensor, False
    if tensor.ndim == 3:
        if tensor.shape[1] == 1:
            return tensor[:, 0, :], False
        if tensor.shape[1] >= 2:
            if mode not in {"logits", "class_index"}:
                raise TinyPPGOutputError(
                    "TinyPPG returned multiple channel outputs. Set "
                    "artifact_output_mode to 'logits' or 'class_index'."
            )
            class_index = max(0, min(int(artifact_class_index), tensor.shape[1] - 1))
            if mode == "logits":
                return torch.softmax(tensor, dim=1)[:, class_index, :], True
            return tensor[:, class_index, :], False
        if tensor.shape[-1] == 1:
            return tensor[..., 0], False
        if tensor.shape[-1] >= 2:
            if mode not in {"logits", "class_index"}:
                raise TinyPPGOutputError(
                    "TinyPPG returned multiple class outputs. Set "
                    "artifact_output_mode to 'logits' or 'class_index'."
            )
            class_index = max(0, min(int(artifact_class_index), tensor.shape[-1] - 1))
            if mode == "logits":
                return torch.softmax(tensor, dim=-1)[..., class_index], True
            return tensor[..., class_index], False
    raise TinyPPGOutputError(f"Unsupported TinyPPG tensor shape: {tuple(tensor.shape)}")


def _apply_output_mode(tensor: torch.Tensor, mode: str, already_probability: bool = False) -> torch.Tensor:
    mode = str(mode)
    if mode == "artifact_probability":
        _raise_if_not_probability(tensor, mode)
        return tensor
    if mode == "clean_probability":
        _raise_if_not_probability(tensor, mode)
        return 1.0 - tensor
    if mode == "logits":
        if already_probability:
            return tensor
        return torch.sigmoid(tensor)
    if mode == "class_index":
        _raise_if_not_probability(tensor, mode)
        return tensor
    raise TinyPPGOutputError(
        "artifact_output_mode must be one of: artifact_probability, "
        "clean_probability, logits, class_index"
    )


def _raise_if_not_probability(tensor: torch.Tensor, mode: str) -> None:
    detached = tensor.detach()
    finite = detached[torch.isfinite(detached)]
    if finite.numel() and (finite.min() < -1e-4 or finite.max() > 1.0001):
        raise TinyPPGOutputError(
            f"artifact_output_mode={mode!r} expects probabilities in [0, 1], "
            f"but saw min={float(finite.min()):.4f}, max={float(finite.max()):.4f}. "
            "Try artifact_output_mode='logits' if this checkpoint emits logits."
        )


def _extract_segmentation_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        if "seg" in output and isinstance(output["seg"], torch.Tensor):
            return output["seg"]
        for key in ("artifact", "mask", "probability", "logits"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
        tensor_keys = [key for key, value in output.items() if isinstance(value, torch.Tensor)]
        raise TinyPPGOutputError(
            "TinyPPG dict output did not contain a known segmentation key. "
            f"Tensor keys were: {tensor_keys}"
        )
    if isinstance(output, (tuple, list)):
        tensors = [item for item in output if isinstance(item, torch.Tensor)]
        if tensors:
            return tensors[0]
    raise TinyPPGOutputError(f"Cannot adapt TinyPPG output type: {type(output)!r}")


def extract_segmentation_tensor(output: Any) -> torch.Tensor:
    """Public diagnostic helper for inspecting raw TinyPPG segmentation output."""

    return _extract_segmentation_tensor(output)


def _as_batch_time(ppg: torch.Tensor) -> torch.Tensor:
    if ppg.ndim == 1:
        return ppg.unsqueeze(0)
    if ppg.ndim == 2:
        return ppg
    if ppg.ndim == 3 and ppg.shape[1] == 1:
        return ppg[:, 0, :]
    raise ValueError(f"Expected PPG shape [time], [batch,time], or [batch,1,time]; got {tuple(ppg.shape)}")


def _as_valid_mask(valid_mask: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones_like(x, dtype=torch.bool)
    mask = valid_mask.to(device=x.device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape != x.shape:
        raise ValueError(f"valid_mask shape {tuple(mask.shape)} does not match PPG shape {tuple(x.shape)}")
    return mask


def _normalize_batch(x: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    valid = valid_mask.float()
    denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (x * valid).sum(dim=1, keepdim=True) / denom
    centered = (x - mean) * valid
    var = (centered.square() * valid).sum(dim=1, keepdim=True) / denom
    return centered / torch.sqrt(var + eps)
