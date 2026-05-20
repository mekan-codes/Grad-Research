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
        artifact_class_index: int = 1,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        self.tinyppg = freeze_module(tinyppg)
        self.threshold = float(threshold)
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
                "adapter": "TinyPPG output dict['seg'] or tensor probability",
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
    artifact_class_index: int = 1,
) -> torch.Tensor:
    """Convert TinyPPG output to ``[batch, time]`` artifact probabilities.

    The local TinyPPG model returns ``{"seg": sigmoid_prob}``. If a future
    checkpoint returns two channels, we use ``artifact_class_index`` after
    softmax. TODO: confirm class index semantics for any non-local checkpoint.
    """

    tensor = _extract_segmentation_tensor(output)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[1] == 1:
            tensor = tensor[:, 0, :]
        elif tensor.shape[1] == 2:
            probs = torch.softmax(tensor, dim=1)
            class_index = max(0, min(int(artifact_class_index), tensor.shape[1] - 1))
            tensor = probs[:, class_index, :]
        elif tensor.shape[-1] == 1:
            tensor = tensor[..., 0]
        elif tensor.shape[-1] == 2:
            probs = torch.softmax(tensor, dim=-1)
            class_index = max(0, min(int(artifact_class_index), tensor.shape[-1] - 1))
            tensor = probs[..., class_index]
        else:
            raise TinyPPGOutputError(f"Unsupported TinyPPG tensor shape: {tuple(tensor.shape)}")
    elif tensor.ndim != 2:
        raise TinyPPGOutputError(f"Unsupported TinyPPG tensor shape: {tuple(tensor.shape)}")

    if tensor.shape[-1] != target_length:
        tensor = F.interpolate(
            tensor.unsqueeze(1),
            size=target_length,
            mode="linear",
            align_corners=False,
        ).squeeze(1)

    finite = tensor.detach()[torch.isfinite(tensor.detach())]
    if finite.numel() and (finite.min() < 0.0 or finite.max() > 1.0):
        tensor = torch.sigmoid(tensor)
    return torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


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
