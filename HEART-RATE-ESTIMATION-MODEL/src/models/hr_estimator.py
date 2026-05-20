from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        groups = 8 if out_channels % 8 == 0 else 1
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HREstimator(nn.Module):
    """Small 1D CNN heart-rate regressor.

    Args:
        padded_ppg: ``[batch, time]`` or ``[batch, 1, time]`` float tensor.
        valid_mask: optional ``[batch, time]`` bool tensor where True marks real
            samples and False marks padding.

    Returns:
        ``[batch]`` predicted heart rate in bpm.
    """

    def __init__(self, hidden_channels: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        h = int(hidden_channels)
        self.features = nn.Sequential(
            ConvBlock(1, h // 2, kernel_size=9, dropout=dropout),
            ConvBlock(h // 2, h, kernel_size=7, dropout=dropout),
            ConvBlock(h, h, kernel_size=5, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h // 2, 1),
        )

    def forward(self, padded_ppg: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = _as_batch_channel_time(padded_ppg)
        features = self.features(x)
        mask = _as_feature_mask(valid_mask, features)
        pooled = masked_global_average_pool1d(features, mask)
        return self.head(pooled).squeeze(-1)


def masked_global_average_pool1d(features: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    """Pool ``[B,C,T]`` features while ignoring padded timesteps."""

    if valid_mask is None:
        return features.mean(dim=-1)
    mask = valid_mask.to(device=features.device, dtype=features.dtype)
    if mask.ndim == 2:
        mask = mask.unsqueeze(1)
    if mask.shape[-1] != features.shape[-1]:
        mask = torch.nn.functional.interpolate(mask, size=features.shape[-1], mode="nearest")
    denom = mask.sum(dim=-1).clamp_min(1.0)
    return (features * mask).sum(dim=-1) / denom


def _as_batch_channel_time(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(1).float()
    if x.ndim == 3 and x.shape[1] == 1:
        return x.float()
    if x.ndim == 1:
        return x.reshape(1, 1, -1).float()
    raise ValueError(f"Expected PPG shape [B,T], [B,1,T], or [T]; got {tuple(x.shape)}")


def _as_feature_mask(valid_mask: torch.Tensor | None, features: torch.Tensor) -> torch.Tensor | None:
    if valid_mask is None:
        return None
    mask = valid_mask.to(device=features.device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.ndim != 2:
        raise ValueError(f"valid_mask must have shape [B,T]; got {tuple(mask.shape)}")
    return mask

