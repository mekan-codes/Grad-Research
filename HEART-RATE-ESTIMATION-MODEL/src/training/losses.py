from __future__ import annotations

import torch
import torch.nn.functional as F


def mae_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(predicted - target))


def huber_loss(predicted: torch.Tensor, target: torch.Tensor, delta: float = 5.0) -> torch.Tensor:
    return F.huber_loss(predicted, target, delta=float(delta))


def get_loss_fn(name: str, huber_delta: float = 5.0):
    key = name.lower()
    if key == "mae":
        return mae_loss
    if key in {"huber", "smooth_l1"}:
        return lambda predicted, target: huber_loss(predicted, target, delta=huber_delta)
    raise ValueError(f"Unknown loss function: {name}")

