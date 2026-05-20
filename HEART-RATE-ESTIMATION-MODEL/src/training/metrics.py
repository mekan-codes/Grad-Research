from __future__ import annotations

import math

import torch


def regression_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = predicted.detach().float().cpu()
    y = target.detach().float().cpu()
    errors = pred - y
    mae = torch.mean(torch.abs(errors)).item()
    rmse = math.sqrt(torch.mean(errors.square()).item())
    return {"mae": float(mae), "rmse": float(rmse)}


def merge_metric_sums(sums: dict[str, float], batch_metrics: dict[str, float], batch_size: int) -> None:
    for key, value in batch_metrics.items():
        sums[key] = sums.get(key, 0.0) + float(value) * batch_size
    sums["n"] = sums.get("n", 0.0) + batch_size


def finalize_metric_sums(sums: dict[str, float]) -> dict[str, float]:
    n = max(1.0, sums.get("n", 0.0))
    return {key: value / n for key, value in sums.items() if key != "n"}

