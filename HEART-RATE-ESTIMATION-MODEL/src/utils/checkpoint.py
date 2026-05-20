from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return output_path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must contain a dict payload: {checkpoint_path}")
    return payload


def best_checkpoint_path(checkpoint_dir: str | Path, filename: str = "best_model.pth") -> Path:
    return Path(checkpoint_dir) / filename

