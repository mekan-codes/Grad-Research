from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.artifact.tinyppg_wrapper import (
    assert_no_trainable_tinyppg_parameters,
    has_trainable_parameters,
    load_tinyppg,
)


def test_freeze_helper_detects_no_trainable_parameters() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    assert has_trainable_parameters(model) is False
    assert_no_trainable_tinyppg_parameters(model)


def test_real_tinyppg_loads_frozen_when_checkpoint_is_present() -> None:
    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root.parent / "Tiny-PPG-master"
    checkpoint = model_dir / "Save_Model" / "model_parameter-2023-5-31-1.pkl"
    if not model_dir.exists() or not checkpoint.exists():
        pytest.skip("Local TinyPPG model/checkpoint not present")

    loaded = load_tinyppg(model_dir=model_dir, checkpoint_path=checkpoint, device="cpu")

    assert loaded.model.training is False
    assert has_trainable_parameters(loaded.model) is False
    assert_no_trainable_tinyppg_parameters(loaded.model)
