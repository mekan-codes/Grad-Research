from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.collate import collate_variable_length_ppg
from src.models.full_framework import build_full_framework, build_hr_model
from src.training.metrics import finalize_metric_sums, merge_metric_sums, regression_metrics
from src.training.train_hr import build_loaders, build_train_val_datasets, _select_device
from src.utils.checkpoint import load_checkpoint


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    config: dict[str, Any] | None = None,
    dataset: Dataset | None = None,
) -> dict[str, float]:
    """Evaluate a checkpoint.

    By default this evaluates the validation split produced by the config.
    Pass ``dataset`` when the caller needs a true held-out test set, such as a
    LOSO fold.
    """

    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    run_config = config or payload.get("config")
    if not isinstance(run_config, dict):
        raise ValueError("Evaluation needs a config, either passed in or stored in the checkpoint")

    device = _select_device(str(run_config.get("device", "cpu")))
    if dataset is None:
        _, val_dataset = build_train_val_datasets(run_config)
        loader = build_loaders(run_config, val_dataset, val_dataset)["val"]
    else:
        batch_size = int(run_config.get("evaluation", {}).get("batch_size", run_config.get("training", {}).get("batch_size", 16)))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_variable_length_ppg,
        )
    model_type = payload.get("model_type", "hr_estimator")

    if model_type == "full_framework":
        model = build_full_framework(run_config, device=device)
        model.hr_model.load_state_dict(payload["hr_model_state_dict"])
    elif model_type == "hr_estimator":
        model = build_hr_model(payload.get("model_config", run_config.get("model", {}))).to(device)
        model.load_state_dict(payload["model_state_dict"])
    else:
        raise ValueError(f"Unknown checkpoint model_type: {model_type}")

    model.eval()
    sums: dict[str, float] = {}
    with torch.no_grad():
        for batch in loader:
            ppg = batch["padded_ppg"].to(device)
            mask = batch["valid_mask"].to(device)
            target = batch["hr_label"].to(device)
            output = model(ppg, mask)
            predicted = output["predicted_hr"] if isinstance(output, dict) else output
            merge_metric_sums(sums, regression_metrics(predicted, target), int(target.numel()))
    return finalize_metric_sums(sums)
