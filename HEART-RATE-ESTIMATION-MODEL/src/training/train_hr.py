from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.data.collate import collate_variable_length_ppg
from src.data.ppg_dalia_dataset import PPGDaLiAWindowDataset, SyntheticPPGWindowDataset
from src.models.full_framework import build_hr_model
from src.training.losses import get_loss_fn
from src.training.metrics import finalize_metric_sums, merge_metric_sums, regression_metrics
from src.utils.checkpoint import save_checkpoint
from src.utils.seed import set_seed


def train_hr_estimator(config: dict[str, Any]) -> dict[str, Any]:
    """Train only the HR estimator on raw/prepared clean windows."""

    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = _select_device(str(config.get("device", "cpu")))
    train_dataset, val_dataset = build_train_val_datasets(config)
    loaders = build_loaders(config, train_dataset, val_dataset)

    model = build_hr_model(config.get("model", {})).to(device)
    training_cfg = config.get("training", {})
    loss_fn = get_loss_fn(
        str(training_cfg.get("loss", "huber")),
        huber_delta=float(training_cfg.get("huber_delta", 5.0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )

    best_mae = float("inf")
    best_payload: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    epochs = int(training_cfg.get("epochs", 1))
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model,
            loaders["train"],
            device,
            loss_fn,
            optimizer=optimizer,
            max_batches=training_cfg.get("max_train_batches"),
        )
        val_metrics = _run_epoch(
            model,
            loaders["val"],
            device,
            loss_fn,
            optimizer=None,
            max_batches=training_cfg.get("max_val_batches"),
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"epoch {epoch:03d} | train loss {train_metrics['loss']:.4f} "
            f"mae {train_metrics['mae']:.2f} | val mae {val_metrics['mae']:.2f}"
        )
        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_payload = {
                "model_type": "hr_estimator",
                "model_state_dict": model.state_dict(),
                "model_config": config.get("model", {}),
                "config": config,
                "epoch": epoch,
                "val_metrics": val_metrics,
            }

    checkpoint_dir = Path(training_cfg.get("hr_checkpoint_dir", training_cfg.get("checkpoint_dir", "checkpoints/hr_estimator")))
    checkpoint_path = checkpoint_dir / "best_model.pth"
    if best_payload is not None:
        save_checkpoint(best_payload, checkpoint_path)
    return {"history": history, "best_mae": best_mae, "checkpoint_path": str(checkpoint_path)}


def build_train_val_datasets(config: dict[str, Any]) -> tuple[Dataset, Dataset]:
    data_cfg = config.get("data", {})
    dataset_name = str(data_cfg.get("dataset", "synthetic"))
    fs = float(data_cfg.get("sampling_rate_hz", 64.0))
    window_sec = float(data_cfg.get("window_sec", 30.0))
    seed = int(config.get("seed", 42))

    if dataset_name == "synthetic":
        dataset: Dataset = SyntheticPPGWindowDataset(
            num_samples=int(data_cfg.get("synthetic_samples", 32)),
            sampling_rate_hz=fs,
            window_sec=window_sec,
            seed=seed,
        )
    elif dataset_name in {"ppg_dalia", "ppg_fieldstudy"}:
        dataset = PPGDaLiAWindowDataset(
            data_dir=data_cfg.get("data_dir", "data/input/prepared"),
            sampling_rate_hz=fs,
            window_sec=window_sec,
            step_sec=data_cfg.get("step_sec"),
            preprocess=config.get("preprocessing", {}),
            max_windows=data_cfg.get("max_windows"),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return split_dataset(dataset, float(data_cfg.get("val_fraction", 0.2)), seed)


def split_dataset(dataset: Dataset, val_fraction: float, seed: int) -> tuple[Dataset, Dataset]:
    n = len(dataset)
    if n < 2:
        return dataset, dataset
    val_size = max(1, int(round(n * val_fraction)))
    train_size = max(1, n - val_size)
    if train_size + val_size > n:
        val_size = n - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_dataset, val_dataset


def build_loaders(config: dict[str, Any], train_dataset: Dataset, val_dataset: Dataset) -> dict[str, DataLoader]:
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 16))
    num_workers = int(data_cfg.get("num_workers", 0))
    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_variable_length_ppg,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_variable_length_ppg,
        ),
    }


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> dict[str, float]:
    if optimizer is None:
        model.eval()
    else:
        model.train()

    sums: dict[str, float] = {}
    total_loss = 0.0
    total_items = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        ppg = batch["padded_ppg"].to(device)
        mask = batch["valid_mask"].to(device)
        target = batch["hr_label"].to(device)
        with torch.set_grad_enabled(optimizer is not None):
            predicted = model(ppg, mask)
            loss = loss_fn(predicted, target)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        batch_size = int(target.numel())
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
        merge_metric_sums(sums, regression_metrics(predicted, target), batch_size)

    metrics = finalize_metric_sums(sums)
    metrics["loss"] = total_loss / max(1, total_items)
    return metrics


def _select_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")

