from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.data.collate import collate_variable_length_ppg
from src.data.ppg_dalia_dataset import PPGDaLiAWindowDataset, SyntheticPPGWindowDataset
from src.models.full_framework import build_hr_model
from src.training.losses import get_loss_fn
from src.training.metrics import finalize_metric_sums, merge_metric_sums, regression_metrics
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.environment import resolve_device
from src.utils.seed import set_seed


def train_hr_estimator(config: dict[str, Any], resume: str | bool | None = None, logger: Any = None) -> dict[str, Any]:
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
    amp_enabled = bool(training_cfg.get("use_amp", config.get("use_amp", False))) and device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)

    best_mae = float("inf")
    best_payload: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    checkpoint_dir = Path(training_cfg.get("hr_checkpoint_dir", training_cfg.get("checkpoint_dir", "checkpoints/hr_estimator")))
    best_path = checkpoint_dir / "best_model.pth"
    last_path = checkpoint_dir / "last_model.pth"
    start_epoch = 1
    resume_path = _resolve_resume_path(resume if resume is not None else training_cfg.get("resume", config.get("resume")), checkpoint_dir)
    if resume_path is not None:
        payload = load_checkpoint(resume_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "scaler_state_dict" in payload and amp_enabled:
            scaler.load_state_dict(payload["scaler_state_dict"])
        best_mae = float(payload.get("best_mae", payload.get("val_metrics", {}).get("mae", best_mae)))
        history = list(payload.get("history", []))
        start_epoch = int(payload.get("epoch", 0)) + 1
        _log(logger, f"Resumed baseline HR estimator from {resume_path} at epoch {start_epoch}")

    epochs = int(training_cfg.get("max_epochs", training_cfg.get("epochs", 1)))
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = _run_epoch(
            model,
            loaders["train"],
            device,
            loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            max_batches=training_cfg.get("max_train_batches"),
        )
        val_metrics = _run_epoch(
            model,
            loaders["val"],
            device,
            loss_fn,
            optimizer=None,
            scaler=None,
            amp_enabled=False,
            max_batches=training_cfg.get("max_val_batches"),
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        _log(
            logger,
            f"epoch {epoch:03d} | train loss {train_metrics['loss']:.4f} "
            f"mae {train_metrics['mae']:.2f} | val mae {val_metrics['mae']:.2f}",
        )
        last_payload = {
            "model_type": "hr_estimator",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if amp_enabled else None,
            "model_config": config.get("model", {}),
            "config": config,
            "epoch": epoch,
            "history": history,
            "best_mae": best_mae,
            "val_metrics": val_metrics,
        }
        save_checkpoint(last_payload, last_path)
        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_payload = {
                "model_type": "hr_estimator",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if amp_enabled else None,
                "model_config": config.get("model", {}),
                "config": config,
                "epoch": epoch,
                "history": history,
                "best_mae": best_mae,
                "val_metrics": val_metrics,
            }
            save_checkpoint(best_payload, best_path)

    if best_payload is None and best_path.exists():
        best_payload = load_checkpoint(best_path, map_location="cpu")
    return {"history": history, "best_mae": best_mae, "checkpoint_path": str(best_path), "last_checkpoint_path": str(last_path)}


def build_train_val_datasets(config: dict[str, Any]) -> tuple[Dataset, Dataset]:
    data_cfg = config.get("data", {})
    dataset_name = str(data_cfg.get("dataset", "synthetic")).lower()
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
        prepared_dir = Path(data_cfg.get("prepared_dir", data_cfg.get("data_dir", "data/input/prepared")))
        train_dir = prepared_dir / "train"
        val_dir = prepared_dir / "val"
        if train_dir.exists() and val_dir.exists():
            train_dataset = PPGDaLiAWindowDataset(
                data_dir=train_dir,
                sampling_rate_hz=fs,
                window_sec=window_sec,
                step_sec=data_cfg.get("step_sec"),
                preprocess=config.get("preprocessing", {}),
                max_windows=data_cfg.get("max_windows"),
            )
            val_dataset = PPGDaLiAWindowDataset(
                data_dir=val_dir,
                sampling_rate_hz=fs,
                window_sec=window_sec,
                step_sec=data_cfg.get("step_sec"),
                preprocess=config.get("preprocessing", {}),
                max_windows=data_cfg.get("max_val_windows"),
            )
            return train_dataset, val_dataset
        dataset = PPGDaLiAWindowDataset(
            data_dir=prepared_dir,
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
    num_workers = int(training_cfg.get("num_workers", data_cfg.get("num_workers", 0)))
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
    scaler: torch.cuda.amp.GradScaler | None,
    amp_enabled: bool,
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
            with autocast_context(amp_enabled):
                predicted = model(ppg, mask)
                loss = loss_fn(predicted, target)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
    return resolve_device(requested)


def _resolve_resume_path(resume: str | bool | None, checkpoint_dir: Path) -> Path | None:
    if resume in {None, False, "", "false", "False"}:
        return None
    if resume is True or str(resume).lower() in {"true", "latest", "auto"}:
        latest = checkpoint_dir / "last_model.pth"
        return latest if latest.exists() else None
    path = Path(str(resume))
    return path if path.exists() else None


def _log(logger: Any, message: str) -> None:
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def make_grad_scaler(amp_enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def autocast_context(amp_enabled: bool):
    if not amp_enabled:
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=True)
