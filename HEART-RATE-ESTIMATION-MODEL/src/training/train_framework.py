from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.models.full_framework import build_full_framework
from src.training.losses import get_loss_fn
from src.training.metrics import finalize_metric_sums, merge_metric_sums, regression_metrics
from src.training.train_hr import (
    autocast_context,
    build_loaders,
    build_train_val_datasets,
    make_grad_scaler,
    _log,
    _select_device,
)
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.seed import set_seed


def train_full_framework(config: dict[str, Any], resume: str | bool | None = None, logger: Any = None) -> dict[str, Any]:
    """Train HR estimator through TinyPPG -> cropper -> HR model.

    TinyPPG stays frozen. The optimizer receives only ``framework.hr_model``
    parameters.
    """

    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = _select_device(str(config.get("device", "cpu")))
    train_dataset, val_dataset = build_train_val_datasets(config)
    loaders = build_loaders(config, train_dataset, val_dataset)

    framework = build_full_framework(config, device=device)
    framework.assert_tinyppg_frozen()
    training_cfg = config.get("training", {})
    loss_fn = get_loss_fn(
        str(training_cfg.get("loss", "huber")),
        huber_delta=float(training_cfg.get("huber_delta", 5.0)),
    )
    optimizer = torch.optim.AdamW(
        [param for param in framework.hr_model.parameters() if param.requires_grad],
        lr=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    amp_enabled = bool(training_cfg.get("use_amp", config.get("use_amp", False))) and device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)

    best_mae = float("inf")
    best_payload: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    checkpoint_dir = Path(training_cfg.get("framework_checkpoint_dir", "checkpoints/full_framework"))
    best_path = checkpoint_dir / "best_framework.pth"
    last_path = checkpoint_dir / "last_framework.pth"
    start_epoch = 1
    resume_path = _resolve_resume_path(resume if resume is not None else training_cfg.get("resume", config.get("resume")), checkpoint_dir)
    if resume_path is not None:
        payload = load_checkpoint(resume_path, map_location=device)
        framework.hr_model.load_state_dict(payload["hr_model_state_dict"])
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "scaler_state_dict" in payload and amp_enabled:
            scaler.load_state_dict(payload["scaler_state_dict"])
        best_mae = float(payload.get("best_mae", payload.get("val_metrics", {}).get("mae", best_mae)))
        history = list(payload.get("history", []))
        start_epoch = int(payload.get("epoch", 0)) + 1
        _log(logger, f"Resumed full framework from {resume_path} at epoch {start_epoch}")

    epochs = int(training_cfg.get("max_epochs", training_cfg.get("epochs", 1)))
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = _run_framework_epoch(
            framework,
            loaders["train"],
            device,
            loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            max_batches=training_cfg.get("max_train_batches"),
        )
        val_metrics = _run_framework_epoch(
            framework,
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
            f"[tinyppg_cropped] epoch {epoch:03d}/{epochs:03d} | train loss {train_metrics['loss']:.4f} "
            f"mae {train_metrics['mae']:.2f} | val mae {val_metrics['mae']:.2f}",
        )
        last_payload = {
            "model_type": "full_framework",
            "hr_model_state_dict": framework.hr_model.state_dict(),
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
                "model_type": "full_framework",
                "hr_model_state_dict": framework.hr_model.state_dict(),
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
    framework.assert_tinyppg_frozen()
    return {"history": history, "best_mae": best_mae, "checkpoint_path": str(best_path), "last_checkpoint_path": str(last_path)}


def _run_framework_epoch(
    framework: torch.nn.Module,
    loader,
    device: torch.device,
    loss_fn,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    amp_enabled: bool,
    max_batches: int | None,
) -> dict[str, float]:
    if optimizer is None:
        framework.eval()
    else:
        framework.train()
        framework.freeze_tinyppg()

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
                output = framework(ppg, mask)
                predicted = output["predicted_hr"]
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
            framework.freeze_tinyppg()
        batch_size = int(target.numel())
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
        merge_metric_sums(sums, regression_metrics(predicted, target), batch_size)

    metrics = finalize_metric_sums(sums)
    metrics["loss"] = total_loss / max(1, total_items)
    return metrics


def _resolve_resume_path(resume: str | bool | None, checkpoint_dir: Path) -> Path | None:
    if resume in {None, False, "", "false", "False"}:
        return None
    if resume is True or str(resume).lower() in {"true", "latest", "auto"}:
        latest = checkpoint_dir / "last_framework.pth"
        return latest if latest.exists() else None
    path = Path(str(resume))
    return path if path.exists() else None
