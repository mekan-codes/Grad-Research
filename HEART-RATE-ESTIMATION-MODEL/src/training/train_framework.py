from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.models.full_framework import build_full_framework
from src.training.losses import get_loss_fn
from src.training.metrics import finalize_metric_sums, merge_metric_sums, regression_metrics
from src.training.train_hr import build_loaders, build_train_val_datasets, _select_device
from src.utils.checkpoint import save_checkpoint
from src.utils.seed import set_seed


def train_full_framework(config: dict[str, Any]) -> dict[str, Any]:
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

    best_mae = float("inf")
    best_payload: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(training_cfg.get("epochs", 1)) + 1):
        train_metrics = _run_framework_epoch(
            framework,
            loaders["train"],
            device,
            loss_fn,
            optimizer=optimizer,
            max_batches=training_cfg.get("max_train_batches"),
        )
        val_metrics = _run_framework_epoch(
            framework,
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
                "model_type": "full_framework",
                "hr_model_state_dict": framework.hr_model.state_dict(),
                "model_config": config.get("model", {}),
                "config": config,
                "epoch": epoch,
                "val_metrics": val_metrics,
            }

    checkpoint_dir = Path(training_cfg.get("framework_checkpoint_dir", "checkpoints/full_framework"))
    checkpoint_path = checkpoint_dir / "best_framework.pth"
    if best_payload is not None:
        save_checkpoint(best_payload, checkpoint_path)
    framework.assert_tinyppg_frozen()
    return {"history": history, "best_mae": best_mae, "checkpoint_path": str(checkpoint_path)}


def _run_framework_epoch(
    framework: torch.nn.Module,
    loader,
    device: torch.device,
    loss_fn,
    optimizer: torch.optim.Optimizer | None,
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
            output = framework(ppg, mask)
            predicted = output["predicted_hr"]
            loss = loss_fn(predicted, target)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
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

