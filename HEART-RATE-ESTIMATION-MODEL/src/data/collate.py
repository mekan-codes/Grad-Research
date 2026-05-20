from __future__ import annotations

from typing import Any

import torch


def pad_ppg_sequences(
    sequences: list[torch.Tensor],
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length PPG tensors.

    Returns:
        padded_ppg: Float tensor with shape ``[batch, max_time]``.
        valid_mask: Bool tensor with shape ``[batch, max_time]`` where True is
            a real sample and False is padding.
    """

    if not sequences:
        raise ValueError("Cannot collate an empty batch")

    flattened = [seq.detach().float().reshape(-1) for seq in sequences]
    max_len = max(1, max(seq.numel() for seq in flattened))
    batch = torch.full((len(flattened), max_len), float(pad_value), dtype=torch.float32)
    mask = torch.zeros((len(flattened), max_len), dtype=torch.bool)
    for row, seq in enumerate(flattened):
        length = seq.numel()
        if length == 0:
            continue
        batch[row, :length] = seq
        mask[row, :length] = True
    return batch, mask


def collate_variable_length_ppg(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate dict samples from cropped or raw PPG datasets."""

    ppg_values = [sample["ppg"] for sample in batch]
    tensors = [value if isinstance(value, torch.Tensor) else torch.as_tensor(value) for value in ppg_values]
    padded, valid_mask = pad_ppg_sequences(tensors)

    labels = []
    for sample in batch:
        label = sample.get("hr_label", sample.get("hr"))
        if label is None:
            raise KeyError("Each sample must contain 'hr_label' or 'hr'")
        labels.append(float(label))

    metadata = [sample.get("metadata", {}) for sample in batch]
    lengths = valid_mask.sum(dim=1).tolist()
    for item, length in zip(metadata, lengths):
        if isinstance(item, dict):
            item.setdefault("collated_length", int(length))

    return {
        "padded_ppg": padded,
        "valid_mask": valid_mask,
        "hr_label": torch.tensor(labels, dtype=torch.float32),
        "metadata": metadata,
    }

