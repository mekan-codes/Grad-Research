from __future__ import annotations

import torch

from src.data.collate import collate_variable_length_ppg


def test_variable_length_collate_pads_and_masks() -> None:
    batch = [
        {"ppg": torch.tensor([1.0, 2.0, 3.0]), "hr_label": 70.0, "metadata": {}},
        {"ppg": torch.tensor([4.0]), "hr_label": 80.0, "metadata": {}},
        {"ppg": torch.tensor([]), "hr_label": 90.0, "metadata": {}},
    ]

    collated = collate_variable_length_ppg(batch)

    assert collated["padded_ppg"].shape == (3, 3)
    assert collated["valid_mask"].tolist() == [
        [True, True, True],
        [True, False, False],
        [False, False, False],
    ]
    assert collated["hr_label"].tolist() == [70.0, 80.0, 90.0]

