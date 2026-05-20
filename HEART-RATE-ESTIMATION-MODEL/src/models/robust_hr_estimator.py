from __future__ import annotations

import torch
from torch import nn

from src.models.hr_estimator import ConvBlock, _as_batch_channel_time, _as_feature_mask


class RobustHREstimator(nn.Module):
    """CNN + GRU + masked attention HR estimator.

    This is more robust than plain pooling because the CNN extracts local PPG
    morphology, the GRU models temporal context across beats, attention can
    down-weight weak regions, and the padding mask prevents cropped-away samples
    from influencing the regression. Accelerometer channels can be added later
    before the GRU as extra temporal features.
    """

    def __init__(self, hidden_channels: int = 64, gru_hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        h = int(hidden_channels)
        self.cnn = nn.Sequential(
            ConvBlock(1, h // 2, kernel_size=9, dropout=dropout),
            ConvBlock(h // 2, h, kernel_size=7, dropout=dropout),
        )
        self.gru = nn.GRU(
            input_size=h,
            hidden_size=int(gru_hidden),
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Linear(int(gru_hidden) * 2, 1)
        self.head = nn.Sequential(
            nn.Linear(int(gru_hidden) * 2, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, 1),
        )

    def forward(self, padded_ppg: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = _as_batch_channel_time(padded_ppg)
        features = self.cnn(x).transpose(1, 2)
        mask = _as_feature_mask(valid_mask, features.transpose(1, 2))
        output, _ = self.gru(features)
        context = self._masked_attention_pool(output, mask)
        return self.head(context).squeeze(-1)

    def _masked_attention_pool(self, sequence: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        logits = self.attention(sequence).squeeze(-1)
        if valid_mask is None:
            weights = torch.softmax(logits, dim=-1)
        else:
            mask = valid_mask.to(device=sequence.device, dtype=torch.bool)
            if mask.shape[-1] != logits.shape[-1]:
                mask = torch.nn.functional.interpolate(
                    mask.float().unsqueeze(1),
                    size=logits.shape[-1],
                    mode="nearest",
                ).squeeze(1).bool()
            safe_mask = mask.clone()
            empty_rows = ~safe_mask.any(dim=1)
            if empty_rows.any():
                safe_mask[empty_rows, 0] = True
            logits = logits.masked_fill(~safe_mask, -1e9)
            weights = torch.softmax(logits, dim=-1)
            weights = weights.masked_fill(~safe_mask, 0.0)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)

