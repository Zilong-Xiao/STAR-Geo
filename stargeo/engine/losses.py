from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricContrastiveLoss(nn.Module):
    """The symmetric ground-to-aerial / aerial-to-ground objective in the paper."""

    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.label_smoothing = float(label_smoothing)

    def forward(
        self,
        query_descriptors: torch.Tensor,
        reference_descriptors: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        query_descriptors = F.normalize(query_descriptors, dim=-1)
        reference_descriptors = F.normalize(reference_descriptors, dim=-1)
        logits = logit_scale * query_descriptors @ reference_descriptors.t()
        labels = torch.arange(logits.shape[0], device=logits.device)
        query_loss = F.cross_entropy(
            logits,
            labels,
            label_smoothing=self.label_smoothing,
        )
        reference_loss = F.cross_entropy(
            logits.t(),
            labels,
            label_smoothing=self.label_smoothing,
        )
        return 0.5 * (query_loss + reference_loss)
