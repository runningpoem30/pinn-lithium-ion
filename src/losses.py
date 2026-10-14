"""Loss functions for DeepHPM training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models import DeepHPMOutput


@dataclass(frozen=True)
class DeepHPMLossValues:
    total: torch.Tensor
    data: torch.Tensor
    physics: torch.Tensor


class DeepHPMLoss(nn.Module):
    """Mean data loss plus a weighted learned-dynamics residual loss.

    Both terms use normalized quantities so their magnitude is not an artifact
    of the physical PCL and cycle units. Physical predictions and residuals are
    still available in :class:`src.models.DeepHPMOutput` for reporting.
    """

    def __init__(self, physics_weight: float) -> None:
        super().__init__()
        if physics_weight < 0:
            raise ValueError("physics_weight must be non-negative")
        self.physics_weight = float(physics_weight)

    def forward(
        self,
        output: DeepHPMOutput,
        target_pcl_normalized: torch.Tensor,
    ) -> DeepHPMLossValues:
        if target_pcl_normalized.shape != output.pcl_normalized.shape:
            raise ValueError(
                "Target shape must match predicted normalized PCL shape; "
                f"got {target_pcl_normalized.shape} and "
                f"{output.pcl_normalized.shape}"
            )
        data_loss = torch.mean(
            torch.square(output.pcl_normalized - target_pcl_normalized)
        )
        physics_loss = torch.mean(torch.square(output.residual_normalized))
        total_loss = data_loss + self.physics_weight * physics_loss
        return DeepHPMLossValues(
            total=total_loss,
            data=data_loss,
            physics=physics_loss,
        )
