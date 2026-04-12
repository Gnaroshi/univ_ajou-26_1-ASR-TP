from __future__ import annotations

import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
