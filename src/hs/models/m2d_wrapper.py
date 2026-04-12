from __future__ import annotations

import torch
import torch.nn as nn


class M2DWrapper(nn.Module):
    def __init__(self, out_dim: int = 768) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.encoder = None  # TODO: attach actual M2D encoder

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: (B, T)
        returns: (B, D)
        """
        batch = audio.size(0)
        device = audio.device

        # TODO: replace with actual M2D forward
        return torch.zeros(batch, self.out_dim, device=device)
