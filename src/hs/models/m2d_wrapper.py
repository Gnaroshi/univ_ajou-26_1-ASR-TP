from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


class M2DWrapper(nn.Module):
    """
    Wrapper around M2D PortableM2D.

    Input:
        audio: (B, T) where T = sample_rate * max_sec
    Output:
        feats: (B, D) pooled clip-level features
    """

    def __init__(
        self,
        weight_path: str,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()

        root = Path(__file__).resolve().parents[3]
        m2d_root = root / "external" / "m2d"

        if not m2d_root.exists():
            raise FileNotFoundError(f"M2D submodule not found: {m2d_root}")

        if str(m2d_root) not in sys.path:
            sys.path.insert(0, str(m2d_root))

        from examples.portable_m2d import PortableM2D  # noqa: WPS433

        self.model = PortableM2D(weight_path)
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for p in self.model.parameters():
                p.requires_grad = False

        self.model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 10 * 16000)
            frame_feats = self.model(dummy)  # (1, frames, D)
            self.out_dim = int(frame_feats.shape[-1])

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            self.model.eval()
            with torch.no_grad():
                frame_feats = self.model(audio)  # (B, frames, D)
        else:
            frame_feats = self.model(audio)

        clip_feats = frame_feats.mean(dim=1)  # (B, D)
        return clip_feats
