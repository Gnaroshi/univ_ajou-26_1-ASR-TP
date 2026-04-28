from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from hs.models.peft import PEFTConfig, configure_peft


class M2DWrapper(nn.Module):
    """
    Wrapper around M2D PortableM2D.

    Input:
        audio: (B, T), usually T = 16000 * 10
    Output:
        feats: (B, D), clip-level pooled features
    """

    def __init__(
        self,
        weight_path: str,
        peft_config: PEFTConfig | None = None,
        keep_frozen_modules_eval: bool = True,
    ) -> None:
        super().__init__()

        root = Path(__file__).resolve().parents[3]
        m2d_root = root / "external" / "m2d"

        if not m2d_root.exists():
            raise FileNotFoundError(f"M2D submodule not found: {m2d_root}")

        if str(m2d_root) not in sys.path:
            sys.path.insert(0, str(m2d_root))

        from examples.portable_m2d import PortableM2D  # noqa: WPS433

        self.weight_path = weight_path
        self.peft_config = peft_config or PEFTConfig(method="none")
        self.keep_frozen_modules_eval = keep_frozen_modules_eval

        self.model = PortableM2D(weight_path)

        self.peft_report = configure_peft(self.model, self.peft_config)
        self.has_trainable_encoder = any(
            p.requires_grad for p in self.model.parameters()
        )

        self.model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 10 * 16000)
            frame_feats = self.model(dummy)
            self.out_dim = int(frame_feats.shape[-1])

    def _set_trainable_encoder_modules_train(self, mode: bool) -> None:
        """
        Keep frozen backbone modules in eval mode, while trainable modules are
        placed in train mode.

        This avoids accidentally enabling dropout / batchnorm behavior in frozen
        parts of the encoder. For last-block tuning, selected block prefixes are
        put in train mode so dropout and other parameter-free child modules behave
        like a normal fine-tuned transformer block. For LoRA, LoRALinear itself is
        put into train mode so its dropout works without enabling the whole block.
        """
        self.model.eval()

        train_prefixes = self.peft_report.get("train_module_prefixes", [])
        for module_name in train_prefixes:
            try:
                self.model.get_submodule(module_name).train(mode)
            except AttributeError:
                continue

        for module in self.model.modules():
            has_direct_trainable_param = any(
                p.requires_grad for p in module.parameters(recurse=False)
            )
            is_lora_layer = bool(getattr(module, "is_lora_layer", False))

            if has_direct_trainable_param or is_lora_layer:
                module.train(mode)

    def train(self, mode: bool = True):
        super().train(mode)

        if not self.has_trainable_encoder:
            self.model.eval()
            return self

        if self.keep_frozen_modules_eval:
            self._set_trainable_encoder_modules_train(mode)
        else:
            self.model.train(mode)

        return self

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if not self.has_trainable_encoder:
            self.model.eval()
            with torch.no_grad():
                frame_feats = self.model(audio)
        else:
            frame_feats = self.model(audio)

        clip_feats = frame_feats.mean(dim=1)
        return clip_feats
