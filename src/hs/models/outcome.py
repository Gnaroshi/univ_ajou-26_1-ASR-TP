from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn as nn

from hs.models.classifier import ClassificationHead
from hs.models.m2d_wrapper import M2DWrapper
from hs.models.peft import PEFTConfig


class OutcomeModel(nn.Module):
    def __init__(
        self,
        m2d_weight_path: str,
        num_classes: int = 2,
        head_hidden_dim: int = 512,
        peft_method: str = "none",
        trainable_last_blocks: int = 1,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_target_keywords: str = "qkv,proj",
        keep_frozen_modules_eval: bool = True,
    ) -> None:
        super().__init__()

        self.model_config = {
            "m2d_weight_path": m2d_weight_path,
            "num_classes": num_classes,
            "head_hidden_dim": head_hidden_dim,
            "peft_method": peft_method,
            "trainable_last_blocks": trainable_last_blocks,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target_keywords": lora_target_keywords,
            "keep_frozen_modules_eval": keep_frozen_modules_eval,
        }

        peft_config = PEFTConfig(
            method=peft_method,  # type: ignore[arg-type]
            trainable_last_blocks=trainable_last_blocks,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_keywords=lora_target_keywords,
        )

        self.encoder = M2DWrapper(
            weight_path=m2d_weight_path,
            peft_config=peft_config,
            keep_frozen_modules_eval=keep_frozen_modules_eval,
        )

        self.head = ClassificationHead(
            in_dim=self.encoder.out_dim,
            hidden_dim=head_hidden_dim,
            num_classes=num_classes,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(audio)
        logits = self.head(feats)
        return logits

    def get_config(self) -> dict:
        return dict(self.model_config)