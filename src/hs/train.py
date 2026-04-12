from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hs.models.classifier import SimpleClassifier
from hs.models.m2d_wrapper import M2DWrapper


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class OutcomeModel(nn.Module):
    def __init__(self, feat_dim: int = 768, num_classes: int = 2) -> None:
        super().__init__()
        self.encoder = M2DWrapper(out_dim=feat_dim)
        self.classifier = SimpleClassifier(in_dim=feat_dim, num_classes=num_classes)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(audio)
        logits = self.classifier(feats)
        return logits


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0

    for batch in loader:
        audio = batch["audio"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(audio)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * audio.size(0)
        pred = logits.argmax(dim=-1)
        correct += (pred == label).sum().item()
        total += audio.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }
