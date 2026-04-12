from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from hs.data.circor import CirCorOutcomeDataset
from hs.data.preprocess import collate_circor_batch
from hs.models.classifier import ClassificationHead
from hs.models.m2d_wrapper import M2DWrapper
from hs.utils import ensure_dir, get_device, set_seed


class OutcomeModel(nn.Module):
    def __init__(self, m2d_weight_path: str, num_classes: int = 2) -> None:
        super().__init__()
        self.encoder = M2DWrapper(
            weight_path=m2d_weight_path,
            freeze_encoder=True,
        )
        self.head = ClassificationHead(
            in_dim=self.encoder.out_dim,
            hidden_dim=512,
            num_classes=num_classes,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(audio)
        logits = self.head(feats)
        return logits


def build_loader(
    manifest_path: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    max_sec: float,
) -> DataLoader:
    ds = CirCorOutcomeDataset(
        manifest_path=manifest_path,
        sample_rate=16000,
        max_sec=max_sec,
        crop_mode="random" if shuffle else "center",
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_circor_batch,
    )


def compute_metrics(labels: torch.Tensor, probs: torch.Tensor) -> dict:
    labels_np = labels.cpu().numpy()
    probs_np = probs.cpu().numpy()
    preds_np = (probs_np >= 0.5).astype(int)

    metrics = {
        "acc": float(accuracy_score(labels_np, preds_np)),
        "f1": float(f1_score(labels_np, preds_np)),
    }

    try:
        metrics["auroc"] = float(roc_auc_score(labels_np, probs_np))
    except ValueError:
        metrics["auroc"] = float("nan")

    return metrics


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_count = 0

    all_probs = []
    all_labels = []

    for batch in loader:
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(audio)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        probs = torch.softmax(logits, dim=-1)[:, 1].detach()

        total_loss += loss.item() * audio.size(0)
        total_count += audio.size(0)

        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_count = 0

    all_probs = []
    all_labels = []

    for batch in loader:
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)

        logits = model(audio)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=-1)[:, 1]

        total_loss += loss.item() * audio.size(0)
        total_count += audio.size(0)

        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics


def save_checkpoint(model, optimizer, epoch, best_valid_auroc, path):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_valid_auroc": best_valid_auroc,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--valid_manifest", type=str, required=True)
    parser.add_argument("--m2d_weight_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/circor_outcome_m2d")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_sec", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = get_device()
    print(f"[INFO] device = {device}")

    train_loader = build_loader(
        manifest_path=args.train_manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        max_sec=args.max_sec,
    )
    valid_loader = build_loader(
        manifest_path=args.valid_manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        max_sec=args.max_sec,
    )

    model = OutcomeModel(
        m2d_weight_path=args.m2d_weight_path,
        num_classes=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr)

    best_valid_auroc = float("-inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        valid_metrics = evaluate(model, valid_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_f1": train_metrics["f1"],
            "train_auroc": train_metrics["auroc"],
            "valid_loss": valid_metrics["loss"],
            "valid_acc": valid_metrics["acc"],
            "valid_f1": valid_metrics["f1"],
            "valid_auroc": valid_metrics["auroc"],
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_acc']:.4f} "
            f"train_f1={row['train_f1']:.4f} "
            f"train_auroc={row['train_auroc']:.4f} | "
            f"valid_loss={row['valid_loss']:.4f} "
            f"valid_acc={row['valid_acc']:.4f} "
            f"valid_f1={row['valid_f1']:.4f} "
            f"valid_auroc={row['valid_auroc']:.4f}"
        )

        with open(Path(args.output_dir) / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_valid_auroc=best_valid_auroc,
            path=Path(args.output_dir) / "last.pt",
        )

        if valid_metrics["auroc"] > best_valid_auroc:
            best_valid_auroc = valid_metrics["auroc"]
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_valid_auroc=best_valid_auroc,
                path=Path(args.output_dir) / "best.pt",
            )
            print(f"[INFO] best checkpoint updated: valid_auroc={best_valid_auroc:.4f}")


if __name__ == "__main__":
    main()
