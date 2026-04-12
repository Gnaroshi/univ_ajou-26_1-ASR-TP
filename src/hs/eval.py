from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from hs.data.circor import CirCorOutcomeDataset
from hs.data.preprocess import collate_circor_batch
from hs.models.classifier import ClassificationHead
from hs.models.m2d_wrapper import M2DWrapper
from hs.utils import get_device


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


def compute_metrics(labels, probs):
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "classification_report": classification_report(labels, preds, output_dict=True),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["auroc"] = float("nan")
    return metrics


@torch.no_grad()
def evaluate_recording_level(model, loader, device):
    model.eval()

    all_probs = []
    all_labels = []
    all_patient_ids = []

    for batch in loader:
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)

        logits = model(audio)
        probs = torch.softmax(logits, dim=-1)[:, 1]

        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
        all_patient_ids.extend(batch["patient_id"])

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    return {
        "metrics": compute_metrics(labels, probs),
        "probs": probs,
        "labels": labels,
        "patient_ids": all_patient_ids,
    }


def evaluate_patient_level(recording_probs, recording_labels, patient_ids):
    grouped_probs = defaultdict(list)
    grouped_labels = {}

    for prob, label, pid in zip(recording_probs, recording_labels, patient_ids):
        grouped_probs[pid].append(float(prob))
        grouped_labels[pid] = int(label)

    patient_probs = []
    patient_labels = []

    for pid in sorted(grouped_probs.keys()):
        patient_probs.append(sum(grouped_probs[pid]) / len(grouped_probs[pid]))
        patient_labels.append(grouped_labels[pid])

    import numpy as np

    patient_probs = np.array(patient_probs)
    patient_labels = np.array(patient_labels)

    return compute_metrics(patient_labels, patient_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--m2d_weight_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_sec", type=float, default=10.0)
    parser.add_argument(
        "--output_json",
        type=str,
        default="outputs/circor_outcome_m2d/test_metrics.json",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"[INFO] device = {device}")

    ds = CirCorOutcomeDataset(
        manifest_path=args.manifest,
        sample_rate=16000,
        max_sec=args.max_sec,
        crop_mode="center",
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_circor_batch,
    )

    model = OutcomeModel(
        m2d_weight_path=args.m2d_weight_path,
        num_classes=2,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    recording = evaluate_recording_level(model, loader, device)
    patient = evaluate_patient_level(
        recording_probs=recording["probs"],
        recording_labels=recording["labels"],
        patient_ids=recording["patient_ids"],
    )

    result = {
        "recording_level": recording["metrics"],
        "patient_level": patient,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
