from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import torch
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
from hs.models.outcome import OutcomeModel
from hs.utils import get_device


def compute_metrics(labels, probs):
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "classification_report": classification_report(
            labels,
            preds,
            output_dict=True,
            zero_division=0,
        ),
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


def aggregate_patient_probs(
    recording_probs,
    recording_labels,
    patient_ids,
    method: str = "mean",
):
    grouped_probs = defaultdict(list)
    grouped_labels = {}

    for prob, label, pid in zip(recording_probs, recording_labels, patient_ids):
        grouped_probs[pid].append(float(prob))
        grouped_labels[pid] = int(label)

    patient_probs = []
    patient_labels = []

    for pid in sorted(grouped_probs.keys()):
        probs = np.array(grouped_probs[pid], dtype=np.float64)

        if method == "mean":
            patient_prob = float(probs.mean())
        elif method == "max":
            patient_prob = float(probs.max())
        elif method == "noisy_or":
            patient_prob = float(1.0 - np.prod(1.0 - probs))
        else:
            raise ValueError(f"Unknown patient aggregation method: {method}")

        patient_probs.append(patient_prob)
        patient_labels.append(grouped_labels[pid])

    return np.array(patient_labels), np.array(patient_probs)


def evaluate_patient_level(
    recording_probs,
    recording_labels,
    patient_ids,
    aggregation: str = "mean",
):
    patient_labels, patient_probs = aggregate_patient_probs(
        recording_probs=recording_probs,
        recording_labels=recording_labels,
        patient_ids=patient_ids,
        method=aggregation,
    )

    return compute_metrics(patient_labels, patient_probs)


def build_model_config_from_args(args) -> dict:
    return {
        "m2d_weight_path": args.m2d_weight_path,
        "num_classes": 2,
        "head_hidden_dim": 512,
        "peft_method": args.peft_method,
        "trainable_last_blocks": args.trainable_last_blocks,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_keywords": args.lora_target_keywords,
        "keep_frozen_modules_eval": args.keep_frozen_modules_eval,
    }


def resolve_model_config(ckpt: dict, args) -> dict:
    """
    If checkpoint has model_config, use it so LoRA / PEFT structure is rebuilt
    correctly before strict load.

    m2d_weight_path is overridden by CLI because another machine may store weights
    in a different path.
    """
    if "model_config" in ckpt:
        config = dict(ckpt["model_config"])
        config["m2d_weight_path"] = args.m2d_weight_path
        return config

    return build_model_config_from_args(args)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--m2d_weight_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_sec", type=float, default=10.0)
    parser.add_argument("--output_json", type=str, default="outputs/circor_outcome_m2d/test_metrics.json")
    parser.add_argument(
        "--patient_aggregation",
        type=str,
        default="mean",
        choices=["mean", "max", "noisy_or"],
    )

    parser.add_argument(
        "--peft_method",
        type=str,
        default="none",
        choices=["none", "norm", "last_blocks", "lora"],
        help="Used only when checkpoint does not contain model_config.",
    )
    parser.add_argument("--trainable_last_blocks", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_keywords", type=str, default="qkv,proj")
    parser.add_argument(
        "--keep_frozen_modules_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
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

    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model_config = resolve_model_config(ckpt, args)

    print("[INFO] resolved model_config:")
    print(json.dumps(model_config, indent=2, ensure_ascii=False))

    model = OutcomeModel(**model_config).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    recording = evaluate_recording_level(model, loader, device)
    patient = evaluate_patient_level(
        recording_probs=recording["probs"],
        recording_labels=recording["labels"],
        patient_ids=recording["patient_ids"],
        aggregation=args.patient_aggregation,
    )

    result = {
        "recording_level": recording["metrics"],
        "patient_level": patient,
        "patient_aggregation": args.patient_aggregation,
        "model_config": model_config,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
