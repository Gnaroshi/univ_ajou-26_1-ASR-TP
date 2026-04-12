from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from hs.data.circor import build_records_from_circor_root, save_records_json


def patient_level_split(records, train_ratio=0.7, valid_ratio=0.15, seed=42):
    patient_to_records = defaultdict(list)
    patient_to_label = {}

    for r in records:
        patient_to_records[r.patient_id].append(r)
        patient_to_label[r.patient_id] = r.outcome_label

    label_to_patients = defaultdict(list)
    for pid, label in patient_to_label.items():
        label_to_patients[label].append(pid)

    rng = random.Random(seed)

    train_ids, valid_ids, test_ids = [], [], []

    for label, pids in label_to_patients.items():
        rng.shuffle(pids)
        n = len(pids)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)

        train_ids.extend(pids[:n_train])
        valid_ids.extend(pids[n_train : n_train + n_valid])
        test_ids.extend(pids[n_train + n_valid :])

    split_map = {}
    for pid in train_ids:
        split_map[pid] = "train"
    for pid in valid_ids:
        split_map[pid] = "valid"
    for pid in test_ids:
        split_map[pid] = "test"

    for r in records:
        r.split = split_map[r.patient_id]

    return records


def summarize(records):
    split_counter = Counter(r.split for r in records)
    label_counter = Counter((r.split, r.outcome_text) for r in records)

    print("=== Split counts ===")
    for k, v in split_counter.items():
        print(f"{k}: {v}")

    print("\n=== Label counts by split ===")
    for k, v in sorted(label_counter.items()):
        print(f"{k}: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--circor_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="artifacts/circor")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = build_records_from_circor_root(args.circor_root)
    records = patient_level_split(records, seed=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_records_json(records, str(out_dir / "records.json"))

    with open(out_dir / "records_train.json", "w", encoding="utf-8") as f:
        json.dump(
            [r.__dict__ for r in records if r.split == "train"],
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(out_dir / "records_valid.json", "w", encoding="utf-8") as f:
        json.dump(
            [r.__dict__ for r in records if r.split == "valid"],
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(out_dir / "records_test.json", "w", encoding="utf-8") as f:
        json.dump(
            [r.__dict__ for r in records if r.split == "test"],
            f,
            indent=2,
            ensure_ascii=False,
        )

    summarize(records)
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
