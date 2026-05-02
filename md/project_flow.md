# Project Flow

This document explains how the current `univ_ajou-26_1-ASR-TP` code is intended to run end to end, based on `SKILL.md`, `md/project_context_en.md`, and the source code in the repository.

The important distinction is that the research plan is broader than the implemented code. The research direction is M2D/M2D-X based heart sound domain adaptation from CirCor to BMD-HS, but the current code mainly implements the first stage: CirCor outcome classification with an M2D backbone and optional PEFT.

## 1. High-Level Research Position

The project is framed as heart sound domain adaptation using a general-purpose audio representation model.

- Source dataset: CirCor
- Current implemented task: CirCor outcome binary classification, normal vs abnormal
- Target dataset in the research plan: BMD-HS
- Target task in the research plan: multi-label disease classification for AS, AR, MR, and MS
- Backbone direction: M2D/M2D-X family rather than Qwen2-Audio or a generative audio-language model
- Core hypothesis: a representation adapted on coarse CirCor normal/abnormal supervision may help fine-grained BMD-HS disease classification

The planned research comparison has three methods:

- Method A: train on CirCor outcome, then transfer the adapted encoder to BMD-HS disease classification.
- Method B: jointly train CirCor and BMD-HS with a shared encoder and task-specific heads.
- Method C: train directly on BMD-HS as a baseline.

Only the CirCor outcome stage is currently represented in executable training and evaluation code.

## 2. Repository-Level Entry Points

There is a `main.py`, but it is only a placeholder:

```text
main.py
└── prints "Hello from hs!"
```

The practical entry points are the scripts and Python modules below.

| Entry point | Role |
|---|---|
| `scripts/prepare_circor.py` | Builds CirCor record manifests and patient-level train/valid/test splits. |
| `scripts/check_m2d_portable.py` | Smoke-tests the external M2D PortableM2D model and a local M2D checkpoint. |
| `scripts/patch_m2d_portable.py` | Patches the external M2D import for `timm` compatibility. |
| `scripts/train_circor_outcome.sh` | Runs the default CirCor outcome training script. |
| `scripts/train_circor_outcome_peft.sh` | Runs CirCor outcome training with configurable PEFT settings. |
| `scripts/eval_circor_outcome.sh` | Evaluates the default CirCor outcome checkpoint. |
| `scripts/eval_circor_outcome_peft.sh` | Evaluates a PEFT or custom CirCor outcome run. |
| `python -m hs.train` | Main Python training module for CirCor outcome classification. |
| `python -m hs.eval` | Main Python evaluation module for CirCor outcome classification. |

The package name is `hs`, and `pyproject.toml` maps it to `src/hs`.

## 3. Current Package Structure

```text
src/hs
├── __init__.py
├── config.py
├── train.py
├── eval.py
├── utils.py
└── models
    ├── classifier.py
    ├── m2d_wrapper.py
    ├── outcome.py
    └── peft.py
```

The current code imports two data modules that are not present in this checkout:

```python
from hs.data.circor import CirCorOutcomeDataset
from hs.data.preprocess import collate_circor_batch
```

and:

```python
from hs.data.circor import build_records_from_circor_root, save_records_json
```

Therefore, the intended flow is clear, but this repository state cannot run the data preparation, training, or evaluation entry points until `src/hs/data/circor.py` and `src/hs/data/preprocess.py` are restored or implemented.

Also, `external/m2d` exists as a directory but contains no source files in this checkout. The M2D submodule code must be available there, especially `external/m2d/examples/portable_m2d.py`, for M2D model loading to work.

## 4. End-to-End Flow

The intended pipeline is:

```text
Raw CirCor dataset
    ↓
scripts/prepare_circor.py
    ↓
artifacts/circor/records_train.json
artifacts/circor/records_valid.json
artifacts/circor/records_test.json
    ↓
scripts/train_circor_outcome*.sh
    ↓
python -m hs.train
    ↓
OutcomeModel
    ├── M2DWrapper
    │   └── external/m2d/examples/portable_m2d.py
    └── ClassificationHead
    ↓
outputs/<run_name>/history.json
outputs/<run_name>/last.pt
outputs/<run_name>/best.pt
    ↓
scripts/eval_circor_outcome*.sh
    ↓
python -m hs.eval
    ↓
recording-level metrics
patient-level metrics
test_metrics*.json
```

## 5. Environment And Package Setup

The README describes the default setup:

```bash
uv venv
source .venv/bin/activate
uv sync
```

The shell scripts currently activate `.venv-m2d` instead:

```bash
source .venv-m2d/bin/activate
```

So the codebase has two environment conventions:

- README convention: `.venv`
- Run-script convention: `.venv-m2d`

The scripts assume that the working directory is the project root. Each script begins by moving from `scripts/` to the repository root:

```bash
cd "$(dirname "$0")/.."
```

## 6. Data Preparation Flow

The data preparation command in the README is:

```bash
export CIRCOR_ROOT=/path/to/physionet.org
python scripts/prepare_circor.py --circor_root "$CIRCOR_ROOT" --out_dir artifacts/circor
```

The execution flow in `scripts/prepare_circor.py` is:

1. Parse `--circor_root`, `--out_dir`, and `--seed`.
2. Call `build_records_from_circor_root(args.circor_root)` from `hs.data.circor`.
3. Pass the returned records to `patient_level_split`.
4. Save the full records list to `records.json` using `save_records_json`.
5. Save split-specific JSON files:
   - `records_train.json`
   - `records_valid.json`
   - `records_test.json`
6. Print split counts and label counts.

The split logic is patient-level and label-stratified:

```text
records
    ↓ group by patient_id
patient_to_label
    ↓ group patient IDs by outcome label
shuffle each label group with seed
    ↓
70 percent train
15 percent valid
remaining test
    ↓
write split value back into every record
```

The records are expected to have at least:

- `patient_id`
- `outcome_label`
- `outcome_text`
- `split`

Training and evaluation later expect the dataset object to return batches containing:

- `audio`
- `label`
- `patient_id`

## 7. M2D Preparation Flow

Before training or evaluation, the shell scripts run:

```bash
python scripts/patch_m2d_portable.py
```

This script edits:

```text
external/m2d/examples/portable_m2d.py
```

It replaces a direct `timm.layers` import with a compatibility fallback:

```python
try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_
```

`scripts/check_m2d_portable.py` is a standalone smoke test:

1. Check that `external/m2d` exists.
2. Add `external/m2d` to `sys.path`.
3. Import `PortableM2D` from `examples.portable_m2d`.
4. Find the first `*.pth` checkpoint under `weights/m2d`.
5. Build `PortableM2D`.
6. Run random 10-second audio shaped `(2, 160000)` through the model.
7. Print frame-level and clip-level feature shapes.

This validates the external M2D dependency independently of the project training loop.

## 8. Training Flow

### 8.1 Shell Script Layer

The default training script is `scripts/train_circor_outcome.sh`.

It does the following:

1. Move to the repository root.
2. Activate `.venv-m2d`.
3. Patch `portable_m2d.py`.
4. Find the first M2D checkpoint under `weights/m2d`.
5. Exit if no checkpoint is found.
6. Launch:

```bash
CUDA_VISIBLE_DEVICES=3 python -m hs.train \
  --train_manifest artifacts/circor/records_train.json \
  --valid_manifest artifacts/circor/records_valid.json \
  --m2d_weight_path "$M2D_WEIGHT" \
  --output_dir outputs/circor_outcome_m2d \
  --epochs 10 \
  --batch_size 8 \
  --num_workers 4 \
  --lr 1e-4 \
  --max_sec 10.0
```

The PEFT training script, `scripts/train_circor_outcome_peft.sh`, is the more flexible version. It reads environment variables such as:

- `PEFT_METHOD`, default `norm`
- `TRAINABLE_LAST_BLOCKS`, default `1`
- `LORA_RANK`, default `8`
- `LORA_ALPHA`, default `16.0`
- `LORA_DROPOUT`, default `0.05`
- `LORA_TARGET_KEYWORDS`, default `qkv,proj`
- `EPOCHS`, default `10`
- `BATCH_SIZE`, default `4`
- `LR`, default `1e-4`
- `ENCODER_LR`, default `1e-5`
- `WEIGHT_DECAY`, default `1e-4`
- `MAX_SEC`, default `10.0`
- `OUTPUT_DIR`, default `outputs/circor_outcome_m2d_${PEFT_METHOD}`

Then it launches `python -m hs.train` with these values.

### 8.2 Python Training Module

The actual training logic is in `src/hs/train.py`.

`main()` performs this sequence:

```text
parse CLI args
    ↓
set_seed(args.seed)
ensure_dir(args.output_dir)
    ↓
device = get_device()
    ↓
build train DataLoader
build valid DataLoader
    ↓
build OutcomeModel
    ↓
print trainability and PEFT report
    ↓
criterion = CrossEntropyLoss
optimizer = build_optimizer(...)
    ↓
for each epoch:
    train_one_epoch(...)
    evaluate(...)
    append metrics to history
    write history.json
    save last.pt
    if valid_auroc improves:
        save best.pt
```

### 8.3 DataLoader Construction

`build_loader` creates:

```python
CirCorOutcomeDataset(
    manifest_path=manifest_path,
    sample_rate=16000,
    max_sec=max_sec,
    crop_mode="random" if shuffle else "center",
)
```

Then it wraps the dataset with:

```python
DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=collate_circor_batch,
)
```

The training loader uses random cropping because `shuffle=True`. The validation loader uses center cropping because `shuffle=False`.

### 8.4 Model Construction

Training builds:

```python
OutcomeModel(
    m2d_weight_path=args.m2d_weight_path,
    num_classes=2,
    head_hidden_dim=512,
    peft_method=args.peft_method,
    trainable_last_blocks=args.trainable_last_blocks,
    lora_rank=args.lora_rank,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    lora_target_keywords=args.lora_target_keywords,
    keep_frozen_modules_eval=args.keep_frozen_modules_eval,
)
```

`OutcomeModel` combines two components:

```text
audio waveform
    ↓
M2DWrapper
    ↓
clip-level feature vector
    ↓
ClassificationHead
    ↓
2-class logits
```

The forward path is:

```python
feats = self.encoder(audio)
logits = self.head(feats)
return logits
```

## 9. M2DWrapper Flow

`src/hs/models/m2d_wrapper.py` adapts external M2D to the local classifier.

Initialization:

1. Resolve the project root from the file path.
2. Locate `external/m2d`.
3. Add `external/m2d` to `sys.path`.
4. Import `PortableM2D` from `examples.portable_m2d`.
5. Build `PortableM2D(weight_path)`.
6. Apply PEFT with `configure_peft`.
7. Detect whether any encoder parameter is trainable.
8. Run a dummy 10-second waveform through M2D to infer `out_dim`.

Forward pass:

```text
audio: (B, T)
    ↓
PortableM2D(audio)
    ↓
frame_feats: (B, frames, D)
    ↓ mean over frame dimension
clip_feats: (B, D)
```

If the encoder has no trainable parameters, the wrapper keeps M2D in eval mode and performs the M2D forward pass under `torch.no_grad()`. If PEFT makes encoder parameters trainable, gradients are allowed through those trainable parts.

The custom `train()` override keeps frozen M2D modules in eval mode when `keep_frozen_modules_eval=True`, while allowing trainable modules such as LoRA layers or selected transformer blocks to switch into train mode.

## 10. Classification Head

`src/hs/models/classifier.py` defines a small MLP head:

```text
Linear(in_dim, hidden_dim)
GELU
Dropout(0.1)
Linear(hidden_dim, num_classes)
```

For the current CirCor task, `num_classes=2`, so the model outputs two logits.

## 11. PEFT Flow

`src/hs/models/peft.py` supports four encoder adaptation modes:

| Method | Behavior |
|---|---|
| `none` | Freeze all M2D encoder parameters. Only the classifier head trains. |
| `norm` | Freeze all encoder parameters, then unfreeze norm-layer parameters and bias parameters. |
| `last_blocks` | Freeze the encoder, find transformer-like blocks, then unfreeze the last N blocks. |
| `lora` | Freeze the encoder, replace target `nn.Linear` modules with `LoRALinear`, and train only LoRA parameters. |

Important functions:

- `freeze_all` and `unfreeze_all`: toggle `requires_grad`.
- `parameter_stats`: count total, trainable, and frozen parameters.
- `format_parameter_stats`: format those counts for logs.
- `trainable_parameter_names`: show a preview of trainable parameters.
- `inject_lora`: replace target linear layers based on keywords.
- `apply_norm_tuning`: enable norm parameters and biases.
- `find_transformer_blocks`: locate likely transformer block lists.
- `apply_last_blocks_tuning`: unfreeze selected final blocks.
- `configure_peft`: central dispatcher used by `M2DWrapper`.

The training script prints the PEFT report before training, so the user can confirm which parameters are trainable.

## 12. Optimizer Flow

`build_optimizer` in `src/hs/train.py` separates trainable parameters into two groups:

```text
head parameters
encoder parameters
```

The head uses `lr`. The encoder group uses `encoder_lr` if provided, otherwise it also uses `lr`.

If no trainable parameters exist, training raises:

```text
RuntimeError: No trainable parameters found.
```

In practice, the head is trainable, so this error should only happen after an unusual model change.

## 13. Training Step Logic

`train_one_epoch` runs the standard supervised loop:

```text
model.train()
for batch in loader:
    audio = batch["audio"].to(device)
    labels = batch["label"].to(device)

    optimizer.zero_grad(set_to_none=True)
    logits = model(audio)
    loss = CrossEntropyLoss(logits, labels)
    loss.backward()
    optimizer.step()

    probs = softmax(logits)[:, 1]
    collect probs and labels
```

At the end of the epoch, it computes:

- loss
- accuracy
- F1
- AUROC

`evaluate` uses the same metric logic but runs under `torch.no_grad()` and does not update weights.

## 14. Checkpoint And History Flow

At each epoch, `src/hs/train.py` writes:

```text
outputs/<run_name>/history.json
outputs/<run_name>/last.pt
```

When validation AUROC improves, it also writes:

```text
outputs/<run_name>/best.pt
```

Each checkpoint stores:

- `model`: model state dict
- `optimizer`: optimizer state dict
- `epoch`: epoch number
- `best_valid_auroc`: best validation AUROC known at that point
- `model_config`: model construction parameters from `OutcomeModel.get_config()`
- `args`: training CLI arguments

Saving `model_config` is important because PEFT, especially LoRA, changes the module structure. Evaluation must rebuild the same structure before strict checkpoint loading.

## 15. Evaluation Flow

### 15.1 Shell Script Layer

The default script `scripts/eval_circor_outcome.sh`:

1. Moves to the repository root.
2. Activates `.venv-m2d`.
3. Patches `portable_m2d.py`.
4. Finds the first M2D checkpoint under `weights/m2d`.
5. Runs `python -m hs.eval` with:

```bash
python -m hs.eval \
  --manifest artifacts/circor/records_test.json \
  --checkpoint outputs/circor_outcome_m2d/best.pt \
  --m2d_weight_path "$M2D_WEIGHT" \
  --batch_size 8 \
  --num_workers 4 \
  --max_sec 10.0 \
  --output_json outputs/circor_outcome_m2d/test_metrics.json
```

The PEFT evaluation script adds configurable values:

- `RUN_DIR`
- `PATIENT_AGGREGATION`, one of `mean`, `max`, `noisy_or`
- `PEFT_METHOD` and PEFT parameters, used only if the checkpoint lacks `model_config`

### 15.2 Python Evaluation Module

The actual evaluation logic is in `src/hs/eval.py`.

`main()` performs this sequence:

```text
parse CLI args
    ↓
device = get_device()
    ↓
build CirCorOutcomeDataset with center crop
build DataLoader
    ↓
load checkpoint with torch.load
    ↓
resolve model_config
    ↓
build OutcomeModel
load model state dict strictly
    ↓
evaluate recording-level probabilities
    ↓
aggregate probabilities by patient
    ↓
compute patient-level metrics
    ↓
print and save JSON result
```

### 15.3 Model Config Resolution

`resolve_model_config` handles two checkpoint styles:

- If the checkpoint contains `model_config`, use it and override only `m2d_weight_path` from the CLI.
- If not, rebuild config from CLI arguments.

This lets a checkpoint trained on one machine be evaluated on another machine where the M2D weight path is different.

### 15.4 Recording-Level Metrics

`evaluate_recording_level`:

1. Runs each recording through the model.
2. Converts logits to abnormal-class probability using `softmax(logits)[:, 1]`.
3. Collects labels and patient IDs.
4. Computes metrics using a threshold of `0.5`.

Metrics include:

- accuracy
- F1
- confusion matrix
- classification report
- AUROC

### 15.5 Patient-Level Metrics

Patient-level evaluation groups recording probabilities by `patient_id`.

Supported aggregation methods:

- `mean`: average recording probabilities
- `max`: use the maximum recording probability
- `noisy_or`: compute `1 - product(1 - p)` across recordings

After aggregation, the same binary metrics are computed at patient level.

## 16. Utility And Config Modules

`src/hs/utils.py` provides:

- `set_seed`: seeds Python `random`, NumPy, PyTorch CPU, and PyTorch CUDA.
- `ensure_dir`: creates an output directory.
- `get_device`: returns `"cuda"` when available, otherwise `"cpu"`.

`src/hs/config.py` defines dataclasses:

- `DataConfig`
- `TrainConfig`
- `ModelConfig`

These dataclasses are not currently wired into `hs.train` or `hs.eval`; the active code uses argparse instead.

`configs/circor_outcome.yaml` stores task defaults:

```yaml
task: circor_outcome
sample_rate: 16000
max_sec: 10.0
batch_size: 8
num_workers: 4
lr: 1e-4
epochs: 10
seed: 42
num_classes: 2
```

This YAML is also not currently loaded by the training code. It is best understood as a future or reference configuration file.

## 17. Observed Experiment Logs

`outputs.txt` contains prior terminal logs for CirCor outcome experiments.

It shows experiments for:

- `PEFT_METHOD=none`
- `PEFT_METHOD=norm`
- `PEFT_METHOD=lora`
- `PEFT_METHOD=last_blocks`

The logs confirm the intended trainability patterns:

- `none`: encoder frozen, only head trainable.
- `norm`: a small number of encoder norm and bias parameters trainable.
- `lora`: LoRA parameters trainable in selected linear layers.
- `last_blocks`: the final transformer block trainable.

The logs also show that validation AUROC is used to select `best.pt`.

## 18. Dependency Graph

```text
scripts/prepare_circor.py
└── hs.data.circor
    ├── build_records_from_circor_root
    └── save_records_json

scripts/train_circor_outcome*.sh
└── python -m hs.train
    ├── hs.data.circor.CirCorOutcomeDataset
    ├── hs.data.preprocess.collate_circor_batch
    ├── hs.models.outcome.OutcomeModel
    │   ├── hs.models.m2d_wrapper.M2DWrapper
    │   │   ├── external/m2d/examples/portable_m2d.PortableM2D
    │   │   └── hs.models.peft.configure_peft
    │   └── hs.models.classifier.ClassificationHead
    └── hs.utils

scripts/eval_circor_outcome*.sh
└── python -m hs.eval
    ├── hs.data.circor.CirCorOutcomeDataset
    ├── hs.data.preprocess.collate_circor_batch
    ├── hs.models.outcome.OutcomeModel
    └── hs.utils.get_device
```

## 19. Current Gaps And Run Blockers

The current repository state has several important blockers:

1. `src/hs/data/circor.py` is missing.
2. `src/hs/data/preprocess.py` is missing.
3. `external/m2d` contains no source files in this checkout.
4. `weights/m2d/*.pth` is required by the scripts but weights are intentionally not stored in the repository.
5. Actual CirCor data under `CIRCOR_ROOT` is required to build manifests.
6. BMD-HS data loading, disease heads, Method A, Method B, and Method C are research-plan items, not current implemented code.

Because of these gaps, this document describes both the intended execution flow and the missing pieces needed to make that flow executable in the current checkout.

## 20. How Future BMD-HS Work Should Extend This Flow

To implement the broader research plan, extend the current structure rather than replacing it.

Recommended extension points:

- Add BMD-HS dataset and preprocessing modules under `src/hs/data`.
- Add a multi-label disease model or head that reuses `M2DWrapper`.
- For Method A, load the encoder state from a CirCor-trained checkpoint and initialize BMD-HS training from it.
- For Method B, build a shared encoder model with two heads: one for CirCor outcome and one for BMD-HS diseases.
- For Method C, train the BMD-HS disease model directly from pretrained M2D/M2D-X without CirCor adaptation.
- Add evaluation code for macro F1, micro F1, AUROC, AUPRC, and class-wise metrics.
- Keep patient-level splitting and patient-level reporting as default medical-audio practice.

The current CirCor pipeline is therefore best viewed as the source-task adaptation stage that future BMD-HS transfer and multi-task experiments should build on.
