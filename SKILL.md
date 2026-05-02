---
name: ajou-asr-heart-sound
description: "Work on the univ_ajou-26_1-ASR-TP Ajou graduate ASR course project for heart sound classification and domain adaptation. Use when Codex needs to implement, debug, document, or plan experiments involving M2D/M2D-X audio backbones, CirCor outcome normal/abnormal classification, BMD-HS multi-label disease classification, coarse-to-fine transfer learning, joint multi-task learning, PEFT, or evaluation for this repository."
---

# Ajou ASR Heart Sound Project

## Core Framing

Treat this project as heart sound domain adaptation research, not an audio-language assistant and not merely another dataset-specific classifier.

Use the primary goal: adapt a general-purpose audio representation model, especially the M2D-X/M2D family, to heart sound data and improve low-resource heart disease classification.

Preserve the main hypothesis: coarse clinical abnormality supervision from CirCor normal/abnormal outcome can improve fine-grained BMD-HS disease classification.

## Datasets And Tasks

- Use CirCor as the source dataset for heart sound domain adaptation and binary outcome classification: normal vs abnormal.
- Interpret the CirCor outcome label as coarse clinical abnormality supervision.
- Use BMD-HS as the target dataset for low-resource fine-grained heart disease multi-label classification.
- Treat Aortic Stenosis, Aortic Regurgitation, Mitral Regurgitation, and Mitral Stenosis as the default BMD-HS target diseases unless the user defines another label set.
- Before implementing BMD-HS experiments, define how to handle normal samples, Multi Disease labels, patient-level vs recording-level splits, and label imbalance.
- Prefer patient-level splits for medical audio evaluation unless a task specifically calls for recording-level analysis.

## Model Direction

- Prefer M2D-X or the current M2D/PortableM2D backbone family for classification and representation transfer.
- Do not redirect the project to Qwen2-Audio or audio-language generation unless the user explicitly asks; that was an earlier rejected direction.
- Use the encoder as a shared representation model and attach task-specific classification heads.
- Reuse PEFT strategies already present in the repo when appropriate: frozen encoder, norm tuning, last blocks, or LoRA.

## Experiment Strategies

Compare these methods when implementing the full research plan:

- Method A, sequential transfer: train M2D-X/M2D on CirCor outcome first, then initialize BMD-HS disease classification from the adapted encoder.
- Method B, joint multi-task: train CirCor and BMD-HS together with a shared encoder and separate heads, alternating batches and updating the relevant head.
- Method C, direct fine-tuning baseline: train only on BMD-HS without CirCor pre-adaptation.

Use Method C as the baseline for whether CirCor-based adaptation helps. Use A vs B to study sequential transfer against multi-task regularization. Consider negative transfer as a valid outcome.

## Current Repository Shape

- Python package: `src/hs`.
- Current implemented focus: CirCor outcome classification with an M2D-based baseline.
- Key files:
  - `src/hs/train.py`: CirCor outcome training.
  - `src/hs/eval.py`: CirCor outcome evaluation with recording-level and patient-level metrics.
  - `src/hs/models/m2d_wrapper.py`: wrapper around `external/m2d/examples/portable_m2d.py`.
  - `src/hs/models/outcome.py`: M2D encoder plus binary classification head.
  - `src/hs/models/peft.py`: PEFT configuration and trainability helpers.
  - `scripts/prepare_circor.py`: create CirCor manifests.
  - `scripts/train_circor_outcome*.sh` and `scripts/eval_circor_outcome*.sh`: existing run entrypoints.
- Note that BMD-HS data loading, disease heads, sequential transfer, and joint multi-task code may not yet exist. Add them consistently with current `hs` package patterns.

## Setup And Commands

Use the project root as the working directory.

Prepare dependencies:

```bash
uv venv
source .venv/bin/activate
uv sync
```

Prepare CirCor manifests:

```bash
export CIRCOR_ROOT=/path/to/physionet.org
python scripts/prepare_circor.py --circor_root "$CIRCOR_ROOT" --out_dir artifacts/circor
```

Check and patch M2D portability before M2D runs:

```bash
python scripts/check_m2d_portable.py
python scripts/patch_m2d_portable.py
```

Train and evaluate the current CirCor baseline:

```bash
bash scripts/train_circor_outcome.sh
bash scripts/eval_circor_outcome.sh
```

Train and evaluate with PEFT:

```bash
PEFT_METHOD=norm bash scripts/train_circor_outcome_peft.sh
PEFT_METHOD=norm bash scripts/eval_circor_outcome_peft.sh
```

Expect M2D checkpoints under `weights/m2d/*.pth` unless the user sets `M2D_WEIGHT`.

## Implementation Guidance

- Preserve the `src/hs` package structure and existing CLI style.
- Keep manifest-driven datasets and explicit train/valid/test splits.
- Save checkpoints with model config and CLI args so evaluation can rebuild PEFT and model structure.
- For binary CirCor outcome, use cross-entropy logits with metrics such as accuracy, F1, and AUROC.
- For BMD-HS multi-label disease classification, use sigmoid outputs with binary cross-entropy style losses and report macro F1, micro F1, AUROC, AUPRC, and class-wise metrics.
- For imbalanced labels, prefer macro and class-wise metrics over accuracy alone; consider class weights or sampling only when justified by the experiment.
- Keep result artifacts under `outputs/` and generated manifests/checkpoints under ignored artifact/output directories.
- Avoid committing datasets, pretrained weights, checkpoints, and bulky generated outputs.

## Evaluation And Interpretation

When summarizing results:

- A > C suggests CirCor coarse outcome pre-adaptation helps BMD-HS disease classification.
- B > A suggests maintaining CirCor supervision may regularize the shared representation.
- A > B suggests target-task specialization may matter more than joint supervision or that negative transfer occurred.
- C > A and C > B suggests task mismatch, source label noise, domain shift, or direct BMD-HS tuning may be stronger.

Also examine secondary CirCor outcome performance when comparing A and B:

- Large CirCor drop after BMD-HS tuning may indicate catastrophic forgetting.
- Stable CirCor and BMD-HS performance in Method B may indicate better general representations.
- Good CirCor but poor BMD-HS performance in Method B may indicate source-task interference.

## Research Writing Guidance

Use the following positioning in papers, reports, slides, and READMEs:

- Title direction: Heart Sound Domain Adaptation using General-Purpose Audio Representation.
- Key phrase: coarse-to-fine transfer learning with CirCor and BMD-HS.
- Emphasize adapting a general-purpose audio foundation/representation model to medical heart sounds.
- Describe CirCor as coarse clinical abnormality supervision and BMD-HS as fine-grained low-resource target disease classification.
- Avoid claiming a heart sound-specific foundation model unless the project actually trains one at scale.
- Present negative results honestly as evidence about domain mismatch, label mismatch, or transfer limits.

## Before Finalizing Work

- Run the smallest relevant validation available, such as import checks, unit-level smoke tests, or CLI `--help`.
- For training changes, at least verify that dataset construction, model initialization, and one short forward/training step can run when data and weights are available.
- Report blockers clearly when local datasets, pretrained M2D weights, or GPU access are missing.
