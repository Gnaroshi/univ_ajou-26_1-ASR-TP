# outputs.txt Command Result Summary

Source: `outputs.txt` in this repository.

Notes:

- The log was captured from `/home/mingyujung/private/ajou_grad/hs`.
- All successful model runs used `.venv-m2d`, CUDA, the M2D checkpoint under
  `weights/m2d`, and `scripts/patch_m2d_portable.py`.
- Some blocks are duplicated in `outputs.txt`; duplicates are marked below.
- Test metrics are shown as recording-level and patient-level
  `accuracy / f1 / auroc`.

## Training Commands

| # | Command / run | Result |
|---|---|---|
| 1 | `PEFT_METHOD=none OUTPUT_DIR=outputs/circor_outcome_m2d_none bash scripts/train_circor_outcome_peft.sh` | Success. Frozen M2D encoder, trainable head only. Trainable params: `1,967,618 / 87,460,612` (`2.2497%`). Best validation: epoch 6, `valid_auroc=0.6754`, `valid_acc=0.6436`, `valid_f1=0.4828`. Final epoch 10: `valid_auroc=0.6616`. This block appears twice in the log. |
| 2 | `PEFT_METHOD=norm EPOCHS=1 BATCH_SIZE=2 LR=1e-4 ENCODER_LR=1e-5 OUTPUT_DIR=outputs/debug_norm bash scripts/train_circor_outcome_peft.sh` | Success debug run. Trainable params: `2,089,730 / 87,460,612` (`2.3893%`). Encoder trainable: `122,112 / 85,492,994` (`0.1428%`). Epoch 1: `valid_auroc=0.6413`, `valid_acc=0.6242`, `valid_f1=0.4238`. |
| 3 | `PEFT_METHOD=lora LORA_RANK=4 LORA_ALPHA=8.0 EPOCHS=1 BATCH_SIZE=2 OUTPUT_DIR=outputs/debug_lora_r4 bash scripts/train_circor_outcome_peft.sh` | Success debug run. Trainable params: `2,188,802 / 87,681,796` (`2.4963%`). Encoder trainable: `221,184 / 85,714,178` (`0.2580%`). Epoch 1: `valid_auroc=0.6403`, `valid_acc=0.6523`, `valid_f1=0.4984`. |
| 4 | `PEFT_METHOD=last_blocks TRAINABLE_LAST_BLOCKS=1 EPOCHS=1 BATCH_SIZE=1 ENCODER_LR=1e-6 OUTPUT_DIR=outputs/debug_last1 bash scripts/train_circor_outcome_peft.sh` | Success debug run. Trainable params: `9,055,490 / 87,460,612` (`10.3538%`). Encoder trainable: `7,087,872 / 85,492,994` (`8.2906%`). Epoch 1: `valid_auroc=0.6485`, `valid_acc=0.6285`, `valid_f1=0.4110`. A repeated command immediately after this has no result captured. |
| 5 | `PEFT_METHOD=norm EPOCHS=15 BATCH_SIZE=4 ENCODER_LR=1e-5 OUTPUT_DIR=outputs/circor_outcome_m2d_norm_elr1e-5 bash scripts/train_circor_outcome_peft.sh` | Success. Best validation: epoch 6, `valid_auroc=0.6799`, `valid_acc=0.6609`, `valid_f1=0.5199`. Final epoch 15: `valid_auroc=0.6465`. |
| 6 | `PEFT_METHOD=norm EPOCHS=15 BATCH_SIZE=4 ENCODER_LR=3e-5 OUTPUT_DIR=outputs/circor_outcome_m2d_norm_elr3e-5 bash scripts/train_circor_outcome_peft.sh` | Success. Best validation: epoch 3, `valid_auroc=0.6817`, `valid_acc=0.6069`, `valid_f1=0.6345`. Final epoch 15: `valid_auroc=0.6447`. |
| 7 | `PEFT_METHOD=lora LORA_RANK=4 LORA_ALPHA=8.0 EPOCHS=15 OUTPUT_DIR=outputs/circor_outcome_m2d_lora_r4 bash scripts/train_circor_outcome_peft.sh` | Success. Trainable params: `2,188,802 / 87,681,796` (`2.4963%`). Best validation: epoch 7, `valid_auroc=0.7067`, `valid_acc=0.6544`, `valid_f1=0.5238`. Final epoch 15: `valid_auroc=0.6894`. |
| 8 | `PEFT_METHOD=lora LORA_RANK=8 LORA_ALPHA=16.0 EPOCHS=15 OUTPUT_DIR=outputs/circor_outcome_m2d_lora_r8 bash scripts/train_circor_outcome_peft.sh` | Success. Trainable params: `2,409,986 / 87,902,980` (`2.7416%`). Best validation: epoch 7, `valid_auroc=0.7031`, `valid_acc=0.6285`, `valid_f1=0.6037`. Final epoch 15: `valid_auroc=0.6599`. |
| 9 | Python history summary over `outputs/circor_outcome_m2d_none`, `norm_elr1e-5`, `norm_elr3e-5`, `lora_r4`, `lora_r8` | Success. Best validation AUROC ranking from the printed summary: `lora_r4=0.7067`, `lora_r8=0.7031`, `norm_elr3e-5=0.6817`, `norm_elr1e-5=0.6799`, `none=0.6754`. |
| 10 | `PEFT_METHOD=last_blocks TRAINABLE_LAST_BLOCKS=1 EPOCHS=10 BATCH_SIZE=2 ENCODER_LR=1e-6 OUTPUT_DIR=outputs/circor_outcome_m2d_last1_elr1e-6 bash scripts/train_circor_outcome_peft.sh` | Success. Best validation: epoch 10, `valid_auroc=0.6723`, `valid_acc=0.6458`, `valid_f1=0.4843`. |
| 11 | `PEFT_METHOD=last_blocks TRAINABLE_LAST_BLOCKS=1 EPOCHS=10 BATCH_SIZE=2 ENCODER_LR=3e-6 OUTPUT_DIR=outputs/circor_outcome_m2d_last1_elr3e-6 bash scripts/train_circor_outcome_peft.sh` | Success. Best validation: epoch 10, `valid_auroc=0.6858`, `valid_acc=0.6393`, `valid_f1=0.4798`. |

## Evaluation Commands

| # | Command / run | Aggregation | Recording result | Patient result |
|---|---|---|---|---|
| 1 | `RUN_DIR=outputs/circor_outcome_m2d_none bash scripts/eval_circor_outcome_peft.sh` | `mean` | `0.5976 / 0.3865 / 0.6543` | `0.6294 / 0.4646 / 0.6680` |
| 2 | `RUN_DIR=outputs/circor_outcome_m2d_none PATIENT_AGGREGATION=max bash scripts/eval_circor_outcome_peft.sh` | `max` | `0.5976 / 0.3865 / 0.6543` | `0.5804 / 0.4643 / 0.6396` |
| 3 | `RUN_DIR=outputs/circor_outcome_m2d_none PATIENT_AGGREGATION=noisy_or bash scripts/eval_circor_outcome_peft.sh` | `noisy_or` | `0.5976 / 0.3865 / 0.6543` | `0.4825 / 0.6373 / 0.6584` |
| 4 | `RUN_DIR=outputs/circor_outcome_m2d_norm_elr1e-5 PATIENT_AGGREGATION=mean bash scripts/eval_circor_outcome_peft.sh` | `mean` | `0.5936 / 0.3879 / 0.6492` | `0.6224 / 0.4600 / 0.6602` |
| 5 | Loop eval for `outputs/circor_outcome_m2d_norm_elr1e-5` | `max` | `0.5936 / 0.3879 / 0.6492` | `0.5664 / 0.4561 / 0.6336` |
| 6 | Loop eval for `outputs/circor_outcome_m2d_norm_elr1e-5` | `noisy_or` | `0.5936 / 0.3879 / 0.6492` | `0.4825 / 0.6373 / 0.6516` |
| 7 | `RUN_DIR=outputs/circor_outcome_m2d_lora_r4 PATIENT_AGGREGATION=mean bash scripts/eval_circor_outcome_peft.sh` | `mean` | `0.6157 / 0.4366 / 0.6366` | `0.6224 / 0.4600 / 0.6579` |
| 8 | Loop eval for `outputs/circor_outcome_m2d_lora_r4` | `max` | `0.6157 / 0.4366 / 0.6366` | `0.6434 / 0.5785 / 0.6455` |
| 9 | Loop eval for `outputs/circor_outcome_m2d_lora_r4` | `noisy_or` | `0.6157 / 0.4366 / 0.6366` | `0.4825 / 0.6408 / 0.6406` |

## Main Takeaways

- Best validation AUROC in `outputs.txt`: `outputs/circor_outcome_m2d_lora_r4`
  with `valid_auroc=0.7067` at epoch 7.
- Best patient-level test AUROC among explicit eval blocks:
  `outputs/circor_outcome_m2d_none` with `PATIENT_AGGREGATION=mean`,
  `patient_auroc=0.6680`.
- Best patient-level test accuracy among explicit eval blocks:
  `outputs/circor_outcome_m2d_lora_r4` with `PATIENT_AGGREGATION=max`,
  `patient_acc=0.6434`.
- `noisy_or` tends to predict many positives in these logs, giving high class-1
  F1 but low patient-level accuracy.
