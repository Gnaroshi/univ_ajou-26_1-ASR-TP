# 프로젝트 흐름

이 문서는 `SKILL.md`, `md/project_context_en.md`, 그리고 저장소 내 소스코드를 바탕으로 현재 `univ_ajou-26_1-ASR-TP` 코드가 end-to-end로 어떤 흐름으로 실행되도록 설계되어 있는지 설명한다.

가장 중요한 점은 연구 계획의 범위가 현재 구현된 코드보다 넓다는 것이다. 연구 방향은 CirCor에서 BMD-HS로 이어지는 M2D/M2D-X 기반 심장음 도메인 적응이지만, 현재 실행 가능한 코드는 주로 첫 단계인 M2D backbone 기반 CirCor outcome classification과 선택적 PEFT를 구현하고 있다.

## 1. 상위 연구 위치

이 프로젝트는 범용 오디오 표현 모델을 이용한 심장음 도메인 적응 연구로 정의된다.

- Source dataset: CirCor
- 현재 구현된 task: CirCor outcome binary classification, normal vs abnormal
- 연구 계획상의 target dataset: BMD-HS
- 연구 계획상의 target task: AS, AR, MR, MS에 대한 multi-label disease classification
- Backbone 방향: Qwen2-Audio나 생성형 audio-language model이 아니라 M2D/M2D-X 계열
- 핵심 가설: CirCor의 coarse normal/abnormal supervision으로 적응한 representation이 BMD-HS의 fine-grained disease classification에 도움이 될 수 있다

연구 계획의 비교 실험은 세 가지 방법으로 구성된다.

- Method A: CirCor outcome으로 먼저 학습한 뒤, 적응된 encoder를 BMD-HS disease classification으로 transfer한다.
- Method B: shared encoder와 task-specific head를 사용하여 CirCor와 BMD-HS를 joint training한다.
- Method C: BMD-HS만 직접 학습하는 baseline이다.

현재 executable training/evaluation code에는 CirCor outcome 단계만 구현되어 있다.

## 2. 저장소 수준 Entry Point

`main.py`가 존재하지만, 실제 학습용 메인 코드는 아니고 placeholder이다.

```text
main.py
└── "Hello from hs!" 출력
```

실제로 사용하는 entry point는 아래의 script와 Python module이다.

| Entry point | 역할 |
|---|---|
| `scripts/prepare_circor.py` | CirCor record manifest와 patient-level train/valid/test split을 만든다. |
| `scripts/check_m2d_portable.py` | 외부 M2D PortableM2D model과 local M2D checkpoint를 smoke test한다. |
| `scripts/patch_m2d_portable.py` | `timm` 호환성을 위해 외부 M2D import를 patch한다. |
| `scripts/train_circor_outcome.sh` | 기본 CirCor outcome training script를 실행한다. |
| `scripts/train_circor_outcome_peft.sh` | configurable PEFT 설정으로 CirCor outcome training을 실행한다. |
| `scripts/eval_circor_outcome.sh` | 기본 CirCor outcome checkpoint를 평가한다. |
| `scripts/eval_circor_outcome_peft.sh` | PEFT 또는 custom CirCor outcome run을 평가한다. |
| `python -m hs.train` | CirCor outcome classification을 위한 main Python training module이다. |
| `python -m hs.eval` | CirCor outcome classification을 위한 main Python evaluation module이다. |

Package 이름은 `hs`이고, `pyproject.toml`에서 이를 `src/hs`에 매핑한다.

## 3. 현재 Package 구조

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

현재 코드는 아래 두 data module을 import하지만, 이 checkout에는 해당 파일들이 존재하지 않는다.

```python
from hs.data.circor import CirCorOutcomeDataset
from hs.data.preprocess import collate_circor_batch
```

그리고 아래 import도 사용한다.

```python
from hs.data.circor import build_records_from_circor_root, save_records_json
```

따라서 의도된 흐름은 명확하지만, 현재 저장소 상태에서는 `src/hs/data/circor.py`와 `src/hs/data/preprocess.py`를 복구하거나 구현하기 전까지 data preparation, training, evaluation entry point가 실행되지 않는다.

또한 `external/m2d` 디렉터리는 존재하지만 이 checkout에는 source file이 들어 있지 않다. M2D model loading이 동작하려면 특히 `external/m2d/examples/portable_m2d.py`를 포함한 M2D submodule code가 해당 위치에 있어야 한다.

## 4. End-to-End 흐름

의도된 전체 pipeline은 다음과 같다.

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

## 5. Environment 및 Package Setup

README에는 기본 setup이 아래처럼 적혀 있다.

```bash
uv venv
source .venv/bin/activate
uv sync
```

하지만 현재 shell script들은 대신 `.venv-m2d`를 activate한다.

```bash
source .venv-m2d/bin/activate
```

즉 코드베이스에는 두 가지 environment convention이 함께 존재한다.

- README convention: `.venv`
- Run-script convention: `.venv-m2d`

Script들은 working directory가 project root라고 가정한다. 각 script는 시작 부분에서 `scripts/`에서 repository root로 이동한다.

```bash
cd "$(dirname "$0")/.."
```

## 6. Data Preparation 흐름

README에 있는 data preparation command는 다음과 같다.

```bash
export CIRCOR_ROOT=/path/to/physionet.org
python scripts/prepare_circor.py --circor_root "$CIRCOR_ROOT" --out_dir artifacts/circor
```

`scripts/prepare_circor.py`의 실행 흐름은 다음과 같다.

1. `--circor_root`, `--out_dir`, `--seed`를 parse한다.
2. `hs.data.circor`의 `build_records_from_circor_root(args.circor_root)`를 호출한다.
3. 반환된 records를 `patient_level_split`에 전달한다.
4. `save_records_json`을 사용하여 전체 records를 `records.json`에 저장한다.
5. Split별 JSON file을 저장한다.
   - `records_train.json`
   - `records_valid.json`
   - `records_test.json`
6. Split count와 label count를 출력한다.

Split logic은 patient-level이고 label-stratified 방식이다.

```text
records
    ↓ patient_id 기준 grouping
patient_to_label
    ↓ outcome label 기준 patient ID grouping
각 label group을 seed로 shuffle
    ↓
70 percent train
15 percent valid
나머지 test
    ↓
모든 record에 split 값 기록
```

Record에는 최소한 아래 field가 있다고 가정된다.

- `patient_id`
- `outcome_label`
- `outcome_text`
- `split`

이후 training과 evaluation에서는 dataset object가 batch에 아래 값을 포함해 반환한다고 기대한다.

- `audio`
- `label`
- `patient_id`

## 7. M2D 준비 흐름

Training 또는 evaluation 전에 shell script들은 아래 command를 실행한다.

```bash
python scripts/patch_m2d_portable.py
```

이 script는 아래 파일을 수정한다.

```text
external/m2d/examples/portable_m2d.py
```

직접적인 `timm.layers` import를 compatibility fallback이 있는 형태로 바꾼다.

```python
try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_
```

`scripts/check_m2d_portable.py`는 독립적인 smoke test이다.

1. `external/m2d`가 존재하는지 확인한다.
2. `external/m2d`를 `sys.path`에 추가한다.
3. `examples.portable_m2d`에서 `PortableM2D`를 import한다.
4. `weights/m2d` 아래의 첫 번째 `*.pth` checkpoint를 찾는다.
5. `PortableM2D`를 생성한다.
6. `(2, 160000)` shape의 random 10-second audio를 model에 통과시킨다.
7. Frame-level 및 clip-level feature shape를 출력한다.

이는 project training loop와 별개로 external M2D dependency가 정상 동작하는지 검증한다.

## 8. Training 흐름

### 8.1 Shell Script Layer

기본 training script는 `scripts/train_circor_outcome.sh`이다.

이 script는 다음 작업을 수행한다.

1. Repository root로 이동한다.
2. `.venv-m2d`를 activate한다.
3. `portable_m2d.py`를 patch한다.
4. `weights/m2d` 아래에서 첫 번째 M2D checkpoint를 찾는다.
5. Checkpoint가 없으면 종료한다.
6. 아래 command를 실행한다.

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

PEFT training script인 `scripts/train_circor_outcome_peft.sh`는 더 유연한 버전이다. 이 script는 아래 environment variable을 읽는다.

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

그 후 이 값들을 이용해 `python -m hs.train`을 실행한다.

### 8.2 Python Training Module

실제 training logic은 `src/hs/train.py`에 있다.

`main()`은 다음 순서로 실행된다.

```text
CLI args parse
    ↓
set_seed(args.seed)
ensure_dir(args.output_dir)
    ↓
device = get_device()
    ↓
train DataLoader 생성
valid DataLoader 생성
    ↓
OutcomeModel 생성
    ↓
trainability 및 PEFT report 출력
    ↓
criterion = CrossEntropyLoss
optimizer = build_optimizer(...)
    ↓
각 epoch마다:
    train_one_epoch(...)
    evaluate(...)
    metrics를 history에 append
    history.json 저장
    last.pt 저장
    valid_auroc가 개선되면:
        best.pt 저장
```

### 8.3 DataLoader 생성

`build_loader`는 아래 dataset을 생성한다.

```python
CirCorOutcomeDataset(
    manifest_path=manifest_path,
    sample_rate=16000,
    max_sec=max_sec,
    crop_mode="random" if shuffle else "center",
)
```

그 뒤 dataset을 아래 `DataLoader`로 감싼다.

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

Training loader는 `shuffle=True`이므로 random cropping을 사용한다. Validation loader는 `shuffle=False`이므로 center cropping을 사용한다.

### 8.4 Model 생성

Training은 아래와 같이 model을 생성한다.

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

`OutcomeModel`은 두 component를 결합한다.

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

Forward path는 다음과 같다.

```python
feats = self.encoder(audio)
logits = self.head(feats)
return logits
```

## 9. M2DWrapper 흐름

`src/hs/models/m2d_wrapper.py`는 external M2D를 local classifier에 맞게 감싼다.

Initialization 순서:

1. File path로부터 project root를 계산한다.
2. `external/m2d`를 찾는다.
3. `external/m2d`를 `sys.path`에 추가한다.
4. `examples.portable_m2d`에서 `PortableM2D`를 import한다.
5. `PortableM2D(weight_path)`를 생성한다.
6. `configure_peft`로 PEFT를 적용한다.
7. Encoder parameter 중 trainable parameter가 있는지 확인한다.
8. Dummy 10-second waveform을 M2D에 통과시켜 `out_dim`을 추론한다.

Forward pass:

```text
audio: (B, T)
    ↓
PortableM2D(audio)
    ↓
frame_feats: (B, frames, D)
    ↓ frame dimension에 대해 mean
clip_feats: (B, D)
```

Encoder에 trainable parameter가 없으면 wrapper는 M2D를 eval mode로 유지하고 `torch.no_grad()` 안에서 M2D forward pass를 수행한다. PEFT로 encoder parameter가 trainable해진 경우에는 해당 trainable part로 gradient가 흐를 수 있다.

Custom `train()` override는 `keep_frozen_modules_eval=True`일 때 frozen M2D module은 eval mode로 유지하면서, LoRA layer나 선택된 transformer block 같은 trainable module만 train mode로 전환할 수 있게 한다.

## 10. Classification Head

`src/hs/models/classifier.py`는 작은 MLP head를 정의한다.

```text
Linear(in_dim, hidden_dim)
GELU
Dropout(0.1)
Linear(hidden_dim, num_classes)
```

현재 CirCor task에서는 `num_classes=2`이므로 model은 두 개의 logit을 출력한다.

## 11. PEFT 흐름

`src/hs/models/peft.py`는 네 가지 encoder adaptation mode를 지원한다.

| Method | 동작 |
|---|---|
| `none` | 모든 M2D encoder parameter를 freeze한다. Classifier head만 학습된다. |
| `norm` | 모든 encoder parameter를 freeze한 뒤 norm-layer parameter와 bias parameter만 unfreeze한다. |
| `last_blocks` | Encoder를 freeze하고 transformer-like block을 찾은 뒤 마지막 N개 block만 unfreeze한다. |
| `lora` | Encoder를 freeze하고 target `nn.Linear` module을 `LoRALinear`로 교체한 뒤 LoRA parameter만 학습한다. |

중요 함수:

- `freeze_all` and `unfreeze_all`: `requires_grad`를 전환한다.
- `parameter_stats`: total, trainable, frozen parameter 수를 계산한다.
- `format_parameter_stats`: parameter count를 log용 문자열로 format한다.
- `trainable_parameter_names`: trainable parameter 이름 preview를 보여준다.
- `inject_lora`: keyword 기준으로 target linear layer를 교체한다.
- `apply_norm_tuning`: norm parameter와 bias를 활성화한다.
- `find_transformer_blocks`: transformer block list처럼 보이는 module을 찾는다.
- `apply_last_blocks_tuning`: 선택된 마지막 block들을 unfreeze한다.
- `configure_peft`: `M2DWrapper`가 사용하는 central dispatcher이다.

Training script는 학습 전에 PEFT report를 출력하므로, 어떤 parameter가 trainable인지 사용자가 확인할 수 있다.

## 12. Optimizer 흐름

`src/hs/train.py`의 `build_optimizer`는 trainable parameter를 두 group으로 나눈다.

```text
head parameters
encoder parameters
```

Head에는 `lr`을 사용한다. Encoder group에는 `encoder_lr`이 주어지면 그것을 사용하고, 없으면 `lr`을 사용한다.

Trainable parameter가 하나도 없으면 training은 아래 error를 발생시킨다.

```text
RuntimeError: No trainable parameters found.
```

실제로는 head가 trainable하므로, 이 error는 특이한 model 변경이 있을 때만 발생할 가능성이 높다.

## 13. Training Step Logic

`train_one_epoch`는 표준 supervised loop를 실행한다.

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
    probs와 labels 수집
```

Epoch 끝에서는 아래 값을 계산한다.

- loss
- accuracy
- F1
- AUROC

`evaluate`는 같은 metric logic을 사용하지만 `torch.no_grad()` 아래에서 실행되며 weight를 update하지 않는다.

## 14. Checkpoint 및 History 흐름

각 epoch마다 `src/hs/train.py`는 아래 파일을 저장한다.

```text
outputs/<run_name>/history.json
outputs/<run_name>/last.pt
```

Validation AUROC가 개선되면 아래 파일도 저장한다.

```text
outputs/<run_name>/best.pt
```

각 checkpoint는 다음을 저장한다.

- `model`: model state dict
- `optimizer`: optimizer state dict
- `epoch`: epoch number
- `best_valid_auroc`: 해당 시점까지의 best validation AUROC
- `model_config`: `OutcomeModel.get_config()`에서 가져온 model construction parameter
- `args`: training CLI arguments

`model_config`를 저장하는 것은 중요하다. 특히 LoRA 같은 PEFT는 module structure를 바꾸기 때문에, evaluation에서 strict checkpoint loading을 하기 전에 같은 구조의 model을 다시 만들어야 한다.

## 15. Evaluation 흐름

### 15.1 Shell Script Layer

기본 script인 `scripts/eval_circor_outcome.sh`는 다음을 수행한다.

1. Repository root로 이동한다.
2. `.venv-m2d`를 activate한다.
3. `portable_m2d.py`를 patch한다.
4. `weights/m2d` 아래에서 첫 번째 M2D checkpoint를 찾는다.
5. 아래 인자로 `python -m hs.eval`을 실행한다.

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

PEFT evaluation script는 아래 configurable value를 추가로 사용한다.

- `RUN_DIR`
- `PATIENT_AGGREGATION`, `mean`, `max`, `noisy_or` 중 하나
- `PEFT_METHOD`와 PEFT parameter들, 단 checkpoint에 `model_config`가 없을 때만 사용

### 15.2 Python Evaluation Module

실제 evaluation logic은 `src/hs/eval.py`에 있다.

`main()`은 다음 순서로 실행된다.

```text
CLI args parse
    ↓
device = get_device()
    ↓
center crop으로 CirCorOutcomeDataset 생성
DataLoader 생성
    ↓
torch.load로 checkpoint load
    ↓
model_config resolve
    ↓
OutcomeModel 생성
model state dict를 strict하게 load
    ↓
recording-level probability 평가
    ↓
patient별 probability aggregation
    ↓
patient-level metric 계산
    ↓
JSON result 출력 및 저장
```

### 15.3 Model Config Resolution

`resolve_model_config`는 두 가지 checkpoint style을 처리한다.

- Checkpoint에 `model_config`가 있으면 그것을 사용하되, `m2d_weight_path`만 CLI 값으로 override한다.
- 없으면 CLI argument로 config를 다시 만든다.

이 방식 덕분에 한 machine에서 학습한 checkpoint를 M2D weight path가 다른 다른 machine에서도 평가할 수 있다.

### 15.4 Recording-Level Metrics

`evaluate_recording_level`은 다음을 수행한다.

1. 각 recording을 model에 통과시킨다.
2. `softmax(logits)[:, 1]`로 abnormal-class probability를 계산한다.
3. Label과 patient ID를 수집한다.
4. Threshold `0.5`를 사용해 metric을 계산한다.

Metric에는 다음이 포함된다.

- accuracy
- F1
- confusion matrix
- classification report
- AUROC

### 15.5 Patient-Level Metrics

Patient-level evaluation은 recording probability를 `patient_id` 기준으로 grouping한다.

지원되는 aggregation method:

- `mean`: recording probability 평균
- `max`: 최대 recording probability 사용
- `noisy_or`: recording 전체에 대해 `1 - product(1 - p)` 계산

Aggregation 이후 같은 binary metric을 patient level에서 계산한다.

## 16. Utility 및 Config Module

`src/hs/utils.py`는 다음을 제공한다.

- `set_seed`: Python `random`, NumPy, PyTorch CPU, PyTorch CUDA seed를 설정한다.
- `ensure_dir`: output directory를 만든다.
- `get_device`: CUDA가 가능하면 `"cuda"`, 아니면 `"cpu"`를 반환한다.

`src/hs/config.py`는 dataclass를 정의한다.

- `DataConfig`
- `TrainConfig`
- `ModelConfig`

이 dataclass들은 현재 `hs.train`이나 `hs.eval`에 연결되어 있지 않다. 활성화된 코드는 argparse를 사용한다.

`configs/circor_outcome.yaml`은 task default를 저장한다.

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

이 YAML 역시 현재 training code에서 load되지 않는다. 지금은 future 또는 reference configuration file로 이해하는 것이 좋다.

## 17. 관찰된 실험 로그

`outputs.txt`에는 이전 CirCor outcome experiment terminal log가 들어 있다.

여기에는 아래 실험들이 보인다.

- `PEFT_METHOD=none`
- `PEFT_METHOD=norm`
- `PEFT_METHOD=lora`
- `PEFT_METHOD=last_blocks`

Log는 의도된 trainability pattern을 확인해준다.

- `none`: encoder는 frozen이고 head만 trainable이다.
- `norm`: 소수의 encoder norm parameter와 bias parameter가 trainable이다.
- `lora`: 선택된 linear layer의 LoRA parameter가 trainable이다.
- `last_blocks`: 마지막 transformer block이 trainable이다.

또한 log에서 validation AUROC로 `best.pt`를 선택한다는 점을 확인할 수 있다.

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

## 19. 현재 Gap 및 실행 Blocker

현재 repository state에는 중요한 blocker가 몇 가지 있다.

1. `src/hs/data/circor.py`가 없다.
2. `src/hs/data/preprocess.py`가 없다.
3. 이 checkout의 `external/m2d`에는 source file이 들어 있지 않다.
4. Script는 `weights/m2d/*.pth`를 필요로 하지만 weight는 의도적으로 repository에 저장되어 있지 않다.
5. Manifest를 만들려면 `CIRCOR_ROOT` 아래 실제 CirCor data가 필요하다.
6. BMD-HS data loading, disease head, Method A, Method B, Method C는 연구 계획 항목이며 현재 구현된 code가 아니다.

따라서 이 문서는 의도된 실행 흐름과, 현재 checkout에서 그 흐름을 실제로 실행하기 위해 필요한 missing piece를 함께 설명한다.

## 20. 향후 BMD-HS 작업 확장 방향

더 넓은 연구 계획을 구현하려면 현재 구조를 대체하기보다 확장하는 방식이 좋다.

추천 extension point:

- `src/hs/data` 아래에 BMD-HS dataset 및 preprocessing module을 추가한다.
- `M2DWrapper`를 재사용하는 multi-label disease model 또는 head를 추가한다.
- Method A의 경우, CirCor-trained checkpoint의 encoder state를 load하여 BMD-HS training을 initialize한다.
- Method B의 경우, CirCor outcome용 head와 BMD-HS disease용 head를 가진 shared encoder model을 만든다.
- Method C의 경우, CirCor adaptation 없이 pretrained M2D/M2D-X에서 바로 BMD-HS disease model을 학습한다.
- Macro F1, micro F1, AUROC, AUPRC, class-wise metric을 위한 evaluation code를 추가한다.
- Medical audio practice에 맞게 patient-level split과 patient-level reporting을 기본으로 유지한다.

따라서 현재 CirCor pipeline은 향후 BMD-HS transfer 및 multi-task experiment가 기반으로 삼아야 할 source-task adaptation stage로 보는 것이 가장 적절하다.
