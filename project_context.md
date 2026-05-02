# Heart Sound Domain Adaptation Research Summary

## 1. Research Background

본 연구는 기존에 medical LLM/vision 연구를 수행하던 연구자가, 대학원 사이드 프로젝트 형태로 **audio AI 분야에 medical domain을 접목**하기 위해 시작한 연구이다. 초기에는 Qwen2-Audio-7B와 같은 audio-language model을 이용하여 heart sound classification 및 medical audio assistant를 구성하는 방향을 고려하였다. 그러나 연구 목적이 주로 **심음 분류 성능 향상**과 **저자원 heart sound dataset에서의 representation transfer**에 맞춰져 있다는 점을 고려하여, generative audio-language model보다는 **BEATs, M2D, M2D-X와 같은 general-purpose audio representation model**을 사용하는 방향으로 연구 방법을 수정하였다.

현재 연구의 핵심은 **Foundation Audio Model을 이용한 Heart Sound Domain Adaptation**이다. 심음 전용 foundation model이 충분히 확립되지 않은 상황에서, M2D-X와 같은 범용 오디오 representation을 heart sound domain에 적응시키고, 이를 통해 heart sound classification 성능을 개선하는 것을 목표로 한다.

---

## 2. Research Motivation

현재 heart sound classification 분야에는 CNN, Transformer, audio foundation model 기반 classifier 등 다양한 모델이 존재한다. 그러나 기존 연구들은 특정 데이터셋에 최적화된 supervised learning 방식에 의존하는 경우가 많고, 심음 데이터의 특성상 다음과 같은 한계가 존재한다.

1. **심음 데이터셋의 규모가 작음**
   - 의료 오디오 데이터는 수집이 어렵고, label annotation 비용이 높다.
   - 특히 fine-grained heart disease label을 가진 데이터셋은 제한적이다.

2. **데이터셋 간 이질성이 큼**
   - 데이터셋마다 recording environment, device, patient population, label definition이 다르다.
   - 하나의 데이터셋에서 학습한 모델이 다른 데이터셋으로 잘 일반화되지 않을 수 있다.

3. **심음 전용 foundation model이 부족함**
   - 이미지/텍스트 분야와 달리, heart sound domain에 특화된 large-scale pretrained foundation model은 아직 제한적이다.
   - 따라서 BEATs, M2D, M2D-X와 같은 범용 audio representation model을 medical audio domain에 적응시키는 전략이 필요하다.

4. **coarse label과 fine-grained disease label 간의 연결 가능성**
   - CirCor dataset은 normal/abnormal outcome과 같은 coarse-level supervision을 제공한다.
   - BMD-HS dataset은 Aortic Stenosis, Aortic Regurgitation, Mitral Regurgitation, Mitral Stenosis와 같은 fine-grained heart disease label을 제공한다.
   - 따라서 coarse abnormality detection에서 학습한 representation을 fine-grained disease classification으로 전이할 수 있는지 검증할 필요가 있다.

---

## 3. Research Goal

본 연구의 목표는 **M2D-X를 backbone으로 사용하여, 범용 audio representation을 heart sound domain에 적응시키고, 이를 통해 저자원 heart disease classification 성능을 향상시키는 것**이다.

보다 구체적으로는 다음과 같은 연구 질문을 다룬다.

1. **General-purpose audio foundation model은 heart sound domain에 효과적으로 적응될 수 있는가?**

2. **CirCor dataset의 coarse outcome supervision을 이용해 학습한 representation이 BMD-HS dataset의 fine-grained disease classification에 도움이 되는가?**

3. **Sequential transfer learning과 joint multi-task learning 중 어떤 전략이 heart sound domain adaptation에 더 효과적인가?**

4. **CirCor를 거치지 않고 BMD-HS만 직접 학습한 baseline과 비교했을 때, coarse-to-fine transfer learning이 성능 향상에 기여하는가?**

---

## 4. Core Research Idea

본 연구는 다음과 같은 가설에 기반한다.

> 심음 전용 foundation model이 부족한 상황에서, M2D-X와 같은 general-purpose audio representation model을 CirCor의 coarse-level normal/abnormal outcome task로 먼저 heart sound domain에 적응시키면, 이후 BMD-HS의 fine-grained heart disease classification task에서 더 좋은 성능을 얻을 수 있다.

즉, 본 연구의 핵심 아이디어는 다음과 같다.

- **Foundation Audio Model을 이용한 Heart Sound Domain Adaptation**
- **Coarse-level abnormality representation 학습**
- **Fine-grained heart disease classification으로의 transfer**
- **Sequential transfer learning과 multi-task learning 비교**
- **저자원 medical audio classification 성능 개선**

---

## 5. Datasets

### 5.1 CirCor Dataset

CirCor dataset은 heart sound classification 연구에서 널리 사용되는 PCG dataset이다. 본 연구에서는 CirCor dataset을 **coarse-level representation learning**을 위한 source dataset으로 사용한다.

#### 역할

CirCor dataset은 본 연구에서 다음 목적을 위해 사용된다.

- Heart sound domain adaptation
- outcome normal/abnormal binary classification
- coarse-level abnormality representation 학습

#### Target Task

본 연구에서는 CirCor dataset의 outcome label을 이용하여 다음 task를 수행한다.

```text
Input: heart sound recording
Output: outcome label
Classes: normal / abnormal
Task type: binary classification
```

#### 해석

CirCor의 outcome label은 심음 자체의 단순 acoustic abnormality라기보다는, 임상적 평가와 연관된 normal/abnormal label로 볼 수 있다. 따라서 본 연구에서는 이를 **coarse clinical abnormality supervision**으로 해석한다.

---

### 5.2 BMD-HS Dataset

BMD-HS는 BUET Multi-disease Heart Sound Dataset으로, heart disease multi-label classification을 위한 target dataset으로 사용한다.

#### 역할

BMD-HS dataset은 본 연구에서 다음 목적을 위해 사용된다.

- fine-grained heart disease classification
- low-resource target task
- CirCor에서 학습한 coarse representation의 transfer 효과 검증

#### Target Task

본 연구에서는 BMD-HS dataset을 이용하여 다음 disease classification task를 수행한다.

```text
Input: heart sound recording
Output: heart disease label
Target diseases:
- Aortic Stenosis
- Aortic Regurgitation
- Mitral Regurgitation
- Mitral Stenosis
Task type: multi-label classification
```

#### 주의할 점

BMD-HS dataset에는 Normal, Multi Disease, 기타 label이 포함될 수 있으므로, 실제 실험 전에 다음 사항을 명확히 정의해야 한다.

- AS, AR, MR, MS 네 가지 disease만 target으로 사용할지
- Normal sample을 negative class로 포함할지
- Multi Disease label을 어떻게 처리할지
- patient-level split을 사용할지, recording-level split을 사용할지
- label imbalance를 어떻게 보정할지

---

## 6. Backbone Model: M2D-X

본 연구에서는 **M2D-X**를 backbone model로 사용한다.

M2D-X는 general-purpose audio representation learning framework인 M2D 계열의 확장으로, 범용 오디오 representation을 다양한 downstream task에 적응시키기 위한 backbone으로 사용할 수 있다.

본 연구에서 M2D-X를 선택한 이유는 다음과 같다.

1. **Audio representation learning에 적합함**
   - Qwen2-Audio와 달리, generative audio-language response가 아니라 classification backbone으로 활용하기 적합하다.

2. **심음 분류 task와 목적이 잘 맞음**
   - 본 연구의 핵심 목표는 text generation이 아니라 heart sound classification 성능 개선이다.

3. **도메인 적응 연구에 적합함**
   - general audio representation을 heart sound domain에 fine-tuning하는 구조가 자연스럽다.

4. **BEATs, M2D 등과 비교 가능한 audio foundation model 계열임**
   - 향후 baseline 비교 및 ablation study 구성에 유리하다.

---

## 7. Initial Direction and Revision

### 7.1 Initial Direction: Qwen2-Audio

초기 아이디어는 다음과 같았다.

1. Qwen2-Audio-7B를 CirCor dataset으로 abnormal/normal heart sound classification SFT 수행
2. 해당 checkpoint를 BMD-HS dataset으로 추가 SFT하여 heart disease multi-label classification 수행

그러나 이 방향에는 다음과 같은 문제가 있었다.

- Qwen2-Audio는 audio-language model로, 강점은 audio instruction following과 text generation에 있음
- 본 연구의 핵심 task는 text generation이 아니라 classification임
- Qwen2-Audio를 classifier처럼만 사용하는 것은 모델의 장점을 충분히 활용하지 못함
- 모델 크기가 커서 실험 비용과 복잡도가 커짐
- 연구 메시지가 “큰 모델을 썼다” 수준으로 약해질 수 있음

따라서 Qwen2-Audio 대신 M2D-X와 같은 audio representation backbone을 사용하는 방향으로 수정하였다.

---

### 7.2 Revised Direction: M2D-X 기반 Heart Sound Domain Adaptation

수정된 방향은 다음과 같다.

> Qwen2-Audio를 이용한 medical audio assistant가 아니라, M2D-X를 이용한 heart sound domain adaptation 및 coarse-to-fine transfer learning을 수행한다.

이 방향에서는 연구 목적이 더 명확해진다.

- general-purpose audio representation을 heart sound domain에 적응
- CirCor의 coarse normal/abnormal supervision을 이용한 representation learning
- BMD-HS의 fine-grained heart disease classification으로 transfer
- sequential transfer learning과 multi-task learning의 효과 비교

---

## 8. Proposed Methodology

본 연구에서는 M2D-X encoder를 backbone으로 사용하고, 세 가지 학습 전략을 비교한다.

---

### 8.1 Method A: Sequential Transfer Learning

#### 개요

Method A는 CirCor dataset으로 먼저 coarse-level representation을 학습한 뒤, BMD-HS dataset으로 fine-grained disease classification을 수행하는 sequential transfer learning 방식이다.

#### 학습 과정

```text
Step 1:
M2D-X encoder → CirCor outcome binary classification

Result:
- Encoder weight: A
- Outcome prediction head: A_h

Step 2:
Encoder A → BMD-HS heart disease multi-label classification

Result:
- Encoder weight: B
- Disease prediction head: B_h
```

#### 설명

먼저 CirCor dataset의 outcome normal/abnormal binary classification을 학습하여 심음의 전반적인 abnormality와 관련된 coarse-level representation을 학습한다. 이후 학습된 encoder A를 초기값으로 사용하여 BMD-HS dataset에서 AS, AR, MR, MS와 같은 fine-grained heart disease classification을 수행한다.

#### 목적

Method A의 목적은 다음과 같다.

- CirCor에서 학습한 coarse abnormality representation이 BMD-HS의 fine-grained disease classification에 도움이 되는지 검증
- coarse-to-fine transfer learning의 효과 확인
- source dataset의 supervision을 target dataset으로 전이

#### 핵심 가설

> CirCor outcome task에서 학습한 coarse representation은 BMD-HS의 fine-grained heart disease classification 성능을 향상시킬 것이다.

---

### 8.2 Method B: Joint Multi-task Learning

#### 개요

Method B는 CirCor와 BMD-HS를 함께 사용하여, 공유 encoder 위에 서로 다른 task-specific head를 두고 multi-task learning을 수행하는 방식이다.

#### 모델 구조

```text
                         ┌── Outcome Head A_h
                         │   CirCor normal/abnormal
Input heart sound → Shared Encoder
                         │
                         └── Disease Head B_h
                             BMD-HS multi-label disease
```

#### 학습 과정

```text
Input batch 1: CirCor batch
- Forward through shared encoder
- Use outcome head
- Compute binary classification loss

Input batch 2: BMD-HS batch
- Forward through shared encoder
- Use disease head
- Compute multi-label classification loss

Training:
- Alternate CirCor and BMD-HS batches
- Jointly update shared encoder
- Update corresponding task-specific head
```

#### 설명

CirCor batch가 입력되면 outcome head를 이용하여 normal/abnormal binary classification loss를 계산한다. BMD-HS batch가 입력되면 disease head를 이용하여 multi-label disease classification loss를 계산한다. 두 task는 encoder를 공유하고, 각 task는 별도의 head를 사용한다.

#### 목적

Method B의 목적은 다음과 같다.

- target disease classification 학습 중에도 coarse outcome supervision을 유지
- source task가 target task 학습에 regularization 역할을 하는지 검증
- sequential transfer보다 joint multi-task learning이 더 효과적인지 확인

#### 핵심 가설

> CirCor outcome supervision을 BMD-HS disease learning과 함께 유지하면, shared encoder가 더 일반화된 heart sound representation을 학습할 수 있다.

#### 주의할 점

Method B는 반드시 더 좋은 방법이라고 가정하면 안 된다. 두 데이터셋은 label space와 task definition이 다르기 때문에 negative transfer가 발생할 수 있다. 따라서 Method B는 다음 질문을 검증하기 위한 실험적 방법으로 설정해야 한다.

> Coarse clinical outcome supervision을 target disease training 중에도 유지하는 것이 BMD-HS disease classification에 도움이 되는가?

---

### 8.3 Method C: Direct Fine-tuning Baseline

#### 개요

Method C는 CirCor dataset을 사용하지 않고, M2D-X encoder를 바로 BMD-HS dataset에 fine-tuning하는 baseline 방식이다.

#### 학습 과정

```text
M2D-X encoder → BMD-HS heart disease multi-label classification
```

#### 설명

Method C에서는 CirCor를 통한 coarse-level representation learning을 수행하지 않는다. M2D-X encoder를 직접 BMD-HS dataset에 fine-tuning하여 heart disease classification을 수행한다.

#### 목적

Method C는 A, B 방법의 효과를 평가하기 위한 baseline이다.

#### 핵심 가설

> 만약 Method A 또는 B가 Method C보다 높은 성능을 보인다면, CirCor를 이용한 heart sound domain adaptation 또는 coarse-to-fine transfer가 효과적이라고 해석할 수 있다.

---

## 9. Comparison Strategy

본 연구에서는 Method A, B, C를 비교하여 각 학습 전략의 효과를 분석한다.

| Method | Description | Purpose |
|---|---|---|
| A | CirCor outcome 학습 후 BMD-HS disease classification | Sequential coarse-to-fine transfer |
| B | CirCor와 BMD-HS를 함께 multi-task learning | Joint supervision / regularization |
| C | CirCor 없이 BMD-HS만 직접 fine-tuning | Direct fine-tuning baseline |

---

## 10. Evaluation Plan

### 10.1 Main Evaluation

주요 평가는 BMD-HS dataset의 heart disease multi-label classification 성능을 기준으로 수행한다.

#### Main target

```text
BMD-HS disease classification performance
```

#### Evaluation metrics

다음 지표를 사용할 수 있다.

- Accuracy
- Macro F1-score
- Micro F1-score
- AUROC
- AUPRC
- Class-wise F1-score
- Class-wise AUROC

BMD-HS는 class imbalance가 존재할 수 있으므로, 단순 accuracy만으로 평가하는 것은 부족하다. 따라서 macro F1, class-wise F1, AUROC, AUPRC 등을 함께 보는 것이 적절하다.

---

### 10.2 Secondary Evaluation

보조적으로 CirCor outcome classification 성능도 확인할 수 있다.

#### 목적

- Method B가 CirCor outcome 성능을 유지하는지 확인
- Multi-task learning이 shared encoder의 generality를 유지하는지 분석
- Method A에서 BMD-HS fine-tuning 후 CirCor 성능이 얼마나 감소하는지 확인하여 catastrophic forgetting 여부 분석

#### Possible analysis

```text
A가 BMD-HS에서는 좋지만 CirCor 성능이 크게 떨어짐:
→ target specialization은 잘 되었지만 source task는 forgetting됨

B가 BMD-HS와 CirCor 모두에서 안정적:
→ multi-task learning이 representation generality 유지에 도움

B가 CirCor는 좋지만 BMD-HS가 낮음:
→ source task가 target disease classification을 방해했을 가능성
```

---

### 10.3 Interpretation of Results

#### Case 1: A > C

```text
CirCor outcome pre-adaptation이 BMD-HS disease classification에 도움이 됨
```

해석:

- coarse-to-fine transfer learning이 유효함
- CirCor에서 학습한 abnormality representation이 BMD-HS disease classification으로 전이됨

---

#### Case 2: B > A

```text
Joint multi-task learning이 sequential transfer보다 효과적임
```

해석:

- target task 학습 중에도 CirCor outcome supervision을 유지하는 것이 도움이 됨
- coarse supervision이 regularization 역할을 수행했을 가능성

---

#### Case 3: A > B

```text
Sequential transfer가 multi-task learning보다 효과적임
```

해석:

- BMD-HS target task에 집중하는 것이 더 중요함
- CirCor와 BMD-HS의 label space 차이로 인해 multi-task learning에서 negative transfer가 발생했을 수 있음

---

#### Case 4: C > A, B

```text
CirCor 기반 pre-adaptation 또는 multi-task learning이 도움이 되지 않음
```

해석:

- CirCor outcome label과 BMD-HS disease label 간의 task mismatch가 클 수 있음
- source dataset의 noisy label 또는 domain difference가 target disease classification을 방해했을 가능성
- BMD-HS target task에 직접 fine-tuning하는 것이 더 적합할 수 있음

---

## 11. Expected Contributions

본 연구의 기대 기여는 다음과 같다.

1. **Heart sound domain adaptation strategy 제안**
   - M2D-X와 같은 general-purpose audio foundation model을 heart sound domain에 적응시키는 방법을 제안한다.

2. **Coarse-to-fine transfer learning 검증**
   - CirCor의 normal/abnormal outcome supervision이 BMD-HS의 fine-grained heart disease classification에 도움이 되는지 분석한다.

3. **Sequential transfer와 multi-task learning 비교**
   - source task를 pre-adaptation으로만 사용할지, target task 학습 중에도 함께 유지할지 비교한다.

4. **저자원 heart disease classification 성능 개선 가능성 탐색**
   - 작은 규모의 BMD-HS dataset에서 general audio representation을 활용해 성능을 향상시킬 수 있는지 검증한다.

5. **심음 전용 foundation model 부족 문제에 대한 대안 제시**
   - 별도의 large-scale heart sound foundation model 없이도, general audio representation model을 활용하는 practical strategy를 제시한다.

---

## 12. Proposed PPT Structure

본 연구 내용을 발표하기 위한 PPT는 다음 흐름으로 구성할 수 있다.

### Slide 1. Title

#### Title

```text
Heart Sound Domain Adaptation using General-Purpose Audio Representation
```

#### Subtitle

```text
Coarse-to-Fine Transfer Learning with CirCor and BMD-HS
```

---

### Slide 2. Research Goal

#### Key message

본 연구는 M2D-X를 이용하여 범용 오디오 representation을 heart sound domain에 적응시키고, 이를 통해 저자원 heart disease classification 성능을 향상시키는 것을 목표로 한다.

#### Bullet points

- Heart sound-specific foundation models are still limited.
- General-purpose audio representation models can be adapted to medical audio.
- We aim to improve heart disease classification through heart sound domain adaptation.
- We investigate coarse-to-fine transfer learning from CirCor to BMD-HS.

---

### Slide 3. Related Work

#### Key message

기존 heart sound classifier는 많지만, 데이터셋 특화 학습, 저자원 문제, domain shift 문제가 여전히 존재한다.

#### Bullet points

- Heart sound classification has been studied with CNN, RNN, Transformer, and audio foundation models.
- General audio representation models such as BEATs and M2D have shown potential in bioacoustic and medical audio tasks.
- However, heart sound datasets are small, noisy, and heterogeneous.
- Domain adaptation and transfer learning strategies are needed.

---

### Slide 4. Proposed Framework

#### Key message

M2D-X를 backbone으로 사용하고, CirCor와 BMD-HS를 이용해 세 가지 학습 전략을 비교한다.

#### Figure idea

```text
M2D-X Encoder
      │
      ├── Method A: CirCor → BMD-HS
      ├── Method B: CirCor + BMD-HS Multi-task
      └── Method C: BMD-HS only
```

---

### Slide 5. Method A: Sequential Transfer

#### Key message

CirCor에서 coarse abnormality representation을 학습한 뒤, BMD-HS의 fine-grained disease classification으로 전이한다.

#### Bullet points

- Step 1: Fine-tune M2D-X on CirCor outcome classification.
- Step 2: Use the adapted encoder as initialization for BMD-HS disease classification.
- Goal: Transfer coarse abnormality representation to fine-grained disease labels.

---

### Slide 6. Method B: Joint Multi-task Learning

#### Key message

CirCor와 BMD-HS를 함께 학습하여 coarse supervision이 target disease learning에 regularization 역할을 하는지 확인한다.

#### Bullet points

- Shared encoder with two task-specific heads.
- Outcome head for CirCor normal/abnormal classification.
- Disease head for BMD-HS multi-label classification.
- Alternating batches from CirCor and BMD-HS.
- Goal: Jointly optimize coarse and fine-grained supervision.

---

### Slide 7. Method C: Direct Fine-tuning Baseline

#### Key message

CirCor를 사용하지 않고 BMD-HS만으로 학습하여, coarse adaptation의 효과를 평가하기 위한 baseline을 구성한다.

#### Bullet points

- Directly fine-tune M2D-X on BMD-HS.
- No CirCor pre-adaptation.
- Used as the baseline for evaluating A and B.

---

### Slide 8. Evaluation Plan

#### Key message

A, B, C 방법론의 BMD-HS disease classification 성능을 비교하여 coarse-to-fine transfer와 multi-task learning의 효과를 분석한다.

#### Bullet points

- Main evaluation: BMD-HS disease classification.
- Metrics: Macro F1, Micro F1, AUROC, AUPRC, class-wise F1.
- Compare A vs C to evaluate the effect of CirCor pre-adaptation.
- Compare B vs A to evaluate the effect of joint multi-task learning.
- Analyze whether coarse supervision improves fine-grained disease classification.

---

### Slide 9. Expected Contribution

#### Key message

본 연구는 general-purpose audio representation을 활용한 heart sound domain adaptation 전략을 제안하고, coarse-to-fine transfer learning의 효과를 검증한다.

#### Bullet points

- Heart sound domain adaptation using M2D-X.
- Coarse-to-fine transfer from CirCor to BMD-HS.
- Comparison of sequential transfer and joint multi-task learning.
- Practical strategy for low-resource heart disease classification.

---

## 13. One-paragraph Summary

본 연구는 심음 전용 foundation model이 부족한 상황에서, M2D-X와 같은 general-purpose audio representation model을 heart sound domain에 적응시켜 저자원 heart disease classification 성능을 향상시키는 것을 목표로 한다. 먼저 CirCor dataset의 outcome normal/abnormal binary classification을 통해 심음의 전반적인 abnormality와 관련된 coarse-level representation을 학습한다. 이후 BMD-HS dataset의 AS, AR, MR, MS 등 fine-grained heart disease classification으로 전이하는 sequential transfer learning 전략과, CirCor와 BMD-HS를 동시에 사용하는 joint multi-task learning 전략을 비교한다. 또한 CirCor를 사용하지 않고 BMD-HS만 직접 fine-tuning하는 baseline을 함께 평가하여, coarse-to-fine transfer와 multi-task learning이 target disease classification 성능에 미치는 영향을 분석한다.

---

## 14. Recommended Terminology

본 연구에서 사용할 주요 용어는 다음과 같이 정리할 수 있다.

| Term | Meaning |
|---|---|
| Foundation Audio Model | 대규모 오디오 데이터로 사전학습된 범용 오디오 representation model |
| Heart Sound Domain Adaptation | 범용 오디오 모델을 심음 데이터에 맞게 적응시키는 과정 |
| Coarse-level Representation | normal/abnormal outcome처럼 큰 범주의 이상 여부를 포착하는 표현 |
| Fine-grained Disease Classification | AS, AR, MR, MS 등 세부 심장질환을 분류하는 task |
| Sequential Transfer Learning | source task 학습 후 target task로 순차적으로 전이하는 학습 방식 |
| Joint Multi-task Learning | 여러 task를 공유 encoder로 동시에 학습하는 방식 |
| Source Dataset | CirCor |
| Target Dataset | BMD-HS |
| Source Task | CirCor outcome normal/abnormal binary classification |
| Target Task | BMD-HS heart disease multi-label classification |

---

## 15. Current Research Direction

현재 연구 방향은 다음과 같이 정리된다.

```text
Qwen2-Audio 기반 medical audio assistant
        ↓
M2D-X 기반 heart sound domain adaptation
        ↓
CirCor outcome task를 이용한 coarse representation 학습
        ↓
BMD-HS disease task로 fine-grained transfer
        ↓
Sequential transfer vs joint multi-task learning 비교
```

최종적으로 본 연구는 **“또 하나의 heart sound classifier를 만드는 것”**이 아니라, **범용 audio foundation model을 heart sound domain에 어떻게 효과적으로 적응시킬 수 있는지**를 분석하는 연구로 정의된다.
