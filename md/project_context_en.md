# Heart Sound Domain Adaptation Research Summary

## 1. Research Background

This research was initiated as a graduate-level side project by a researcher with prior experience in medical LLM and medical vision research, with the goal of incorporating the medical domain into audio AI. The initial direction considered using an audio-language model such as Qwen2-Audio-7B for heart sound classification and building a medical audio assistant. However, since the main research objective is focused on **improving heart sound classification performance** and **transferring representations in low-resource heart sound datasets**, the research direction was revised to use **general-purpose audio representation models such as BEATs, M2D, and M2D-X**, rather than generative audio-language models.

The core of the current research is **Heart Sound Domain Adaptation using a Foundation Audio Model**. In a setting where heart sound-specific foundation models are not yet sufficiently established, this study aims to adapt a general-purpose audio representation model such as M2D-X to the heart sound domain and improve heart sound classification performance.

---

## 2. Research Motivation

There are currently various models for heart sound classification, including CNN-based classifiers, Transformer-based models, and audio foundation model-based classifiers. However, many existing studies rely on supervised learning optimized for specific datasets, and heart sound data have the following limitations.

1. **Small scale of heart sound datasets**
   - Medical audio data are difficult to collect, and label annotation is costly.
   - Datasets with fine-grained heart disease labels are especially limited.

2. **High heterogeneity across datasets**
   - Each dataset differs in recording environment, device, patient population, and label definition.
   - A model trained on one dataset may not generalize well to another dataset.

3. **Lack of heart sound-specific foundation models**
   - Unlike the image and text domains, large-scale pretrained foundation models specialized for the heart sound domain are still limited.
   - Therefore, a strategy is needed to adapt general-purpose audio representation models such as BEATs, M2D, and M2D-X to the medical audio domain.

4. **Potential connection between coarse labels and fine-grained disease labels**
   - The CirCor dataset provides coarse-level supervision such as normal/abnormal outcomes.
   - The BMD-HS dataset provides fine-grained heart disease labels such as Aortic Stenosis, Aortic Regurgitation, Mitral Regurgitation, and Mitral Stenosis.
   - Therefore, it is necessary to investigate whether representations learned from coarse abnormality detection can be transferred to fine-grained disease classification.

---

## 3. Research Goal

The goal of this study is to **use M2D-X as the backbone, adapt a general-purpose audio representation to the heart sound domain, and improve low-resource heart disease classification performance**.

More specifically, this study addresses the following research questions.

1. **Can a general-purpose audio foundation model be effectively adapted to the heart sound domain?**

2. **Does the representation learned using coarse outcome supervision from the CirCor dataset help fine-grained disease classification on the BMD-HS dataset?**

3. **Which strategy is more effective for heart sound domain adaptation: sequential transfer learning or joint multi-task learning?**

4. **Compared with a baseline that directly trains on BMD-HS without using CirCor, does coarse-to-fine transfer learning contribute to performance improvement?**

---

## 4. Core Research Idea

This study is based on the following hypothesis.

> In a setting where heart sound-specific foundation models are limited, adapting a general-purpose audio representation model such as M2D-X to the heart sound domain using the coarse-level normal/abnormal outcome task from CirCor will lead to better performance on the fine-grained heart disease classification task in BMD-HS.

In other words, the core ideas of this study are as follows.

- **Heart Sound Domain Adaptation using a Foundation Audio Model**
- **Learning coarse-level abnormality representations**
- **Transfer to fine-grained heart disease classification**
- **Comparison between sequential transfer learning and multi-task learning**
- **Improvement of low-resource medical audio classification performance**

---

## 5. Datasets

### 5.1 CirCor Dataset

The CirCor dataset is a PCG dataset widely used in heart sound classification research. In this study, the CirCor dataset is used as the source dataset for **coarse-level representation learning**.

#### Role

The CirCor dataset is used for the following purposes in this study.

- Heart sound domain adaptation
- Outcome normal/abnormal binary classification
- Learning coarse-level abnormality representations

#### Target Task

In this study, the outcome label from the CirCor dataset is used for the following task.

```text
Input: heart sound recording
Output: outcome label
Classes: normal / abnormal
Task type: binary classification
```

#### Interpretation

The outcome label in CirCor can be interpreted not simply as an acoustic abnormality of the heart sound itself, but as a normal/abnormal label associated with clinical evaluation. Therefore, in this study, it is interpreted as **coarse clinical abnormality supervision**.

---

### 5.2 BMD-HS Dataset

BMD-HS stands for the BUET Multi-disease Heart Sound Dataset and is used as the target dataset for heart disease multi-label classification.

#### Role

The BMD-HS dataset is used for the following purposes in this study.

- Fine-grained heart disease classification
- Low-resource target task
- Evaluation of the transfer effect of the coarse representation learned from CirCor

#### Target Task

In this study, the BMD-HS dataset is used for the following disease classification task.

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

#### Points to Clarify

Since the BMD-HS dataset may include Normal, Multi Disease, and other labels, the following points must be clearly defined before conducting experiments.

- Whether to use only the four diseases AS, AR, MR, and MS as target labels
- Whether to include normal samples as negative classes
- How to handle the Multi Disease label
- Whether to use a patient-level split or a recording-level split
- How to handle label imbalance

---

## 6. Backbone Model: M2D-X

This study uses **M2D-X** as the backbone model.

M2D-X is an extension of the M2D family, which is a general-purpose audio representation learning framework. It can be used as a backbone for adapting general audio representations to various downstream tasks.

The reasons for selecting M2D-X in this study are as follows.

1. **It is suitable for audio representation learning**
   - Unlike Qwen2-Audio, it is more suitable for use as a classification backbone rather than for generative audio-language responses.

2. **It matches the objective of the heart sound classification task**
   - The main goal of this study is not text generation, but improvement of heart sound classification performance.

3. **It is suitable for domain adaptation research**
   - The structure of fine-tuning a general audio representation to the heart sound domain is natural.

4. **It belongs to the same family of audio foundation models that can be compared with BEATs and M2D**
   - This is advantageous for future baseline comparisons and ablation studies.

---

## 7. Initial Direction and Revision

### 7.1 Initial Direction: Qwen2-Audio

The initial idea was as follows.

1. Perform SFT of Qwen2-Audio-7B on the CirCor dataset for abnormal/normal heart sound classification
2. Further fine-tune the resulting checkpoint on the BMD-HS dataset for heart disease multi-label classification

However, this direction had the following limitations.

- Qwen2-Audio is an audio-language model whose strengths lie in audio instruction following and text generation.
- The core task of this study is classification, not text generation.
- Using Qwen2-Audio merely as a classifier does not fully utilize the strengths of the model.
- The model size increases experimental cost and complexity.
- The research message may become weak, reducing to simply “using a large model.”

Therefore, the direction was revised to use an audio representation backbone such as M2D-X instead of Qwen2-Audio.

---

### 7.2 Revised Direction: Heart Sound Domain Adaptation based on M2D-X

The revised direction is as follows.

> Instead of building a medical audio assistant using Qwen2-Audio, this study performs heart sound domain adaptation and coarse-to-fine transfer learning using M2D-X.

This direction makes the research objective clearer.

- Adapt general-purpose audio representations to the heart sound domain
- Learn representations using coarse normal/abnormal supervision from CirCor
- Transfer to fine-grained heart disease classification in BMD-HS
- Compare the effects of sequential transfer learning and multi-task learning

---

## 8. Proposed Methodology

This study uses the M2D-X encoder as the backbone and compares three training strategies.

---

### 8.1 Method A: Sequential Transfer Learning

#### Overview

Method A is a sequential transfer learning strategy in which the model first learns coarse-level representations using the CirCor dataset and then performs fine-grained disease classification using the BMD-HS dataset.

#### Training Process

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

#### Description

First, the model learns coarse-level representations related to overall abnormality in heart sounds by training on the normal/abnormal outcome binary classification task of the CirCor dataset. Then, the learned encoder A is used as the initialization for fine-grained heart disease classification on the BMD-HS dataset, including AS, AR, MR, and MS.

#### Purpose

The purposes of Method A are as follows.

- To verify whether the coarse abnormality representation learned from CirCor helps fine-grained disease classification on BMD-HS
- To evaluate the effect of coarse-to-fine transfer learning
- To transfer supervision from the source dataset to the target dataset

#### Core Hypothesis

> The coarse representation learned from the CirCor outcome task will improve fine-grained heart disease classification performance on BMD-HS.

---

### 8.2 Method B: Joint Multi-task Learning

#### Overview

Method B uses CirCor and BMD-HS together and performs multi-task learning with a shared encoder and separate task-specific heads.

#### Model Structure

```text
                         ┌── Outcome Head A_h
                         │   CirCor normal/abnormal
Input heart sound → Shared Encoder
                         │
                         └── Disease Head B_h
                             BMD-HS multi-label disease
```

#### Training Process

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

#### Description

When a CirCor batch is used as input, the model computes the normal/abnormal binary classification loss using the outcome head. When a BMD-HS batch is used as input, the model computes the multi-label disease classification loss using the disease head. The two tasks share the encoder, while each task uses its own separate head.

#### Purpose

The purposes of Method B are as follows.

- To maintain coarse outcome supervision during target disease classification training
- To verify whether the source task acts as a regularizer for the target task
- To examine whether joint multi-task learning is more effective than sequential transfer learning

#### Core Hypothesis

> If CirCor outcome supervision is maintained together with BMD-HS disease learning, the shared encoder can learn more generalizable heart sound representations.

#### Points to Note

Method B should not be assumed to be necessarily better. Since the two datasets have different label spaces and task definitions, negative transfer may occur. Therefore, Method B should be positioned as an experimental method for verifying the following question.

> Does maintaining coarse clinical outcome supervision during target disease training help BMD-HS disease classification?

---

### 8.3 Method C: Direct Fine-tuning Baseline

#### Overview

Method C is a baseline strategy in which the M2D-X encoder is directly fine-tuned on the BMD-HS dataset without using the CirCor dataset.

#### Training Process

```text
M2D-X encoder → BMD-HS heart disease multi-label classification
```

#### Description

In Method C, coarse-level representation learning through CirCor is not performed. The M2D-X encoder is directly fine-tuned on the BMD-HS dataset for heart disease classification.

#### Purpose

Method C serves as the baseline for evaluating the effects of Methods A and B.

#### Core Hypothesis

> If Method A or B outperforms Method C, it can be interpreted that heart sound domain adaptation or coarse-to-fine transfer using CirCor is effective.

---

## 9. Comparison Strategy

This study compares Methods A, B, and C to analyze the effects of each training strategy.

| Method | Description | Purpose |
|---|---|---|
| A | Train on CirCor outcome, then perform BMD-HS disease classification | Sequential coarse-to-fine transfer |
| B | Perform multi-task learning using CirCor and BMD-HS together | Joint supervision / regularization |
| C | Directly fine-tune only on BMD-HS without CirCor | Direct fine-tuning baseline |

---

## 10. Evaluation Plan

### 10.1 Main Evaluation

The main evaluation is based on the heart disease multi-label classification performance on the BMD-HS dataset.

#### Main Target

```text
BMD-HS disease classification performance
```

#### Evaluation Metrics

The following metrics can be used.

- Accuracy
- Macro F1-score
- Micro F1-score
- AUROC
- AUPRC
- Class-wise F1-score
- Class-wise AUROC

Since the BMD-HS dataset may have class imbalance, accuracy alone is insufficient for evaluation. Therefore, macro F1, class-wise F1, AUROC, and AUPRC should be considered together.

---

### 10.2 Secondary Evaluation

As a secondary evaluation, CirCor outcome classification performance can also be examined.

#### Purpose

- To check whether Method B maintains CirCor outcome performance
- To analyze whether multi-task learning helps maintain the generality of the shared encoder
- To examine how much CirCor performance decreases after BMD-HS fine-tuning in Method A, thereby analyzing catastrophic forgetting

#### Possible Analysis

```text
A performs well on BMD-HS but shows a large drop in CirCor performance:
→ Target specialization is successful, but the source task is forgotten.

B performs stably on both BMD-HS and CirCor:
→ Multi-task learning helps maintain representation generality.

B performs well on CirCor but poorly on BMD-HS:
→ The source task may have interfered with target disease classification.
```

---

### 10.3 Interpretation of Results

#### Case 1: A > C

```text
CirCor outcome pre-adaptation helps BMD-HS disease classification.
```

Interpretation:

- Coarse-to-fine transfer learning is effective.
- The abnormality representation learned from CirCor is transferred to BMD-HS disease classification.

---

#### Case 2: B > A

```text
Joint multi-task learning is more effective than sequential transfer.
```

Interpretation:

- Maintaining CirCor outcome supervision during target task training is helpful.
- Coarse supervision may have acted as a regularizer.

---

#### Case 3: A > B

```text
Sequential transfer is more effective than multi-task learning.
```

Interpretation:

- Focusing on the BMD-HS target task is more important.
- Negative transfer may have occurred in multi-task learning due to differences in label space between CirCor and BMD-HS.

---

#### Case 4: C > A, B

```text
CirCor-based pre-adaptation or multi-task learning does not help.
```

Interpretation:

- The task mismatch between the CirCor outcome label and BMD-HS disease labels may be large.
- Noisy source labels or domain differences may have interfered with target disease classification.
- Direct fine-tuning on the BMD-HS target task may be more appropriate.

---

## 11. Expected Contributions

The expected contributions of this study are as follows.

1. **Proposal of a heart sound domain adaptation strategy**
   - This study proposes a method for adapting a general-purpose audio foundation model such as M2D-X to the heart sound domain.

2. **Validation of coarse-to-fine transfer learning**
   - This study analyzes whether normal/abnormal outcome supervision from CirCor helps fine-grained heart disease classification on BMD-HS.

3. **Comparison between sequential transfer and multi-task learning**
   - This study compares whether the source task should be used only for pre-adaptation or maintained during target task training.

4. **Exploration of performance improvement in low-resource heart disease classification**
   - This study investigates whether general audio representations can improve performance on the small-scale BMD-HS dataset.

5. **A practical alternative to the lack of heart sound-specific foundation models**
   - This study presents a practical strategy that uses general audio representation models without requiring a separate large-scale heart sound foundation model.

---

## 12. Proposed PPT Structure

The PPT for presenting this research can be organized as follows.

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

#### Key Message

This study aims to adapt a general-purpose audio representation to the heart sound domain using M2D-X and improve low-resource heart disease classification performance.

#### Bullet Points

- Heart sound-specific foundation models are still limited.
- General-purpose audio representation models can be adapted to medical audio.
- We aim to improve heart disease classification through heart sound domain adaptation.
- We investigate coarse-to-fine transfer learning from CirCor to BMD-HS.

---

### Slide 3. Related Work

#### Key Message

Although many heart sound classifiers exist, dataset-specific training, low-resource data, and domain shift remain major challenges.

#### Bullet Points

- Heart sound classification has been studied with CNN, RNN, Transformer, and audio foundation models.
- General audio representation models such as BEATs and M2D have shown potential in bioacoustic and medical audio tasks.
- However, heart sound datasets are small, noisy, and heterogeneous.
- Domain adaptation and transfer learning strategies are needed.

---

### Slide 4. Proposed Framework

#### Key Message

This study uses M2D-X as the backbone and compares three training strategies using CirCor and BMD-HS.

#### Figure Idea

```text
M2D-X Encoder
      │
      ├── Method A: CirCor → BMD-HS
      ├── Method B: CirCor + BMD-HS Multi-task
      └── Method C: BMD-HS only
```

---

### Slide 5. Method A: Sequential Transfer

#### Key Message

After learning coarse abnormality representations from CirCor, the model transfers them to fine-grained disease classification on BMD-HS.

#### Bullet Points

- Step 1: Fine-tune M2D-X on CirCor outcome classification.
- Step 2: Use the adapted encoder as initialization for BMD-HS disease classification.
- Goal: Transfer coarse abnormality representation to fine-grained disease labels.

---

### Slide 6. Method B: Joint Multi-task Learning

#### Key Message

This method jointly trains on CirCor and BMD-HS to examine whether coarse supervision acts as a regularizer for target disease learning.

#### Bullet Points

- Shared encoder with two task-specific heads.
- Outcome head for CirCor normal/abnormal classification.
- Disease head for BMD-HS multi-label classification.
- Alternating batches from CirCor and BMD-HS.
- Goal: Jointly optimize coarse and fine-grained supervision.

---

### Slide 7. Method C: Direct Fine-tuning Baseline

#### Key Message

This method trains only on BMD-HS without using CirCor and serves as the baseline for evaluating the effect of coarse adaptation.

#### Bullet Points

- Directly fine-tune M2D-X on BMD-HS.
- No CirCor pre-adaptation.
- Used as the baseline for evaluating A and B.

---

### Slide 8. Evaluation Plan

#### Key Message

The performance of Methods A, B, and C is compared on BMD-HS disease classification to analyze the effects of coarse-to-fine transfer and multi-task learning.

#### Bullet Points

- Main evaluation: BMD-HS disease classification.
- Metrics: Macro F1, Micro F1, AUROC, AUPRC, class-wise F1.
- Compare A vs C to evaluate the effect of CirCor pre-adaptation.
- Compare B vs A to evaluate the effect of joint multi-task learning.
- Analyze whether coarse supervision improves fine-grained disease classification.

---

### Slide 9. Expected Contribution

#### Key Message

This study proposes a heart sound domain adaptation strategy using general-purpose audio representations and validates the effect of coarse-to-fine transfer learning.

#### Bullet Points

- Heart sound domain adaptation using M2D-X.
- Coarse-to-fine transfer from CirCor to BMD-HS.
- Comparison of sequential transfer and joint multi-task learning.
- Practical strategy for low-resource heart disease classification.

---

## 13. One-paragraph Summary

This study aims to improve low-resource heart disease classification performance by adapting a general-purpose audio representation model such as M2D-X to the heart sound domain, in a setting where heart sound-specific foundation models are still limited. First, the model learns coarse-level representations related to overall abnormality in heart sounds through outcome normal/abnormal binary classification on the CirCor dataset. Then, sequential transfer learning is performed by transferring the learned representation to fine-grained heart disease classification on the BMD-HS dataset, including AS, AR, MR, and MS. This study also compares the sequential transfer strategy with joint multi-task learning using CirCor and BMD-HS together. In addition, a direct fine-tuning baseline using only BMD-HS without CirCor is evaluated to analyze how coarse-to-fine transfer and multi-task learning affect target disease classification performance.

---

## 14. Recommended Terminology

The key terms used in this study can be defined as follows.

| Term | Meaning |
|---|---|
| Foundation Audio Model | A general-purpose audio representation model pretrained on large-scale audio data |
| Heart Sound Domain Adaptation | The process of adapting a general-purpose audio model to heart sound data |
| Coarse-level Representation | A representation that captures broad abnormality, such as normal/abnormal outcome |
| Fine-grained Disease Classification | A task that classifies specific heart diseases such as AS, AR, MR, and MS |
| Sequential Transfer Learning | A learning strategy that first trains on the source task and then transfers to the target task |
| Joint Multi-task Learning | A learning strategy that trains multiple tasks simultaneously using a shared encoder |
| Source Dataset | CirCor |
| Target Dataset | BMD-HS |
| Source Task | CirCor outcome normal/abnormal binary classification |
| Target Task | BMD-HS heart disease multi-label classification |

---

## 15. Current Research Direction

The current research direction can be summarized as follows.

```text
Qwen2-Audio-based medical audio assistant
        ↓
M2D-X-based heart sound domain adaptation
        ↓
Coarse representation learning using the CirCor outcome task
        ↓
Fine-grained transfer to the BMD-HS disease task
        ↓
Comparison between sequential transfer and joint multi-task learning
```

Ultimately, this study is not defined as **“building yet another heart sound classifier,”** but as a study that analyzes **how to effectively adapt a general-purpose audio foundation model to the heart sound domain**.
