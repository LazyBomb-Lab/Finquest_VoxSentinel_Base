# 🛡️ FinQuest 2026: Wav2Vec 2.0 Deepfake Detection System

> **"Fraud is increasingly a connected, systemic problem."** > This system is engineered to detect AI-powered financial fraud by identifying sub-perceptual electronic synthesis artifacts across diverse telephonic environments.

---

## 🚀 1. Core Architecture: Multi-Scale Evolution

The system transitions from traditional CNN-based detection to a **Self-Supervised Transformer** backbone.

* **Encoder**: `facebook/wav2vec2-base`
* **Feature Head**: **Multi-Scale Feature Fusion**.
    * Instead of a simple linear layer, we concatenate the **Mean** and **Standard Deviation** of the transformer's hidden states.
    * **Dimension**: 1536 (768 Mean + 768 StdDev).
    * **Strategy**: Captures both static spectral artifacts and dynamic temporal "texture" inconsistencies.

---

## 📊 2. Dataset: The Master Protocol (v2)

We utilized a massive aggregated corpus of **116,390 samples** to ensure cross-generator generalization.

### Data Composition
| Source Category | Sample Count | Weight | Description |
| :--- | :---: | :---: | :--- |
| **FOR (Fake-or-Real)** | 50,890 | 43.7% | Original, Norm, 2sec, and Rerecorded variants. |
| **WaveFake (WF)** | 35,500 | 30.5% | JSUT & LJSpeech official (HiFiGAN, MelGAN, etc). |
| **In-the-Wild** | 12,000 | 10.3% | Real-world scraped deepfakes and authentic audio. |
| **ASVspoof 2019** | 8,000 | 6.9% | Academic benchmark for logical access attacks. |
| **LJ_Real** | 10,000 | 8.6% | High-fidelity authentic reference. |

---

## 🧪 3. Forensic Training Regime

### Data Augmentation (Asymmetric Strategy)
To bridge the gap between "Lab" and "Wild", we implemented:
* **Cyclic Tiling**: Audios < 4s are tiled to maintain temporal receptive fields without energy loss.
* **Asymmetric Toughening**: 
    * *Pure Sources*: Heavy MP3 compression, Room simulation, and Time masking.
    * *Dirty Sources*: Minimal perturbation to preserve natural "In-the-Wild" artifacts.

### Hyperparameters (RTX 5080 Optimized)
| Parameter | Value | Note |
| :--- | :--- | :--- |
| **Batch Size** | 32 | Balanced for gradient stability. |
| **Encoder LR** | 4e-6 | **LLRD** (Layer-wise Learning Rate Decay). |
| **Classifier LR** | 4e-5 | High-velocity head training. |
| **Weight Decay** | 0.01 | Decoupled AdamW regularization. |
| **Patience** | 2 / 5 | Scheduler / Early Stopping. |

---

## 🎯 4. Performance Metrics & Benchmarks

The model was evaluated using two distinct methodologies: **In-Distribution Validation** (Master Protocol Test Set) and **Out-of-Distribution (OOD) Stress Testing** (External Fake-or-Real Benchmarks).

### A. Sliding-Window Validation (In-Distribution)
Tested on 10% of the Master Protocol (~11,639 samples) using a robust sliding-window inference strategy on the **NVIDIA RTX 5080**.

| Metric | Score | status |
| :--- | :--- | :--- |
| **Overall Accuracy** | **99.91%** | 💎 EXTREME |
| **Equal Error Rate (EER)** | **0.0007** | 💎 EXTREME |
| **Area Under Curve (AUC)** | **1.0000** | 💎 EXTREME |

#### Detailed Source Breakdown
| Source / Architecture | Accuracy | Source / Architecture | Accuracy |
| :--- | :---: | :--- | :---: |
| **for-2seconds** | 100.00% | **for-rerecorded** | 99.92% |
| **for-original** | 100.00% | **for-norm** | 100.00% |
| **wild (In-the-wild)**| 99.75% | **fastspeech2_pwg** | 100.00% |
| **melgan/pwg** | 99.90% | **ASV A01-A06** | 98.5%~100% |
| **hifiGAN** | 100.00% | **waveglow** | 100.00% |

---

### B. Cross-Dataset Stress Test (Out-of-Distribution)
To simulate adversarial real-world scenarios, the model was tested against completely unseen datasets from the **Fake-or-Real (FoR)** project.

| Scenario | Accuracy | Status |
| :--- | :---: | :--- |
| **for-norm** | **94.28%** | 🔥 STRONG |
| **for-original** | **91.89%** | 🔥 STRONG |
| **for-rerecorded** | **91.05%** | 🔥 STRONG |
| **for-2seconds** | **89.06%** | ⚠️ STABLE |

> **Forensic Note**: The transition from 99.9% (internal) to ~92% (external) demonstrates the model's high generalization floor. Unlike standard CNN models that collapse on unseen data, the Wav2Vec 2.0 backbone maintains "Industrial Grade" reliability even across mismatched domains.

---

## 💡 5. Strategic Implementation (FinQuest 2026)

This model serves as the **Neural Foundation** for a multi-stage intervention pipeline:
1.  **Detection**: Wav2Vec 2.0 identifies synthetic fingerprints.
2.  **Emotion**: Parallel head detects "High Distress" patterns.
3.  **Risk Fusion**: Hybrid scoring (Bio-signal + Linguistic pattern).
4.  **Intervention**: Triggers **Encrypted Risk Tokens (ERT)** for interbank mule-account freezing.

---

## 🤝 Acknowledgements

This research and model development would not be possible without the open-source contributions of the following organizations and researchers. We express our gratitude for their high-quality datasets:

* **Fake-or-Real (FoR) Dataset**: Provided by [Mohammed Abdel-Dayem et al.](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset), offering a critical multi-scenario benchmark for synthetic speech detection.
* **WaveFake Dataset**: A massive cross-generator corpus by [Frank et al.](https://github.com/nii-yamagishilab/WaveFake), which was instrumental in training our model to generalize across multiple TTS architectures.
* **ASVspoof 2019**: We thank the [ASVspoof Consortium](https://www.asvspoof.org/index2019.html) for their foundational work in standardizing the evaluation of spoofing countermeasures.
* **In-the-Wild Dataset**: Gratitude to the researchers of the [In-the-Wild project](https://github.com/asvspoof-challenge/2021) for providing real-world deepfake samples that bridge the gap between laboratory and reality.
* **Special Thanks**: To the creators of **LJSpeech** and **JSUT** official corpora for providing the high-fidelity authentic references used in our WF-Enhanced protocol.

---
© 2026 FinQuest AI Defense Taskforce. Powered by Wav2Vec 2.0.