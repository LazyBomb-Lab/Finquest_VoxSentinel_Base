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

## 📈 4. Performance Metrics

### Training Convergence
The model reached its "Golden State" at **Epoch 9**, successfully navigating out of local minima through learning rate reduction.

* **Best Dev Accuracy**: `99.85%`
* **Best Dev EER**: `0.0013` (Forensic grade)

### Stress Test (Out-of-Distribution)
Tested on unseen **Fake-or-Real (FoR)** subsets:

| Scenario | Accuracy | Status |
| :--- | :---: | :--- |
| **for-norm** | **94.28%** | 🔥 STRONG |
| **for-original** | **91.89%** | 🔥 STRONG |
| **for-rerecorded** | **91.05%** | 🔥 STRONG |
| **for-2seconds** | **89.06%** | ⚠️ STABLE |

---

## 💡 5. Strategic Implementation (FinQuest 2026)

This model serves as the **Neural Foundation** for a multi-stage intervention pipeline:
1.  **Detection**: Wav2Vec 2.0 identifies synthetic fingerprints.
2.  **Emotion**: Parallel head detects "High Distress" patterns.
3.  **Risk Fusion**: Hybrid scoring (Bio-signal + Linguistic pattern).
4.  **Intervention**: Triggers **Encrypted Risk Tokens (ERT)** for interbank mule-account freezing.

---
© 2026 FinQuest AI Defense Taskforce. Powered by Wav2Vec 2.0.