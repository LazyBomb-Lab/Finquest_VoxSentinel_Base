# 🛡️ Sentinel: VoxSentinel-Base(FinQuest 2026)

#### Hugging Face Repo: https://huggingface.co/JesseHuang922/VoxSentinel-Base

#### Dataset Repo (in case you want to experiment yourself): https://huggingface.co/datasets/JesseHuang922/Wav2Vec2-Deepfake-Sentinel-Base-Dataset 

## 🔍 Executive Overview

As generative voice cloning reaches human-level parity, synthetic speech has become a primary vector for high-stakes financial fraud, vishing (voice phishing), and biometric spoofing. Traditional anti-spoofing models often fail in real-world deployment due to channel degradation, compression artifacts, and overfitting to known Text-to-Speech (TTS) generators.

**VoxSentinel-Base** addresses this gap with a forensic-first design built specifically for **FinQuest 2026**:
* **Beyond Surface Acoustics**: Leverages a self-supervised `Wav2Vec 2.0` backbone paired with a custom **Multi-Scale Attentive Pooling (MSAP)** head to capture both static spectral anomalies and dynamic temporal variance ($\mu + \sigma$).
* **Built for the "Wild"**: Trained on the **116k Master Protocol (v2)** using an **Asymmetric Toughening** regime (heavy MP3 compression & room impulse simulation), ensuring resilience against noisy telephonic and rerecorded channels.
* **Proven Generalization**: Achieves **99.56% accuracy** on in-the-wild benchmarks while maintaining **>91% cross-dataset accuracy** on completely unseen adversarial scenarios.


> **"In the era of generative AI, trust is no longer a default setting—it must be verified."**
> Sentinel is an industrial-grade forensic system engineered to detect AI-powered financial fraud by identifying sub-perceptual electronic synthesis artifacts across diverse telephonic environments.

---

## 🚀 1. Core Architecture: Sentinel-Base (v1.0)

The system transitions from traditional CNN-based detection to a **Self-Supervised Transformer** backbone with a custom forensic head.

* **Backbone**: `facebook/wav2vec2-base` (Fine-tuned via Layer-wise Learning Rate Decay).
* **Feature Head**: **Multi-Scale Attentive Pooling (MSAP)**.
    * **Mechanism**: Instead of a simple mean, MSAP calculates **Attention-weighted Mean ($\mu$)** and **Attention-weighted Standard Deviation ($\sigma$)**.
    * **Dimension**: $768 (\mu) + 768 (\sigma) = 1536$ total features.
    * **Logic**: Captures both static spectral artifacts and dynamic temporal "texture" inconsistencies (e.g., unnatural smoothness in neural vocoders).
* **Classifier**: 3-Layer MLP ($1536 \rightarrow 512 \rightarrow 256 \rightarrow 2$) with BatchNorm and Dropout (0.3).

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
* **Asymmetric Toughening**: Heavy MP3 compression and Room simulation were applied to clean sources to simulate real-world telephonic degradation.

### Hyperparameters (RTX 5080 Optimized)
| Parameter | Value | Note |
| :--- | :--- | :--- |
| **Batch Size** | 32 | Balanced for gradient stability. |
| **Encoder LR** | 4e-6 | **LLRD** (Layer-wise Learning Rate Decay). |
| **Top-Layer LR** | 4e-5 | High-velocity head training (Pooling & MLP). |
| **Optimizer** | AdamW | Weight decay 0.01. |

---

## 📈 4. Performance & Validation

### A. Internal Benchmark (Master Protocol)
* **Best Dev Accuracy**: `99.85%`
* **Best Dev EER**: `0.0013`
* **In-the-Wild Test Accuracy**: **99.56%** (on 31,779 samples).

### B. Cross-Dataset Stress Test (Generalization)
Tested against completely unseen datasets to simulate adversarial scenarios:
* **Unseen FoR-Norm**: **94.28%** Accuracy
* **Unseen FoR-Rerecorded**: **91.05%** Accuracy

> **Forensic Note**: The consistency across unseen data ($>90\%$) proves that the **MSAP head** effectively learns intrinsic synthesis fingerprints rather than over-fitting to specific dataset biases.

---

## 💻 5. Implementation & Usage

This model requires the **SentinelForSyntheticDetection** class wrapper to handle the 1536-dim feature fusion.

```python
import torch
import torch.nn as nn
import librosa
import numpy as np
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model, AutoProcessor

# ==============================================================================
# 1. Architecture Definition (Required for custom MSAP head)
# ==============================================================================
class MultiScaleAttentivePooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128), 
            nn.Tanh(), 
            nn.Linear(128, 1)
        )
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_size)
        w = torch.softmax(self.attention(x), dim=1)
        mu = torch.sum(w * x, dim=1)
        # Robust variance calculation
        delta = x - mu.unsqueeze(1)
        var = torch.sum(w * (delta**2), dim=1)
        std = torch.sqrt(torch.clamp(var, min=1e-9))
        return torch.cat([mu, std], dim=-1) # (batch, hidden_size * 2)

class SentinelForSyntheticDetection(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.pooling = MultiScaleAttentivePooling(config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, 512), 
            nn.ReLU(), 
            nn.BatchNorm1d(512), 
            nn.Dropout(0.3), 
            nn.Linear(512, 256), 
            nn.ReLU(), 
            nn.Linear(256, 2)
        )
        self.post_init()

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        # Using last_hidden_state (Batch, Seq_Len, Hidden_Size)
        pooled_output = self.pooling(outputs.last_hidden_state)
        return self.classifier(pooled_output)

# ==============================================================================
# 2. Inference Logic (Forensic Grade)
# ==============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
repo = "JesseHuang922/VoxSentinel-Base"

# Load model and processor from Hugging Face
model = SentinelForSyntheticDetection.from_pretrained(repo).to(device).eval()
processor = AutoProcessor.from_pretrained(repo)

def predict(audio_path):
    """
    Forensic prediction with automatic channel mixing and resampling.
    """
    # 1. Load with mono=True: Automatically mixes stereo to mono (L+R)/2
    # 2. sr=16000: Forces resampling to the model's native frequency
    try:
        wav, _ = librosa.load(audio_path, sr=16000, mono=True)
        
        # Simple Voice Activity Detection (Optional: trims silence)
        wav, _ = librosa.effects.trim(wav, top_db=20)

        # Preprocess features
        inputs = processor(wav, return_tensors="pt", sampling_rate=16000)
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            logits = model(input_values)
            # Softmax to get confidence scores
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_idx].item()

        # Label Mapping: 1 -> Real, 0 -> Fake
        result = "Real" if pred_idx == 1 else "Fake"
        return f"Result: {result} ({confidence:.2%} confidence)"

    except Exception as e:
        return f"Error processing {audio_path}: {e}"

# Usage Example:
# print(predict("evidence_stereo_file.mp3"))

```

## 🤝 Acknowledgements

This research and model development would not be possible without the open-source contributions of the following organizations and researchers. We express our gratitude for their high-quality datasets:

* **Fake-or-Real (FoR) Dataset**: Provided by [Mohammed Abdel-Dayem et al.](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset), offering a critical multi-scenario benchmark for synthetic speech detection.
* **WaveFake Dataset**: A massive cross-generator corpus by [Frank et al.](https://github.com/nii-yamagishilab/WaveFake), which was instrumental in training our model to generalize across multiple TTS architectures.
* **ASVspoof 2019**: We thank the [ASVspoof Consortium](https://www.asvspoof.org/index2019.html) for their foundational work in standardizing the evaluation of spoofing countermeasures.
* **In-the-Wild Dataset**: Gratitude to the researchers of the [In-the-Wild project](https://github.com/asvspoof-challenge/2021) for providing real-world deepfake samples that bridge the gap between laboratory and reality.
* **Special Thanks**: To the creators of **LJSpeech** and **JSUT** official corpora for providing the high-fidelity authentic references used in our WF-Enhanced protocol.

---
© 2026 FinQuest AI Defense Taskforce. Powered by Wav2Vec 2.0.
