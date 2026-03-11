# 🚀 Punctuation-Restoration Pro

Advanced punctuation and case restoration system built using **RoBERTa-base** with production-grade training strategies and deployment-ready inference.

---

## 🔗 Live Demo

Hugging Face Space:  
https://huggingface.co/spaces/cds006/Punctuation/tree/main

Model Repository:  
https://huggingface.co/spaces/cds006/Punctuation/tree/main

---

## 🎯 Project Overview

Automatic Speech Recognition (ASR) systems produce raw lowercase text without punctuation:

Input: hello how are you doing today i hope everything is fine

Output: Hello, how are you doing today? I hope everything is fine.


This project restores:
- Proper punctuation
- Sentence boundaries
- Capitalization
- Context-aware casing after punctuation

Designed for real-world ASR post-processing pipelines.

---

## 🧠 Model Architecture

- **Backbone:** `roberta-base`
- **Task:** Token Classification
- **Labels (8 composite classes):**
  - O
  - COMMA
  - PERIOD
  - PERIOD+CAPS
  - QM
  - QM+CAPS
  - EXCLAM
  - EXCLAM+CAPS
- **Loss Function:** Focal Loss (γ = 2.0)
- **Imbalance Handling:** Log-smoothed class weights + 50× rare-class oversampling
- **Mixed Precision:** FP16 training
- **Inference Strategy:** Sliding window with overlap stitching

---

## 📊 Evaluation Metrics

Primary Metric:
- **Macro F1 Score:** 0.7064

Additional Metrics:
- Accuracy: 97%
- Weighted F1: 0.97
- Per-class F1:
  - COMMA ≈ 0.90
  - PERIOD ≈ 0.93
  - QM ≈ 0.93
  - EXCLAM ≈ 0.83

Macro F1 is used as the primary credibility metric due to severe class imbalance.

---

## 🏗 Training Strategy

Key techniques used:

- Rare class oversampling (×50)
- Focal Loss for hard-example emphasis
- Log-smoothed dynamic class weighting
- Linear warmup scheduler (10%)
- Gradient clipping
- FP16 mixed precision
- Best-checkpoint saving based on Macro F1

Training configuration is defined in `config.py`.

---

## 📂 Project Structure
```
├── app.py
├── config.py
├── dataset.py
├── inference_engine.py
├── train.py
├── requirements.txt
└── models
└── punctuation_restorer
```

⚠ Known Limitations

- Lyrics and Poetry : Non-standard formatting may not be preserved.

- Code Snippets : Programming syntax is not supported.

- Extremely Rare CAPS Variants : Very low-frequency classes (QM+CAPS, EXCLAM+CAPS) remain challenging.

- Long-Range Context : Context beyond 512 tokens relies on sliding-window merging.


👤 Author

Devinder Solanki

