# News Topic Classifier - Fine-tuned BERT on AG News

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=flat&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat&logo=huggingface)
![Gradio](https://img.shields.io/badge/Gradio-5.x-orange?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)

> **DevelopersHub Corporation - AI/ML Engineering Internship**

A fine-tuned `bert-base-uncased` transformer model for multi-class news topic classification, trained on the AG News dataset (120,000 articles). Deployed as an interactive Gradio web application with real-time headline classification and confidence scores.

---

## Overview

With millions of news articles published daily, automatically categorizing them into topics is a fundamental NLP challenge. This project demonstrates **transfer learning** with BERT - taking a model pre-trained on 3.3 billion words and fine-tuning it for news classification with just 8,000 training samples, achieving ~93–95% accuracy on the full AG News test set.

### Categories

| Label | Category | Description |
|---|---|---|
| 0 | 🌍 World | International news, politics, geopolitics |
| 1 | 🏅 Sports | Sports events, athletes, tournaments |
| 2 | 💼 Business | Finance, markets, corporate, economy |
| 3 | 💻 Sci/Tech | Science, technology, AI, space exploration |

---

## Features

- Fine-tuned `bert-base-uncased` (110M parameters) on AG News
- Stratified training subset (8,000 samples, 2,000 per class) for balanced learning
- HuggingFace `Trainer` API with early stopping and best checkpoint saving
- Evaluation with **Accuracy**, **Weighted F1**, and **Macro F1**
- Visualizations: class distribution, training curves, confusion matrix, per-class F1, confidence analysis
- Interactive **Gradio** web app with confidence scores and probability bars for all 4 classes
- 10 pre-loaded example headlines for instant demo

---

## Results

| Metric | Score |
|---|---|
| Test Accuracy | ~93–95% |
| Weighted F1 | ~93–95% |
| Macro F1 | ~93–95% |

> *BERT fine-tuned on full 120K AG News reaches ~94.6% - published benchmark*

---

## Demo

```
Input  : "NASA's James Webb Telescope captures image of distant galaxy cluster"
Output : 💻 Sci/Tech (97.3% confident)

Input  : "Real Madrid wins Champions League final against Manchester City"
Output : 🏅 Sports (99.1% confident)

Input  : "Federal Reserve holds interest rates amid inflation concerns"
Output : 💼 Business (95.8% confident)

Input  : "United Nations calls emergency meeting over Middle East tensions"
Output : 🌍 World (98.2% confident)
```

---

## Project Structure

```
bert-news-classifier/
├── task1_bert_news_classifier.ipynb   # Training notebook (run on Colab GPU)
├── gradio_app.py                      # Gradio deployment app
├── requirements.txt                   # Dependencies
├── README.md
├── bert_news_classifier_final/        # Saved model (download separately)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   ├── vocab.txt
│   └── special_tokens_map.json
└── visualizations/
    ├── eda_plots.png
    ├── training_metrics.png
    ├── confusion_matrix.png
    ├── per_class_f1.png
    └── confidence_analysis.png
```

---

## Setup and Installation

### Prerequisites
- Python 3.10+
- GPU recommended for training (Google Colab T4 works perfectly)
- CPU sufficient for inference / running the Gradio app

### 1. Clone the repository
```bash
git clone https://github.com/RabiyaMalik242/bert-news-classifier.git
cd bert-news-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Training (Google Colab - GPU Required)

Fine-tuning BERT on CPU takes 8–12 hours. Use Google Colab with a free T4 GPU (~15–25 minutes).

**1. Enable GPU in Colab:**
```
Runtime → Change runtime type → T4 GPU → Save
```

**2. Upload and open the notebook:**
Upload `task1_bert_news_classifier.ipynb` to Colab and run all cells.

**3. Download the trained model:**
```python
import shutil
from google.colab import files
shutil.make_archive('bert_news_classifier_final', 'zip', './bert_news_classifier_final')
files.download('bert_news_classifier_final.zip')
```

**4. Unzip** into your project folder alongside `gradio_app.py`.

---

## Running the Gradio App (Local - No GPU Needed)

Once you have the trained model folder:

```bash
python gradio_app.py
```

Open `http://localhost:7860` in your browser.

The app loads the saved model from `./bert_news_classifier_final/` — no retraining, no API key, completely offline.

---

## Model Architecture

```
Input Text
    ↓
BERT Tokenizer (WordPiece, max_length=128)
    ↓
bert-base-uncased
    ├── 12 Transformer Layers
    ├── 768 Hidden Dimensions
    └── 12 Attention Heads
    ↓
[CLS] Token Representation (768-dim)
    ↓
Dropout
    ↓
Linear Layer (768 → 4)
    ↓
Softmax
    ↓
Class Probabilities (World / Sports / Business / Sci/Tech)
```

---

## Training Configuration

| Parameter | Value |
|---|---|
| Base Model | `bert-base-uncased` |
| Dataset | AG News (HuggingFace: `fancyzhx/ag_news`) |
| Training Samples | 8,000 (stratified, 2,000 per class) |
| Validation Samples | 2,000 (500 per class) |
| Test Samples | 7,600 (full AG News test set) |
| Epochs | 3 |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Warmup Ratio | 0.1 |
| Weight Decay | 0.01 |
| Max Sequence Length | 128 tokens |
| Mixed Precision | FP16 (GPU only) |
| Early Stopping | Patience = 2 |

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch |
| Transformer Model | `bert-base-uncased` (HuggingFace Transformers) |
| Dataset | AG News via HuggingFace Datasets (`fancyzhx/ag_news`) |
| Training | HuggingFace `Trainer` API |
| Evaluation | Scikit-learn (Accuracy, F1, Confusion Matrix) |
| Deployment | Gradio |
| Visualization | Matplotlib, Seaborn |

---

## Skills Demonstrated

- Transfer learning and transformer fine-tuning
- NLP tokenization with BERT WordPiece tokenizer
- HuggingFace Transformers and Trainer API
- Multi-class text classification
- Evaluation metrics: Accuracy, Weighted F1, Macro F1, ROC-AUC
- Confusion matrix analysis and confidence calibration
- Gradio model deployment
- GPU-accelerated training on Google Colab

---

## Author

**Rabiya Malik** - 
BS Software Engineering

AI/ML Engineering Intern @ DevelopersHub Corporation.

[![GitHub](https://img.shields.io/badge/GitHub-RabiyaMalik242-black?style=flat&logo=github)](https://github.com/RabiyaMalik242)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)]((https://www.linkedin.com/in/rabiya-malik-325848336/))

---

## License

This project is licensed under the MIT License.
