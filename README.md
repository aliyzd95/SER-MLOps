# Whisper-Small Persian Fine-Tuning for ASR

This project presents a fine-tuned version of OpenAI's [Whisper](https://huggingface.co/openai/whisper-small) model for Persian Automatic Speech Recognition (ASR). It utilizes the [Common Voice 21.0 Persian](https://huggingface.co/datasets/aliyzd95/common_voice_21_0_fa) dataset for training and evaluation.

---

## 🔍 Overview

- **Model**: Whisper-small (fine-tuned on Persian)
- **Dataset**: Mozilla Common Voice v21.0 (Persian subset)
- **Tasks**: End-to-end speech-to-text transcription
- **Framework**: Hugging Face Transformers + Datasets

---

## 📦 Resources

- 🔗 **Fine-tuned Whisper model on Hugging Face**:  
  👉 [aliyzd95/whisper-small-persian-v1](https://huggingface.co/aliyzd95/whisper-small-persian-v1)

- 📚 **Persian Common Voice v21.0 Dataset** (preprocessed for ASR):  
  👉 [aliyzd95/common_voice_21_0_fa](https://huggingface.co/datasets/aliyzd95/common_voice_21_0_fa)

---

## 🗂 Project Files

- `create_dataset.ipynb`: Prepare and preprocess the Common Voice dataset for fine-tuning.
- `fine_tune.ipynb`: Fine-tune the Whisper model on the Persian dataset using Hugging Face `Trainer`.
- `evaluate.ipynb`: Evaluate the model performance and compute metrics like WER.

---

## 📊 Results

- **Training Loss**: `0.3323`
- **Word Error Rate (WER)**: `31.93%` (on validation set)

---

## 🚀 Quick Start

You can load and use the fine-tuned model and the dataset with just a few lines of code:

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

dataset_train = load_dataset("aliyzd95/common_voice_21_0_fa", split="train+validation")
dataset_test = load_dataset("aliyzd95/common_voice_21_0_fa", split="test")

model = WhisperForConditionalGeneration.from_pretrained("aliyzd95/whisper-small-persian-v1")
processor = WhisperProcessor.from_pretrained("aliyzd95/whisper-small-persian-v1")
