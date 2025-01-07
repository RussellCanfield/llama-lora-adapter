# Llama LoRA Adapter Fine-tuning Project

A Python project for fine-tuning Llama 3.2 models using LoRA adapters.

## Overview

This project fine-tunes a Llama 3.2 model on sample JSON data.

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or Apple silicon
- At least 16GB RAM

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd lora-adapters

pip install torch torchvision torchaudio
pip install transformers peft trl datasets huggingface-hub python-dotenv tensorboard
pip install bitsandbytes-apple-silicon

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Running the training script:

```bash
python src/main.py
```

## Testing

```bash
python src/inference.py
```