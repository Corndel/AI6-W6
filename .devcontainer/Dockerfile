FROM python:3.11-slim

# Install uv for fast dependency installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install git (needed for cloning, used by some HF utilities)
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy and install all requirements (wandb file chains to requirements.txt via -r)
COPY requirements.txt requirements-wandb.txt ./
RUN uv pip install --system --no-cache -r requirements-wandb.txt

# Pre-cache the DistilBERT model and tokeniser so the workshop runs offline/fast
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('distilbert-base-uncased'); \
AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3); \
print('Model cache OK')"

# Pre-cache the financial_phrasebank dataset
RUN python -c "\
from datasets import load_dataset; \
load_dataset('financial_phrasebank', 'sentences_allagree', trust_remote_code=True); \
print('Dataset cache OK')"

EXPOSE 8888
