FROM python:3.11-slim

# Install system dependencies and uv (via installer — avoids multi-stage COPY --from syntax)
RUN apt-get update && apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/* && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /workspace

# Install all workshop requirements (wandb file chains to requirements.txt via -r)
COPY requirements.txt requirements-wandb.txt ./
RUN uv pip install --system --no-cache -r requirements-wandb.txt

EXPOSE 8888
