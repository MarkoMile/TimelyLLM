# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
        libsndfile1 libgl1 libglib2.0-0 \
        espeak-ng flite \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first (cache-friendly)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Copy project source
COPY . .

# Install the project itself (with notebook extra for Jupyter support)
RUN uv sync --frozen --extra notebook

EXPOSE 8888

# Working directory at project root (timelyllm/, dataset/, run_experiments.ipynb all visible)
WORKDIR /app

# CLI entrypoint: cd into timelyllm/ for correct relative paths
ENTRYPOINT ["sh", "-c", "cd timelyllm && uv run python3 rtllm.py \"$@\"", "--"]
CMD ["--list-presets"]
