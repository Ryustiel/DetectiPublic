# Builder
FROM python:3.12-slim-bookworm AS builder

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
ARG VISION_EXTRAS=""

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml ./

RUN if [ -n "$VISION_EXTRAS" ]; then \
        UV_HTTP_TIMEOUT=300 uv sync --no-install-project --no-dev --extra "$VISION_EXTRAS"; \
    else \
        UV_HTTP_TIMEOUT=300 uv sync --no-install-project --no-dev; \
    fi

# Runtime
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME="/app/assets/models" \
    TOKENIZERS_PARALLELISM=false

COPY --from=builder /app/.venv /app/.venv
