# Linux image for anything that needs ML wheels.
#
# The host dev machine (x86_64 macOS) can no longer install torch or onnxruntime —
# upstream dropped those wheels. Rather than pin ancient versions, model work runs
# here, which is also where it runs in production.

FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first, so a source edit doesn't invalidate the wheel layer
COPY pyproject.toml uv.lock* ./
RUN uv sync --extra embed --no-install-project --frozen || uv sync --extra embed --no-install-project

COPY . .
RUN uv sync --extra embed --frozen || uv sync --extra embed

# Model weights cached in a volume so a rebuild doesn't re-download 130 MB
VOLUME ["/models"]

CMD ["uv", "run", "uvicorn", "omaha.api:app", "--host", "0.0.0.0", "--port", "8000"]
