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

# Model weights live in /models so a rebuild doesn't re-download 130 MB. No VOLUME
# instruction: Railway rejects the Dockerfile outright if it finds one ("docker VOLUME
# is not supported, use Railway Volumes"), and it buys nothing here — compose already
# names the volume explicitly, and hosted platforms attach theirs out of band.
RUN mkdir -p /models

RUN chmod +x scripts/start.sh

# Migrations then uvicorn, on whatever port the platform assigns. See scripts/start.sh
# for why both matter — locally $PORT is unset and this falls back to 8000, so compose
# behaviour is unchanged.
CMD ["./scripts/start.sh"]
