#!/bin/sh
# Container entrypoint for hosted deployment.
#
# Two things differ from `docker compose up`, and both are silent failures if missed:
#
# 1. **The platform assigns the port.** Railway, Render and Fly all inject $PORT and
#    route to it. A container listening on a hardcoded 8000 passes its own health check
#    and never receives traffic — the deploy looks green and the service is unreachable.
#
# 2. **The database starts empty.** A fresh managed Postgres has no tables, and nothing
#    else in the app runs migrations. Without this line the first request fails on a
#    missing relation, which reads like a code bug and isn't.
#
# `exec` matters: uvicorn replaces the shell as PID 1 so it receives SIGTERM directly.
# Otherwise the platform's graceful-shutdown window expires and it SIGKILLs instead,
# which can interrupt a sweep mid-write.
set -e

echo "running migrations..."
uv run alembic upgrade head

echo "starting api on port ${PORT:-8000}"
exec uv run uvicorn omaha.api:app --host 0.0.0.0 --port "${PORT:-8000}"
