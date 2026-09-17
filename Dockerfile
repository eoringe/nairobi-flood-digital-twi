# syntax=docker/dockerfile:1
# =============================================================================
#  Nairobi Flood Digital Twin - dashboard container
#
#  Self-contained: the image carries the code plus the ~45 MB of trained model
#  and processed data the dashboard reads, so it runs on any machine with
#  Docker. Only writable state (scenario history, forecast cache) lives on a
#  volume, so it survives container restarts and rebuilds.
#
#  Local : docker compose up --build   ->  http://localhost:8050
#  Cloud : Railway builds this file (railway.json); see DEPLOYMENT.md
# =============================================================================

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so editing code does not reinstall them on every build.
COPY requirements-dashboard.txt .
RUN pip install -r requirements-dashboard.txt

# rasterio's wheel bundles GDAL but links against the system expat library,
# which the slim base image omits ("libexpat.so.1: cannot open shared object").
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user rather than root.
RUN useradd --create-home --uid 10001 twin

# Server config, code, trained model and processed data. .dockerignore narrows
# data/ and models/ to exactly the files the dashboard reads.
COPY --chown=twin:twin gunicorn.conf.py ./
COPY --chown=twin:twin src/ ./src/
COPY --chown=twin:twin data/processed/ ./data/processed/
COPY --chown=twin:twin models/time_series/ ./models/time_series/

RUN mkdir -p /app/state /app/data/raw && chown twin:twin /app/state /app/data /app/data/raw

USER twin

# TWIN_STATE_DIR  where the SQLite scenario history and live-forecast cache go
#                 (a volume in docker-compose.yml; a Railway volume at /app/state).
# TWIN_WARMUP     gunicorn imports the app instead of calling main(), so the
#                 background warm-up (road graph, outlooks, cached model runs)
#                 starts on import.
# OMP/MKL_NUM_THREADS  cloud hosts can report dozens of CPUs; an uncapped
#                 PyTorch thread pool on a shared allocation is slower, not faster.
# PORT            Railway injects its own; 8050 is the local default.
ENV TWIN_STATE_DIR=/app/state \
    TWIN_WARMUP=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    PORT=8050

EXPOSE 8050

# Used by docker compose. Railway ignores Docker HEALTHCHECK and uses
# railway.json's healthcheckPath (/healthz) instead.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os, sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8050') + '/healthz', timeout=4).status == 200 else 1)"

# Bind address, worker model and timeouts live in gunicorn.conf.py, which reads
# the PORT the platform injects.
CMD ["gunicorn", "--config", "gunicorn.conf.py", "src.dashboard.app:server"]
