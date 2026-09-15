# syntax=docker/dockerfile:1
# =============================================================================
#  Nairobi Flood Digital Twin - dashboard container
#
#  Self-contained: the image carries the code plus the ~45 MB of trained model
#  and processed data the dashboard reads, so it runs on any machine with
#  Docker. Only writable state (scenario history, forecast cache) lives on a
#  volume, so it survives container restarts and rebuilds.
#
#  Build and run : docker compose up --build
#  Open          : http://localhost:8050
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

# Code, trained model and processed data. .dockerignore narrows data/ and
# models/ to exactly the files the dashboard reads.
COPY --chown=twin:twin src/ ./src/
COPY --chown=twin:twin data/processed/ ./data/processed/
COPY --chown=twin:twin models/time_series/ ./models/time_series/

RUN mkdir -p /app/state /app/data/raw && chown twin:twin /app/state /app/data /app/data/raw

USER twin

# TWIN_STATE_DIR: where the SQLite scenario history and live-forecast cache go
# (mounted as a volume by docker-compose.yml).
# TWIN_WARMUP: gunicorn imports the app instead of calling main(), so the
# background warm-up (road graph, outlooks, cached model runs) starts on import.
ENV TWIN_STATE_DIR=/app/state \
    TWIN_WARMUP=1

EXPOSE 8050

# Start-up loads the model, road graph and forecasts (~40 s), hence the grace period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8050/', timeout=4).status == 200 else 1)"

# One worker, many threads: the model, road graph and forecast caches live in
# process memory, so extra worker processes would each load their own copy and
# not share caches. The long timeout covers first-time replay downloads.
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "1", "--threads", "8", \
     "--timeout", "180", "--access-logfile", "-", "src.dashboard.app:server"]
