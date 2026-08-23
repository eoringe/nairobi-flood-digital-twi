# =============================================================================
#  Nairobi Urban Flood Digital Twin — Dashboard Container
#
#  Builds and serves the Plotly Dash + Pydeck WebGL dashboard
#  (src/dashboard/app.py). Trained model weights (models/) and processed
#  arrays (data/processed/) are expected to be mounted as volumes rather
#  than baked into the image — MEMORY_CONSTRAINTS.md already treats these
#  as external artefacts refreshed by the ingestion/training pipeline, and
#  baking multi-GB .npy files into an image defeats the point of a
#  container that's meant to be cheap to rebuild and ship.
#
#  Build : docker build -t nairobi-flood-twin .
#  Run   : docker run -p 8050:8050 -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" nairobi-flood-twin
#  Or    : docker compose up
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# build-essential is a safety net for any dependency without a prebuilt
# manylinux wheel on this platform; most of the geospatial stack (rasterio,
# shapely, fiona, pyproj) ships its own GDAL/GEOS/PROJ binaries in-wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

# Data/model directories are created empty here; real content is mounted
# in at `docker run` / `docker compose up` time.
RUN mkdir -p data/processed/arrays data/raw models/autoencoder models/time_series

ENV PYTHONUNBUFFERED=1
EXPOSE 8050

CMD ["python", "-m", "src.dashboard.app", "--host", "0.0.0.0", "--port", "8050"]
