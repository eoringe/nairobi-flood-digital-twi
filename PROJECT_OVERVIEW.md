# Nairobi Urban Flood Digital Twin — Project Overview

## Vision

A real-time, predictive digital twin of Nairobi's urban flood dynamics.  
The system ingests multi-source geospatial data — satellite imagery, IoT sensor
streams, and meteorological forecasts — to drive a deep-learning engine that
produces spatiotemporal flood-risk surfaces rendered on an interactive 3-D
web dashboard.

## Core Capabilities

| Layer            | Responsibility                                                       |
| ---------------- | -------------------------------------------------------------------- |
| **Ingestion**    | Pull and cache satellite tiles (Sentinel-2 via Google Earth Engine), weather API feeds, and IoT sensor telemetry. |
| **Processing**   | Clean, reproject, and tile raster/vector data into analysis-ready formats (GeoTIFF, GeoParquet). |
| **Modelling**    | Convolutional encoder–decoder for spatial feature extraction; LSTM sequence head for temporal flood forecasting. |
| **Dashboard**    | Plotly Dash + Pydeck 3-D map overlays showing live risk surfaces, historical replay, and scenario simulation. |
| **Utils**        | Memory guard-rails, logging, configuration management, and shared helpers. |

## Repository Layout

```
nairobi-flood-digital-twin/
├── config/             # Environment profiles, map-style keys
├── data/
│   ├── raw/            # Untouched source files
│   ├── processed/      # Analysis-ready artefacts
│   ├── spatial/        # Shapefiles, GeoJSON, GeoPackages
│   └── scenarios.db    # SQLite scenario-run history (src/persistence/)
├── models/             # Trained weights & checkpoints
├── src/
│   ├── grid_config.py  # Single source of truth for the Nairobi prediction grid
│   ├── ingestion/      # Data download & caching modules
│   ├── preprocessing/  # Terrain cropping, dataset assembly, buildings
│   ├── models/         # PyTorch model definitions & training loops
│   ├── dashboard/      # Dash app layout & callbacks
│   ├── persistence/    # Scenario-run history store (PostGIS upgrade path documented inline)
│   └── utils/          # Memory checks, logging, helpers
├── scripts/dev/        # One-off diagnostic scripts (not part of the test suite)
├── tests/               # Unit & integration tests (pytest)
├── .github/workflows/   # CI (runs the test suite on push/PR)
├── Dockerfile, docker-compose.yml
├── requirements.txt, requirements-dev.txt
├── setup_workspace.bat
├── .env.example
└── README.md
```

## Guiding Principles

1. **Memory-first design** — every tensor operation must respect the 16 GB RAM ceiling (see `MEMORY_CONSTRAINTS.md`).
2. **Reproducibility** — pinned dependencies, deterministic seeds, version-controlled configs.
3. **Decoupled layers** — ingestion, modelling, and visualisation are independently testable packages.
4. **Local-first** — the full pipeline must run on a single workstation without cloud GPUs.
