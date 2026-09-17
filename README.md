# Nairobi Flood Digital Twin

A web-based 3D digital twin of Nairobi that predicts where floods will happen,
warns before they happen, and routes people around flooded roads.

Capstone project, BSc Informatics and Computer Science, Strathmore University.
Oringe Emmanuel Magunga · Supervisor: Eunice Manyasi

---

## Features

- **Flood prediction:** a deep-learning model (U-Net) gives the chance of flooding
  for every 70 m square of Nairobi from rainfall and terrain.
- **3D dashboard:** interactive map with 3D buildings, flood risk by place, and
  people at risk.
- **12-hour flood outlook:** uses the live rainfall forecast to warn when flooding
  is expected, e.g. *"moderate flooding in Mathare in about 2 hours"*.
- **Flood-aware routing:** routes around predicted flooding and re-plans as the
  forecast changes.
- **Location search:** about 18,000 Nairobi places, roads and landmarks.
- **Storm replays:** re-runs the Nov 2023, Apr 2024 and Mar 2026 floods.

## Results at a glance

| | |
|---|---|
| Accuracy on unseen storm seasons | F1 0.937 |
| Documented Nairobi floods detected | 6 of 6 (live system since 2022: 3 of 3) |
| Probability calibration error | 0.020 |
| Time to produce a flood map | about 1 second on a laptop CPU |

Details and caveats: [RESULTS.md](RESULTS.md) and [LIMITATIONS.md](LIMITATIONS.md).

---

## Run it

### With Docker (recommended)

```bash
docker compose up --build
```

Open http://localhost:8050 once `docker compose ps` shows **healthy** (1–3 minutes).

### Without Docker

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m src.dashboard.app
```

Open http://127.0.0.1:8050.

### Deploy to Railway

See [DEPLOYMENT.md](DEPLOYMENT.md).

---

## Project structure

```
src/
  dashboard/     web app: layout, callbacks, 3D map
  models/        U-Net training, prediction, calibration
  forecast/      12-hour outlook and rainfall correction
  routing/       flood-aware routing and location search
  ingestion/     data download and preparation scripts
  validation/    checks against real flood events
  persistence/   SQLite database
data/processed/  prepared data used by the app
models/          trained model files
```

## Data sources

CHIRPS rainfall · SRTM elevation · MERIT Hydro · ESA WorldCover ·
OpenStreetMap (roads and places) · WorldPop (population) · Open-Meteo (forecasts)

## Documentation

| Document | Contents |
|---|---|
| [TECH_STACK.md](TECH_STACK.md) | Tools and data acquisition |
| [RESULTS.md](RESULTS.md) | Evaluation results |
| [LIMITATIONS.md](LIMITATIONS.md) | What the system can and cannot claim |
| [REVISED_OBJECTIVES.md](REVISED_OBJECTIVES.md) | Research objectives |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Cloud deployment |
| [nairobi_flood_colab_training.ipynb](nairobi_flood_colab_training.ipynb) | Model training on Google Colab |

---

Map data © OpenStreetMap contributors (ODbL). Population data © WorldPop (CC BY 4.0).
