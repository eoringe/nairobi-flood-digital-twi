#!/usr/bin/env bash
# ============================================================================
#  Nairobi Urban Flood Digital Twin — Workspace Layout Generator
#  Creates the canonical directory structure for local development.
#  Run once after cloning the repository.
# ============================================================================

set -euo pipefail

echo ""
echo "===================================================="
echo "  Nairobi Flood Digital Twin — Workspace Setup"
echo "===================================================="
echo ""

# ---------- Data directories ----------
echo "[1/5] Creating data directories ..."
mkdir -p data/{raw,processed,spatial}

# ---------- Model checkpoint directory ----------
echo "[2/5] Creating model checkpoint directory ..."
mkdir -p models

# ---------- Source package skeleton ----------
echo "[3/5] Creating source packages ..."
for pkg in src src/ingestion src/models src/dashboard src/utils; do
    mkdir -p "$pkg"
    [ -f "$pkg/__init__.py" ] || echo "# Nairobi Flood Digital Twin — package marker" > "$pkg/__init__.py"
done

# ---------- Config directory ----------
echo "[4/5] Creating config directory ..."
mkdir -p config

if [ ! -f config/settings.yaml ]; then
cat > config/settings.yaml << 'EOF'
# ---------------------------------------------------------------
# Project Configuration — Nairobi Flood Digital Twin
# ---------------------------------------------------------------

project:
  name: nairobi-flood-digital-twin
  version: 0.1.0

map_styles:
  mapbox_dark:  "mapbox://styles/mapbox/dark-v11"
  mapbox_light: "mapbox://styles/mapbox/light-v11"
  mapbox_satellite: "mapbox://styles/mapbox/satellite-streets-v12"

environment:
  profile: development   # development | staging | production
  log_level: DEBUG
  max_workers: 4
EOF
fi

# ---------- Tests directory ----------
echo "[5/5] Creating tests directory ..."
mkdir -p tests
[ -f tests/__init__.py ] || echo "# test-suite package marker" > tests/__init__.py

# ---------- .gitkeep files ----------
for dir in data/raw data/processed data/spatial models; do
    [ -f "$dir/.gitkeep" ] || touch "$dir/.gitkeep"
done

echo ""
echo "===================================================="
echo "  Workspace layout created successfully!"
echo "===================================================="
echo ""
echo "  data/raw/           — raw source files"
echo "  data/processed/     — analysis-ready artefacts"
echo "  data/spatial/       — shapefiles, GeoJSON, GeoPackages"
echo "  models/             — trained weights & checkpoints"
echo "  src/ingestion/      — data download & caching"
echo "  src/models/         — PyTorch model definitions"
echo "  src/dashboard/      — Dash app layout & callbacks"
echo "  src/utils/          — memory checks, logging, helpers"
echo "  config/             — environment profiles & map styles"
echo "  tests/              — unit & integration tests"
echo ""
