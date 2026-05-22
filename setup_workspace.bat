@echo off
REM ============================================================================
REM  Nairobi Urban Flood Digital Twin — Workspace Layout Generator
REM  Creates the canonical directory structure for local development.
REM  Run once after cloning the repository.
REM ============================================================================

echo.
echo ====================================================
echo   Nairobi Flood Digital Twin — Workspace Setup
echo ====================================================
echo.

REM ---------- Data directories ----------
echo [1/5] Creating data directories ...
if not exist "data\raw"        mkdir "data\raw"
if not exist "data\processed"  mkdir "data\processed"
if not exist "data\spatial"    mkdir "data\spatial"

REM ---------- Model checkpoint directory ----------
echo [2/5] Creating model checkpoint directory ...
if not exist "models" mkdir "models"

REM ---------- Source package skeleton ----------
echo [3/5] Creating source packages ...
if not exist "src\ingestion"  mkdir "src\ingestion"
if not exist "src\models"     mkdir "src\models"
if not exist "src\dashboard"  mkdir "src\dashboard"
if not exist "src\utils"      mkdir "src\utils"

REM  Drop __init__.py into every package so Python treats them as importable
for %%d in (src src\ingestion src\models src\dashboard src\utils) do (
    if not exist "%%d\__init__.py" (
        echo # Nairobi Flood Digital Twin — package marker> "%%d\__init__.py"
    )
)

REM ---------- Config directory ----------
echo [4/5] Creating config directory ...
if not exist "config" mkdir "config"

REM  Seed a blank YAML config if none exists
if not exist "config\settings.yaml" (
    (
        echo # ---------------------------------------------------------------
        echo # Project Configuration — Nairobi Flood Digital Twin
        echo # ---------------------------------------------------------------
        echo.
        echo project:
        echo   name: nairobi-flood-digital-twin
        echo   version: 0.1.0
        echo.
        echo map_styles:
        echo   mapbox_dark:  "mapbox://styles/mapbox/dark-v11"
        echo   mapbox_light: "mapbox://styles/mapbox/light-v11"
        echo   mapbox_satellite: "mapbox://styles/mapbox/satellite-streets-v12"
        echo.
        echo environment:
        echo   profile: development        # development ^| staging ^| production
        echo   log_level: DEBUG
        echo   max_workers: 4
    ) > "config\settings.yaml"
)

REM ---------- Tests directory ----------
echo [5/5] Creating tests directory ...
if not exist "tests" mkdir "tests"
if not exist "tests\__init__.py" (
    echo # test-suite package marker> "tests\__init__.py"
)

REM ---------- Placeholder .gitkeep files (so Git tracks empty dirs) ----------
for %%d in (data\raw data\processed data\spatial models) do (
    if not exist "%%d\.gitkeep" type nul > "%%d\.gitkeep"
)

echo.
echo ====================================================
echo   Workspace layout created successfully!
echo ====================================================
echo.
echo   data\raw\           — raw source files
echo   data\processed\     — analysis-ready artefacts
echo   data\spatial\       — shapefiles, GeoJSON, GeoPackages
echo   models\             — trained weights ^& checkpoints
echo   src\ingestion\      — data download ^& caching
echo   src\models\         — PyTorch model definitions
echo   src\dashboard\      — Dash app layout ^& callbacks
echo   src\utils\          — memory checks, logging, helpers
echo   config\             — environment profiles ^& map styles
echo   tests\              — unit ^& integration tests
echo.
