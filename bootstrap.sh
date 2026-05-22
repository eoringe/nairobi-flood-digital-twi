#!/usr/bin/env bash
# ============================================================================
#  Nairobi Urban Flood Digital Twin — Bootstrap Script (Linux / WSL2)
#  One-command setup: venv creation → pip upgrade → install → scaffold → verify
#
#  Usage:   chmod +x bootstrap.sh && ./bootstrap.sh
#  Expects: Python 3.10+ available as `python3`
# ============================================================================

set -euo pipefail

echo ""
echo "============================================================"
echo "  Nairobi Flood Digital Twin — Full Bootstrap"
echo "============================================================"
echo ""

# ---------- 1. Verify Python version ----------
echo "[1/6] Checking Python version ..."
PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: $PYTHON is not on PATH. Install Python 3.10+ and retry."
    exit 1
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "ERROR: Python 3.10+ required, found $PY_VERSION"
    exit 1
fi
echo "  Found Python $PY_VERSION — OK"

# ---------- 2. Create virtual environment ----------
echo ""
echo "[2/6] Creating virtual environment (.venv) ..."
if [ ! -d ".venv" ]; then
    "$PYTHON" -m venv .venv
    echo "  Virtual environment created."
else
    echo "  .venv already exists — skipping creation."
fi

# ---------- 3. Activate venv ----------
echo ""
echo "[3/6] Activating virtual environment ..."
# shellcheck disable=SC1091
source .venv/bin/activate

# ---------- 4. Upgrade native tools ----------
echo ""
echo "[4/6] Upgrading pip, setuptools, wheel ..."
python -m pip install --upgrade pip setuptools wheel --quiet

# ---------- 5. Install requirements ----------
echo ""
echo "[5/6] Installing pinned requirements (this may take a few minutes) ..."
python -m pip install -r requirements.txt --quiet
echo "  All dependencies installed successfully."

# ---------- 6. Run workspace scaffold ----------
echo ""
echo "[6/6] Running workspace layout generator ..."
chmod +x setup_workspace.sh
./setup_workspace.sh

# ---------- 7. Verify memory constraints ----------
echo ""
echo "============================================================"
echo "  Running memory diagnostics ..."
echo "============================================================"
python -m src.utils.memory_check

echo ""
echo "============================================================"
echo "  Bootstrap complete!"
echo "============================================================"
echo ""
echo "  To activate the environment in future sessions:"
echo "    source .venv/bin/activate"
echo ""
echo "  To launch the dashboard (when implemented):"
echo "    python -m src.dashboard.app"
echo ""
