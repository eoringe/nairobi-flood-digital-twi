"""
src.dashboard.app
=================
Nairobi Urban Flood Digital Twin — Main Web Application Entrypoint

USAGE
-----
    python -m src.dashboard.app
    or
    python src/dashboard/app.py --port 8050
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from dash import Dash
import dash_bootstrap_components as dbc
from loguru import logger

from src.dashboard.layouts import build_dashboard_layout
from src.dashboard.callbacks import register_callbacks, start_background_warmup, warmup_status

GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Chakra+Petch:wght@500;600;700"
    "&family=IBM+Plex+Sans:wght@400;500;600"
    "&family=IBM+Plex+Mono:wght@400;500;600"
    "&display=swap"
)

# Initialize Dash application. DARKLY supplies base component behavior
# (grid, form controls, accordion mechanics); src/dashboard/assets/custom.css
# loads after it and redefines every visual token — see that file for the
# design system (palette, type, panel/badge/slider overrides).
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, GOOGLE_FONTS_URL],
    title="Nairobi Urban Flood Digital Twin",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.layout = build_dashboard_layout()
register_callbacks(app)
server = app.server  # Flask WSGI server instance


@server.route("/healthz")
def healthz():
    """
    Liveness check for the hosting platform (railway.json healthcheckPath).

    Returns 200 as soon as the app can serve pages. The road graph and cached
    forecasts keep loading in the background; `warmup_complete` reports whether
    they are ready, without holding the deploy back while they finish.
    """
    return {"status": "ok", **warmup_status()}, 200

# A WSGI server such as gunicorn (the Docker image) imports `server` and never
# calls main(), so warm-up is started here when the environment asks for it.
if os.environ.get("TWIN_WARMUP") == "1":
    start_background_warmup()


def main(host: str = "127.0.0.1", port: int = 8050, debug: bool = False) -> None:
    logger.info("============================================================")
    logger.info("🌊 Starting Nairobi Urban Flood Digital Twin Web Server...")
    logger.info(f"   URL  : http://{host}:{port}/")
    logger.info("   Stack: Dash + Pydeck WebGL + PyTorch U-Net (Model B)")
    logger.info("   Output: flood EXTENT as per-cell probability, not depth")
    logger.info("   Loading road network and outlooks in the background...")
    logger.info("============================================================")
    # Not started in debug mode's reloader parent, which would warm up twice.
    if not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        start_background_warmup()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nairobi Urban Flood Digital Twin Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP address")
    parser.add_argument("--port", type=int, default=8050, help="Server port number")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    main(host=args.host, port=args.port, debug=args.debug)
