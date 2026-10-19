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
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from loguru import logger

try:
    # Local runs read Firebase and account settings from .env; in Docker,
    # docker-compose.yml passes the same variables. Existing variables win.
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.auth.routes import init_auth
from src.dashboard.admin_dashboard import register_admin_callbacks
from src.dashboard.auth_pages import register_auth_callbacks
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
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP, GOOGLE_FONTS_URL],
    title="Nairobi Urban Flood Digital Twin",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    # The account and admin pages are rendered per user, so their components
    # are not in the initial layout.
    suppress_callback_exceptions=True,
)

# The map page stays mounted while other pages are shown, so its state (map,
# route, outlook) survives a visit to the account or admin page.
app.layout = html.Div([
    dcc.Location(id="url"),
    html.Div(id="page-twin", children=build_dashboard_layout()),
    html.Div(id="page-other", className="twin-app", hidden=True),
])
# Runs before anything downloads: applies the saved theme (no dark flash for
# light-theme users), and when the page is an emailed sign-in link, shows the
# "Signing you in" screen at once instead of a blank page and then the form.
# assets/firebase_login.js completes the link and removes the screen on error.
_EARLY_SCRIPT = """
<script>
(function () {
  try {
    var t = JSON.parse(localStorage.getItem("theme-store") || "null");
    if (t !== "light" && t !== "dark") {
      t = (window.matchMedia && matchMedia("(prefers-color-scheme: light)").matches) ? "light" : "dark";
    }
    document.documentElement.setAttribute("data-theme", t);
  } catch (e) { /* storage blocked: the theme callback sets it shortly */ }
  var q = location.search;
  if (q.indexOf("oobCode=") > -1 && q.indexOf("mode=signIn") > -1) {
    document.write('<div id="twin-signing-in" role="status" aria-live="polite"><div class="twin-signing-card">' +
      '<div class="twin-signing-spinner"></div><div class="twin-signing-title">Signing you in\\u2026</div>' +
      '<div class="twin-signing-sub">Confirming your email link</div></div></div>');
  }
})();
</script>
"""
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        """ + _EARLY_SCRIPT + """
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

register_callbacks(app)
register_auth_callbacks(app)
register_admin_callbacks(app)
server = app.server  # Flask WSGI server instance
init_auth(server)


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
    # Firebase sign-in treats "localhost" and "127.0.0.1" as different domains;
    # "localhost" is authorised by default, so point people there.
    shown_host = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    logger.info(f"   URL  : http://{shown_host}:{port}/")
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
