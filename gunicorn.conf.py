"""
Gunicorn configuration for the dashboard container (local Docker and Railway).

Everything is read from the environment so the same image runs anywhere:

    PORT          port to listen on. Railway injects it; defaults to 8050.
    WEB_THREADS   request threads in the single worker (default 8).

One worker process only: the flood model, road graph and forecast caches live in
process memory. More workers would each load their own copy (~650 MB apiece) and
not share caches; threads share them.
"""

import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8050')}"
workers = 1
worker_class = "gthread"
threads = int(os.environ.get("WEB_THREADS", "8"))

# A cold replay download or a city-wide flood render can take tens of seconds.
timeout = 180
graceful_timeout = 30
keepalive = 5

accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")

# Trust Railway's proxy for the client address and scheme (HTTPS).
forwarded_allow_ips = "*"
