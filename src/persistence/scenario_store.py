"""
src.persistence.scenario_store
===============================
Nairobi Urban Flood Digital Twin — Scenario Run Persistence

PURPOSE
-------
The proposal's ERD (Chapter 3, Database Schema) commits to a persisted
"Simulation Scenarios" entity backed by PostgreSQL/PostGIS. Nothing in the
dashboard previously persisted anything — every run was recomputed from
scratch and discarded on the next callback. This module is a genuinely
working, scoped-down stand-in: a local SQLite table that records every
scenario the dashboard runs (rainfall input, resulting depth/area/
population impact, and per-zone risk levels), queryable for a "recent
scenarios" panel.

UPGRADE PATH
------------
The schema below maps directly onto a PostGIS table (swap sqlite3 for
psycopg2, add a `region_geom geometry(MultiPolygon, 4326)` column derived
from src.dashboard.components.map_3d's flood polygons) without changing
the calling code in src.dashboard.callbacks — only this module's
connection/DDL would need to change.

Failures here never take down the live dashboard: every public function
catches its own exceptions, logs a warning, and returns a safe default
(None / empty list) so a locked or missing database degrades the "recent
scenarios" panel rather than the whole app.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

DB_PATH = Path("data/scenarios.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scenario_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at_utc          TEXT NOT NULL,
    rainfall_mm_day     REAL NOT NULL,
    time_hour           REAL,
    display_mode        TEXT,
    max_depth_m         REAL,
    flooded_area_km2    REAL,
    est_affected_pop    INTEGER,
    critical_zone_count INTEGER,
    region_risks_json   TEXT
);
"""


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    try:
        with _get_conn() as conn:
            conn.execute(_SCHEMA)
    except sqlite3.Error as e:
        logger.warning(f"scenario_store: could not initialize {DB_PATH}: {e}")


def save_scenario_run(
    rainfall_mm_day: float,
    time_hour: float,
    display_mode: str,
    max_depth_m: float,
    flooded_area_km2: float,
    est_affected_pop: int,
    region_risks: dict,
) -> int | None:
    """Persist one scenario run. Returns the new row id, or None on failure."""
    critical_count = sum(1 for r in region_risks.values() if r.get("risk_level") in ("CRITICAL", "HIGH"))
    try:
        with _get_conn() as conn:
            conn.execute(_SCHEMA)
            cur = conn.execute(
                """INSERT INTO scenario_runs
                   (run_at_utc, rainfall_mm_day, time_hour, display_mode, max_depth_m,
                    flooded_area_km2, est_affected_pop, critical_zone_count, region_risks_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    float(rainfall_mm_day), float(time_hour), display_mode,
                    float(max_depth_m), float(flooded_area_km2), int(est_affected_pop),
                    critical_count, json.dumps(region_risks),
                ),
            )
            return cur.lastrowid
    except sqlite3.Error as e:
        logger.warning(f"scenario_store: failed to save scenario run: {e}")
        return None


def list_recent_scenarios(limit: int = 5) -> list[dict]:
    """Most recent scenario runs, newest first. Returns [] on failure."""
    try:
        with _get_conn() as conn:
            conn.execute(_SCHEMA)
            rows = conn.execute(
                "SELECT * FROM scenario_runs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
    except sqlite3.Error as e:
        logger.warning(f"scenario_store: failed to read recent scenarios: {e}")
        return []
