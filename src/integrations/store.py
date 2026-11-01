"""
src.integrations.store
======================
Dashboard-owned records (specification §8): data source status, the call log
behind "failed requests", prediction runs, map layer and user preferences,
report definitions and generated reports, and administrative boundaries.

Operational records stay in the systems that own them (§8: "should not
unnecessarily duplicate operational records"). What is kept here is either the
dashboard's own configuration or the history it needs to show trends and
compare predictions with what happened.

SQLite in TWIN_STATE_DIR, like the account store. The SQL is plain, and
geometry is stored as WKT text, so moving to PostgreSQL/PostGIS changes the
connection and the geometry column type, not the callers.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS data_source_status (
    source_key     TEXT PRIMARY KEY,
    last_success   TEXT,
    last_failure   TEXT,
    last_error     TEXT,
    last_latency_ms REAL,
    last_observed  TEXT
);
CREATE TABLE IF NOT EXISTS data_source_calls (
    call_id     INTEGER PRIMARY KEY,
    source_key  TEXT NOT NULL,
    at          TEXT NOT NULL,
    ok          INTEGER NOT NULL,
    latency_ms  REAL,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_calls_source ON data_source_calls(source_key, at);
CREATE TABLE IF NOT EXISTS prediction_runs (
    run_id         INTEGER PRIMARY KEY,
    issued_at      TEXT NOT NULL,
    source         TEXT NOT NULL,
    model_version  TEXT NOT NULL,
    horizon_h      INTEGER NOT NULL,
    rain_72h_mm    REAL,
    chance         REAL,
    chance_call    TEXT,
    city_level     INTEGER NOT NULL,
    place_levels   TEXT NOT NULL,
    headline       TEXT,
    UNIQUE (issued_at, source)
);
CREATE TABLE IF NOT EXISTS map_layers (
    layer_key    TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT NOT NULL,
    source       TEXT NOT NULL,
    category     TEXT NOT NULL,
    default_on   INTEGER NOT NULL DEFAULT 0,
    min_role     TEXT
);
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id   INTEGER PRIMARY KEY,
    prefs     TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS report_definitions (
    report_key  TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT NOT NULL,
    formats     TEXT NOT NULL,
    permission  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS generated_reports (
    report_id   INTEGER PRIMARY KEY,
    report_key  TEXT NOT NULL,
    user_id     INTEGER NOT NULL,
    created_at  TEXT NOT NULL,
    format      TEXT NOT NULL,
    filters     TEXT NOT NULL,
    rows        INTEGER
);
CREATE TABLE IF NOT EXISTS geographic_boundaries (
    boundary_id  TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    level        TEXT NOT NULL,
    parent_id    TEXT,
    wkt          TEXT NOT NULL,
    source       TEXT NOT NULL
);
"""

#: Calls older than this are pruned from the call log.
CALL_LOG_DAYS = 7


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iso(t) -> str | None:
    if t is None:
        return None
    if isinstance(t, str):
        return t
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone(timedelta(hours=3)))
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


class OpsStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------- source status --
    def record_call(self, source_key: str, ok: bool, latency_ms: float | None = None,
                    error: str | None = None, observed_at=None) -> None:
        now = _now()
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO data_source_calls (source_key, at, ok, latency_ms, error) VALUES (?,?,?,?,?)",
                      (source_key, now, int(ok), latency_ms, error))
            c.execute("INSERT OR IGNORE INTO data_source_status (source_key) VALUES (?)", (source_key,))
            if ok:
                c.execute("UPDATE data_source_status SET last_success=?, last_latency_ms=?, "
                          "last_observed=COALESCE(?, last_observed) WHERE source_key=?",
                          (now, latency_ms, _iso(observed_at), source_key))
            else:
                c.execute("UPDATE data_source_status SET last_failure=?, last_error=?, last_latency_ms=? "
                          "WHERE source_key=?", (now, error, latency_ms, source_key))
            cutoff = (datetime.now(timezone.utc) - timedelta(days=CALL_LOG_DAYS)).isoformat(timespec="seconds")
            c.execute("DELETE FROM data_source_calls WHERE at < ?", (cutoff,))

    def status(self, source_key: str) -> dict:
        with self._conn() as c:
            row = c.execute("SELECT * FROM data_source_status WHERE source_key=?", (source_key,)).fetchone()
            since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(timespec="seconds")
            fails, calls = c.execute(
                "SELECT COALESCE(SUM(1-ok),0), COUNT(*) FROM data_source_calls WHERE source_key=? AND at>=?",
                (source_key, since)).fetchone()
        out = dict(row) if row else {"source_key": source_key}
        out.update({"failures_24h": int(fails), "calls_24h": int(calls)})
        return out

    def failed_calls(self, limit: int = 50, source_key: str | None = None) -> list[dict]:
        q = "SELECT * FROM data_source_calls WHERE ok=0"
        args: list = []
        if source_key:
            q += " AND source_key=?"
            args.append(source_key)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY at DESC LIMIT ?", (*args, limit))]

    # ----------------------------------------------------- prediction runs --
    def save_prediction_run(self, run: dict) -> int | None:
        """One row per outlook; the same issue time and source is stored once."""
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO prediction_runs (issued_at, source, model_version, horizon_h, rain_72h_mm, "
                "chance, chance_call, city_level, place_levels, headline) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (_iso(run["issued_at"]), run["source"], run["model_version"], run["horizon_h"],
                 run.get("rain_72h_mm"), run.get("chance"), run.get("chance_call"), run["city_level"],
                 json.dumps(run["place_levels"]), run.get("headline")))
            return cur.lastrowid if cur.rowcount else None

    def prediction_runs(self, limit: int = 50, source: str | None = None, since: str | None = None,
                        until: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM prediction_runs WHERE 1=1", []
        if source:
            q += " AND source=?"; args.append(source)
        if since:
            q += " AND issued_at>=?"; args.append(since)
        if until:
            q += " AND issued_at<=?"; args.append(until)
        with self._conn() as c:
            rows = [dict(r) for r in c.execute(q + " ORDER BY issued_at DESC LIMIT ?", (*args, limit))]
        for r in rows:
            r["place_levels"] = json.loads(r["place_levels"])
        return rows

    def prediction_run(self, run_id: int) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM prediction_runs WHERE run_id=?", (run_id,)).fetchone()
        if r is None:
            return None
        r = dict(r)
        r["place_levels"] = json.loads(r["place_levels"])
        return r

    # --------------------------------------------------- user preferences --
    def preferences(self, user_id: int) -> dict:
        with self._conn() as c:
            r = c.execute("SELECT prefs FROM user_preferences WHERE user_id=?", (user_id,)).fetchone()
        return json.loads(r["prefs"]) if r else {}

    def set_preferences(self, user_id: int, updates: dict) -> dict:
        prefs = {**self.preferences(user_id), **updates}
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO user_preferences (user_id, prefs, updated_at) VALUES (?,?,?) "
                      "ON CONFLICT(user_id) DO UPDATE SET prefs=excluded.prefs, updated_at=excluded.updated_at",
                      (user_id, json.dumps(prefs), _now()))
        return prefs

    # -------------------------------------------------- layers and reports --
    def upsert_layers(self, layers: list[dict]) -> None:
        with self._lock, self._conn() as c:
            c.executemany(
                "INSERT INTO map_layers (layer_key, name, description, source, category, default_on, min_role) "
                "VALUES (:layer_key,:name,:description,:source,:category,:default_on,:min_role) "
                "ON CONFLICT(layer_key) DO UPDATE SET name=excluded.name, description=excluded.description, "
                "source=excluded.source, category=excluded.category, default_on=excluded.default_on, "
                "min_role=excluded.min_role", [{"min_role": None, **l} for l in layers])

    def layers(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM map_layers ORDER BY category, name")]

    def upsert_report_definitions(self, defs: list[dict]) -> None:
        with self._lock, self._conn() as c:
            c.executemany(
                "INSERT INTO report_definitions (report_key, name, description, formats, permission) "
                "VALUES (:report_key,:name,:description,:formats,:permission) ON CONFLICT(report_key) DO UPDATE "
                "SET name=excluded.name, description=excluded.description, formats=excluded.formats, "
                "permission=excluded.permission", defs)

    def report_definitions(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM report_definitions ORDER BY name")]

    def record_report(self, report_key: str, user_id: int, fmt: str, filters: dict, rows: int) -> int:
        with self._lock, self._conn() as c:
            return c.execute("INSERT INTO generated_reports (report_key, user_id, created_at, format, filters, rows) "
                             "VALUES (?,?,?,?,?,?)", (report_key, user_id, _now(), fmt, json.dumps(filters), rows)
                             ).lastrowid

    def generated_reports(self, limit: int = 30) -> list[dict]:
        with self._conn() as c:
            rows = [dict(r) for r in c.execute("SELECT * FROM generated_reports ORDER BY created_at DESC LIMIT ?",
                                               (limit,))]
        for r in rows:
            r["filters"] = json.loads(r["filters"])
        return rows

    # ---------------------------------------------------------- boundaries --
    def replace_boundaries(self, rows: list[dict]) -> None:
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM geographic_boundaries")
            c.executemany("INSERT INTO geographic_boundaries (boundary_id, name, level, parent_id, wkt, source) "
                          "VALUES (:boundary_id,:name,:level,:parent_id,:wkt,:source)", rows)

    def boundaries(self, level: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM geographic_boundaries", []
        if level:
            q += " WHERE level=?"; args.append(level)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY level, name", args)]


_store: OpsStore | None = None
_store_lock = threading.Lock()


def default_path() -> Path:
    return Path(os.environ.get("TWIN_STATE_DIR", "data")) / "operations.db"


def get_ops_store() -> OpsStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = OpsStore(default_path())
        return _store


def set_ops_store(store: OpsStore | None) -> None:
    """Swap the shared store (tests use a temporary file)."""
    global _store
    with _store_lock:
        _store = store
