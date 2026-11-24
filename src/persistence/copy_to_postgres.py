"""
Copy records from the SQLite files in TWIN_STATE_DIR into the PostgreSQL
database in DATABASE_URL, once, when moving a deployment to PostgreSQL.

    DATABASE_URL=postgresql://... TWIN_STATE_DIR=data python -m src.persistence.copy_to_postgres

Each store's tables are created first (by opening the store), then every row
is copied with its original id, and the id counters are moved past the
highest copied id. Rows whose key already exists are left as they are, so
running it twice copies nothing twice. Missing SQLite files are skipped.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

#: SQLite file -> (PostgreSQL schema, tables in an order that respects foreign keys)
STORES = {
    "accounts.db": ("accounts", ["users", "audit_log", "alert_subscriptions"]),
    "operations.db": ("operations", ["data_source_status", "data_source_calls", "prediction_runs", "map_layers",
                                     "user_preferences", "report_definitions", "generated_reports",
                                     "geographic_boundaries"]),
    "reports.db": ("reports", ["flood_reports", "report_votes"]),
    "scenarios.db": ("scenarios", ["scenario_runs"]),
}


def _open_stores(state: Path) -> None:
    """Create every schema and table in PostgreSQL through the stores themselves."""
    from src.auth.store import AuthStore
    from src.integrations.store import OpsStore
    from src.persistence import scenario_store
    from src.reports.store import ReportStore
    AuthStore(state / "accounts.db")
    OpsStore(state / "operations.db")
    ReportStore(state / "reports.db")
    scenario_store.init_db()


def copy_all(state: Path) -> dict[str, int]:
    import psycopg
    from src.persistence import backend

    if not backend.using_postgres():
        raise SystemExit("Set DATABASE_URL to the PostgreSQL database to copy into.")
    _open_stores(state)
    copied: dict[str, int] = {}
    with psycopg.connect(backend.database_url()) as pg:
        for filename, (schema, tables) in STORES.items():
            path = state / filename
            if not path.exists():
                continue
            src = sqlite3.connect(path)
            try:
                have = {r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                for table in tables:
                    if table not in have:
                        continue
                    pg_cols = {r[0] for r in pg.execute(
                        "SELECT column_name FROM information_schema.columns WHERE table_schema=%s AND table_name=%s",
                        (schema, table))}
                    cols = [r[1] for r in src.execute(f"PRAGMA table_info({table})") if r[1] in pg_cols]
                    rows = src.execute(f"SELECT {', '.join(cols)} FROM {table}").fetchall()
                    if rows:
                        marks = ", ".join(["%s"] * len(cols))
                        with pg.cursor() as cur:
                            cur.executemany(f'INSERT INTO "{schema}".{table} ({", ".join(cols)}) VALUES ({marks}) '
                                            f"ON CONFLICT DO NOTHING", rows)
                    copied[f"{schema}.{table}"] = len(rows)
                    # Identity columns: new ids continue after the copied ones.
                    for (col,) in pg.execute(
                            "SELECT column_name FROM information_schema.columns WHERE table_schema=%s "
                            "AND table_name=%s AND is_identity='YES'", (schema, table)):
                        pg.execute(f'SELECT setval(pg_get_serial_sequence(\'"{schema}".{table}\', \'{col}\'), '
                                   f'GREATEST((SELECT COALESCE(MAX({col}), 0) FROM "{schema}".{table}), 1))')
            finally:
                src.close()
        pg.commit()
    return copied


def main() -> None:
    state = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TWIN_STATE_DIR", "data"))
    copied = copy_all(state)
    if not copied:
        print(f"No SQLite records found in {state}.")
    for table, n in copied.items():
        print(f"{table}: {n} rows")


if __name__ == "__main__":
    main()
