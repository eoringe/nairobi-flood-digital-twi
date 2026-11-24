"""
src.persistence.backend
=======================
Where the dashboard's own records are kept: PostgreSQL when DATABASE_URL is
set, otherwise SQLite files in TWIN_STATE_DIR.

WHY
---
On a cloud host the container's disk is wiped on every deploy, so accounts,
roles, flood reports, prediction history and saved scenarios were lost with
it. A managed PostgreSQL database outlives deploys. Locally, and in the
tests, SQLite needs nothing installed, so it stays the default.

HOW
---
The four stores (accounts, operations, reports, scenarios) are written in
plain SQLite SQL. `connect()` gives them either a sqlite3 connection or a
PostgreSQL connection that behaves like one for what they use:

* `?` and `:name` placeholders, rows readable by index and by column name
  (`dict(row)` works), `cursor.lastrowid` and `rowcount`;
* `executescript()` for their schemas, with SQLite column types mapped to
  PostgreSQL ones (an `INTEGER PRIMARY KEY` becomes an identity column);
* `INSERT OR IGNORE`, `PRAGMA table_info(...)` (used to add columns to
  older tables) and `PRAGMA foreign_keys` (always on in PostgreSQL);
* `with conn:` commits on success and rolls back on an error, and the
  connection then goes back to the pool.

Each store gets its own PostgreSQL schema (accounts, operations, reports,
scenarios), so their tables never collide, and one database serves all four.
"""

from __future__ import annotations

import atexit
import os
import re
import sqlite3
import threading
from pathlib import Path

from loguru import logger

#: Raised by either engine on a storage failure; callers that must not break
#: a request catch this tuple rather than sqlite3.Error alone.
DB_ERRORS: tuple[type[BaseException], ...] = (sqlite3.Error, OSError)
try:                                                    # the driver is only needed with DATABASE_URL
    import psycopg
    DB_ERRORS = (sqlite3.Error, OSError, psycopg.Error)
except ImportError:                                     # pragma: no cover
    psycopg = None


def database_url() -> str | None:
    """The PostgreSQL URL in DATABASE_URL, or None (SQLite) when unset or not PostgreSQL."""
    url = (os.environ.get("DATABASE_URL") or "").strip()
    return url if url.startswith(("postgres://", "postgresql://")) else None


def using_postgres() -> bool:
    return database_url() is not None


def connect(sqlite_path: Path | str, schema: str):
    """A connection for one store: PostgreSQL schema `schema`, or the SQLite file."""
    if using_postgres():
        return _PgConnection(_pool(schema), schema)
    conn = sqlite3.connect(sqlite_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def describe() -> str:
    """Where records are kept, for the start-up log (never the password)."""
    url = database_url()
    if not url:
        return f"SQLite files in {os.environ.get('TWIN_STATE_DIR', 'data')}"
    host = re.sub(r"^.*@", "", url).split("/")[0]
    return f"PostgreSQL at {host}"


# --------------------------------------------------------------- pooling --
_pools: dict[tuple[str, str], object] = {}
_pools_lock = threading.Lock()


def _pool(schema: str):
    if psycopg is None:
        raise RuntimeError("DATABASE_URL is set but the psycopg driver is not installed.")
    from psycopg_pool import ConnectionPool

    url = database_url()
    key = (url, schema)
    with _pools_lock:
        pool = _pools.get(key)
        if pool is None:
            with psycopg.connect(url, autocommit=True) as c:
                c.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')

            def configure(conn) -> None:
                # Unqualified table names resolve to this store's schema. Set per
                # session rather than in the URL, which some poolers reject.
                conn.execute(f'SET search_path TO "{schema}"')
                conn.commit()

            pool = ConnectionPool(url, min_size=1, max_size=int(os.environ.get("DB_POOL_SIZE", "5")),
                                  kwargs={"cursor_factory": psycopg.ClientCursor},
                                  configure=configure, open=True, name=f"twin-{schema}")
            _pools[key] = pool
            logger.info(f"Records for '{schema}' are kept in {describe()}.")
        return pool


def close_pools() -> None:
    """Close every pool (tests, and a clean shutdown)."""
    with _pools_lock:
        for pool in _pools.values():
            pool.close()
        _pools.clear()


# Without this a script that used the database waits for the pools' background
# threads on exit ("couldn't stop thread ... within 5.0 seconds").
atexit.register(close_pools)


# ---------------------------------------------------- SQLite -> PostgreSQL --
#: Tables whose first column is an INTEGER PRIMARY KEY: an INSERT into them
#: returns that column, which becomes cursor.lastrowid. Filled from each
#: store's schema as it is created (every store creates its schema on start).
_IDENTITY: dict[tuple[str, str], str] = {}

_CREATE = re.compile(r"CREATE TABLE IF NOT EXISTS\s+(\w+)\s*\(\s*(\w+)\s+INTEGER PRIMARY KEY", re.I)
_INSERT = re.compile(r"^\s*INSERT\s+(?:OR\s+IGNORE\s+)?INTO\s+(\w+)", re.I)
_NAMED = re.compile(r"(?<![:\w]):([A-Za-z_]\w*)")


def _ddl(sql: str) -> str:
    sql = re.sub(r"\bINTEGER PRIMARY KEY AUTOINCREMENT\b", "BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY",
                 sql, flags=re.I)
    sql = re.sub(r"\bINTEGER PRIMARY KEY\b", "BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY", sql, flags=re.I)
    sql = re.sub(r"\bREAL\b", "DOUBLE PRECISION", sql)
    return re.sub(r"\bBLOB\b", "BYTEA", sql)


def translate(sql: str, schema: str, has_params: bool = True) -> tuple[str, str | None]:
    """(PostgreSQL SQL, identity column to return or None) for one SQLite statement."""
    for table, pk in _CREATE.findall(sql):
        _IDENTITY[(schema, table.lower())] = pk
    if re.match(r"\s*CREATE\s", sql, re.I):
        return _ddl(sql), None
    returning = None
    ignore = re.match(r"\s*INSERT\s+OR\s+IGNORE\s+INTO", sql, re.I)
    if ignore:
        sql = re.sub(r"INSERT\s+OR\s+IGNORE\s+INTO", "INSERT INTO", sql, count=1, flags=re.I)
        sql = sql.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"
    insert = _INSERT.match(sql)
    if insert:
        returning = _IDENTITY.get((schema, insert.group(1).lower()))
        if returning:
            sql = sql.rstrip().rstrip(";") + f" RETURNING {returning}"
    if not has_params:                                  # no placeholder substitution happens
        return sql, returning
    sql = sql.replace("%", "%%")
    sql = _NAMED.sub(r"%(\1)s", sql)
    return sql.replace("?", "%s"), returning


def _params(params):
    """Booleans as 0/1: the stores keep flags in INTEGER columns, as SQLite does."""
    if params is None:
        return None
    if isinstance(params, dict):
        return {k: int(v) if isinstance(v, bool) else v for k, v in params.items()}
    return tuple(int(v) if isinstance(v, bool) else v for v in params)


class Row(tuple):
    """A result row readable as row[0] or row["column"], like sqlite3.Row."""

    def __new__(cls, values, names):
        row = super().__new__(cls, values)
        row._names = names
        return row

    def keys(self):
        return list(self._names)

    def __getitem__(self, key):
        if isinstance(key, str):
            return tuple.__getitem__(self, self._names.index(key))
        return tuple.__getitem__(self, key)


class _Cursor:
    def __init__(self, rows=None, names=(), rowcount=-1, lastrowid=None):
        self._rows = [Row(r, names) for r in (rows or [])]
        self.rowcount = rowcount
        self.lastrowid = lastrowid

    def fetchone(self):
        return self._rows.pop(0) if self._rows else None

    def fetchall(self):
        rows, self._rows = self._rows, []
        return rows

    def __iter__(self):
        return iter(self.fetchall())


class _PgConnection:
    """One pooled PostgreSQL connection, used like a sqlite3 connection."""

    def __init__(self, pool, schema: str) -> None:
        self._pool = pool
        self._schema = schema
        self._conn = pool.getconn()

    # sqlite3's `with conn:` commits or rolls back; here the connection also
    # goes back to the pool, since the stores never close their connections.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
        finally:
            self._release()
        return False

    def _release(self) -> None:
        if self._conn is not None:
            self._pool.putconn(self._conn)
            self._conn = None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.rollback()
        self._release()

    def commit(self) -> None:
        self._conn.commit()

    def __del__(self):                                  # a connection never used with `with`
        try:
            self.close()
        except Exception:                               # noqa: BLE001
            pass

    def execute(self, sql: str, params=None) -> _Cursor:
        pragma = re.match(r"\s*PRAGMA\s+(\w+)\s*(?:\((\w+)\))?", sql, re.I)
        if pragma:
            if pragma.group(1).lower() == "table_info":
                with self._conn.cursor() as cur:
                    cur.execute("SELECT ordinal_position - 1, column_name FROM information_schema.columns "
                                "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
                                (self._schema, pragma.group(2).lower()))
                    return _Cursor(cur.fetchall(), ("cid", "name"))
            return _Cursor()                            # foreign_keys and the like: PostgreSQL defaults
        pg_sql, returning = translate(sql, self._schema, params is not None)
        with self._conn.cursor() as cur:
            cur.execute(pg_sql, _params(params))
            names = tuple(d.name for d in cur.description) if cur.description else ()
            rows = cur.fetchall() if cur.description else []
            lastrowid = None
            if returning:
                lastrowid = rows[0][0] if rows else None
                rows, names = [], ()
            return _Cursor(rows, names, cur.rowcount, lastrowid)

    def executemany(self, sql: str, seq) -> _Cursor:
        pg_sql, returning = translate(sql, self._schema)
        if returning:
            pg_sql = pg_sql[: pg_sql.rindex(" RETURNING ")]
        seq = [_params(p) for p in seq]
        with self._conn.cursor() as cur:
            if seq:
                cur.executemany(pg_sql, seq)
            return _Cursor(rowcount=cur.rowcount)

    def executescript(self, script: str) -> None:
        for statement in script.split(";"):
            if statement.strip():
                self.execute(statement)
