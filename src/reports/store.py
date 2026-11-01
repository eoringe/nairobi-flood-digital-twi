"""
src.reports.store
=================
Flood reports from people on the ground, kept in a local SQLite file.

WHY
---
The model forecasts where flooding is likely; it cannot see a drain blocked
this morning or water already standing in a street. Like reporting an accident
on a navigation app, signed-in users report flooding where they are, and the
map shows it to everyone within seconds.

HOW A REPORT LIVES
------------------
* A report is a point, a kind (water on the road, road impassable, homes
  flooded) and an optional short note. Reporters are never shown publicly.
* It expires REPORT_TTL after it was made, unless people confirm it: each
  "still flooded" keeps it for at least CONFIRM_TTL from that moment.
* A report within MERGE_RADIUS_M of an active one counts as a confirmation of
  it, so one flooded junction shows as one report confirmed by many.
* "Cleared" from the reporter, or from CLEARS_TO_CLOSE other people (more than
  have confirmed it), closes it. Responders and administrators can verify a
  report or remove it.
* At most RATE_LIMIT reports per person per hour.

Every create, vote, verification and removal is also written to the audit log
by the caller (src.dashboard.reports_ui).
"""

from __future__ import annotations

import math
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

KINDS = {
    "water": "Water on the road",
    "impassable": "Road impassable",
    "homes": "Homes flooded",
}
REPORT_TTL = timedelta(hours=6)
CONFIRM_TTL = timedelta(hours=3)
MERGE_RADIUS_M = 150.0
MERGE_WINDOW = timedelta(hours=2)
CLEARS_TO_CLOSE = 2
RATE_LIMIT = 5
NOTE_MAX = 140

_SCHEMA = """
CREATE TABLE IF NOT EXISTS flood_reports (
    report_id     INTEGER PRIMARY KEY,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    expires_at    TEXT NOT NULL,
    user_id       INTEGER NOT NULL,
    lat           REAL NOT NULL,
    lon           REAL NOT NULL,
    place         TEXT NOT NULL,
    kind          TEXT NOT NULL CHECK (kind IN ('water','impassable','homes')),
    note          TEXT,
    status        TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','cleared','removed')),
    verified_by   INTEGER,
    confirmations INTEGER NOT NULL DEFAULT 0,
    clears        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_reports_status ON flood_reports(status, expires_at);
CREATE TABLE IF NOT EXISTS report_votes (
    report_id  INTEGER NOT NULL REFERENCES flood_reports(report_id) ON DELETE CASCADE,
    user_id    INTEGER NOT NULL,
    vote       TEXT NOT NULL CHECK (vote IN ('still','cleared')),
    voted_at   TEXT NOT NULL,
    PRIMARY KEY (report_id, user_id)
);
"""


class ReportError(Exception):
    """A report the store refuses, with a message safe to show the user."""


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


def _metres(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    kx = 111320.0 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot((lat2 - lat1) * 110574.0, (lon2 - lon1) * kx)


class ReportStore:
    """Flood reports and the votes on them, behind one lock."""

    def __init__(self, path: Path | str, now=None) -> None:
        self.path = Path(path)
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ----------------------------------------------------------- reading --
    def get(self, report_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM flood_reports WHERE report_id=?", (report_id,)).fetchone()
        return dict(row) if row else None

    def active(self) -> list[dict]:
        """Reports still standing, newest first."""
        now = _iso(self._now())
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM flood_reports WHERE status='active' AND expires_at > ? "
                "ORDER BY updated_at DESC", (now,)).fetchall()
        return [dict(r) for r in rows]

    def history(self, start: str | None = None, end: str | None = None, limit: int = 2000) -> list[dict]:
        """
        Reports created between `start` and `end` (ISO dates or times), newest
        first, for incident history. Removed reports are left out. A report past
        its expiry reads as 'expired'. Reporter and moderator ids are not
        returned: reporters are never shown (FR-14).
        """
        q, args = "SELECT * FROM flood_reports WHERE status != 'removed'", []
        if start:
            q += " AND created_at >= ?"; args.append(str(start))
        if end:
            q += " AND created_at <= ?"; args.append(str(end) + ("T23:59:59" if len(str(end)) == 10 else ""))
        now = _iso(self._now())
        with self._conn() as conn:
            rows = [dict(r) for r in conn.execute(q + " ORDER BY created_at DESC LIMIT ?", (*args, limit))]
        for r in rows:
            if r["status"] == "active" and r["expires_at"] <= now:
                r["status"] = "expired"
            r["verified"] = r.pop("verified_by") is not None
            r.pop("user_id", None)
        return rows

    def vote_of(self, report_id: int, user_id: int) -> str | None:
        with self._conn() as conn:
            row = conn.execute("SELECT vote FROM report_votes WHERE report_id=? AND user_id=?",
                               (report_id, user_id)).fetchone()
        return row["vote"] if row else None

    def count_since(self, hours: float = 24.0) -> int:
        since = _iso(self._now() - timedelta(hours=hours))
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM flood_reports WHERE created_at > ?",
                                (since,)).fetchone()[0]

    # ----------------------------------------------------------- writing --
    def add(self, user_id: int, lat: float, lon: float, kind: str, place: str,
            note: str | None = None) -> tuple[dict, bool]:
        """
        Record a report. Returns (report, merged): merged is True when it was
        counted as a confirmation of an active report close by.
        """
        if kind not in KINDS:
            raise ReportError("Choose what kind of flooding you see.")
        note = (note or "").strip()[:NOTE_MAX] or None
        now = self._now()
        with self._lock, self._conn() as conn:
            recent = conn.execute(
                "SELECT COUNT(*) FROM flood_reports WHERE user_id=? AND created_at > ?",
                (user_id, _iso(now - timedelta(hours=1)))).fetchone()[0]
            if recent >= RATE_LIMIT:
                raise ReportError(f"You can send {RATE_LIMIT} reports an hour. Please try again later.")
            near = conn.execute(
                "SELECT * FROM flood_reports WHERE status='active' AND expires_at > ? AND created_at > ?",
                (_iso(now), _iso(now - MERGE_WINDOW))).fetchall()
            for row in near:
                if _metres(lat, lon, row["lat"], row["lon"]) <= MERGE_RADIUS_M:
                    # Counted as a confirmation; a more serious kind wins.
                    order = list(KINDS)
                    kind_now = max(row["kind"], kind, key=order.index)
                    if row["user_id"] == user_id:
                        # The reporter again: refreshed, but not a confirmation.
                        expires = max(datetime.fromisoformat(row["expires_at"]), now + CONFIRM_TTL)
                        conn.execute("UPDATE flood_reports SET updated_at=?, expires_at=? WHERE report_id=?",
                                     (_iso(now), _iso(expires), row["report_id"]))
                    else:
                        self._confirm(conn, row["report_id"], user_id, now)
                    conn.execute("UPDATE flood_reports SET kind=? WHERE report_id=?", (kind_now, row["report_id"]))
                    return dict(conn.execute("SELECT * FROM flood_reports WHERE report_id=?",
                                             (row["report_id"],)).fetchone()), True
            cur = conn.execute(
                "INSERT INTO flood_reports (created_at, updated_at, expires_at, user_id, lat, lon, place, kind, note) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (_iso(now), _iso(now), _iso(now + REPORT_TTL), user_id, round(lat, 6), round(lon, 6),
                 place, kind, note))
            return dict(conn.execute("SELECT * FROM flood_reports WHERE report_id=?",
                                     (cur.lastrowid,)).fetchone()), False

    def _confirm(self, conn, report_id: int, user_id: int, now: datetime) -> None:
        conn.execute(
            "INSERT INTO report_votes (report_id, user_id, vote, voted_at) VALUES (?,?,?,?) "
            "ON CONFLICT(report_id, user_id) DO UPDATE SET vote=excluded.vote, voted_at=excluded.voted_at",
            (report_id, user_id, "still", _iso(now)))
        self._recount(conn, report_id)
        row = conn.execute("SELECT expires_at FROM flood_reports WHERE report_id=?", (report_id,)).fetchone()
        expires = max(datetime.fromisoformat(row["expires_at"]), now + CONFIRM_TTL)
        conn.execute("UPDATE flood_reports SET updated_at=?, expires_at=? WHERE report_id=?",
                     (_iso(now), _iso(expires), report_id))

    def _recount(self, conn, report_id: int) -> None:
        conn.execute(
            "UPDATE flood_reports SET "
            "confirmations=(SELECT COUNT(*) FROM report_votes WHERE report_id=? AND vote='still'), "
            "clears=(SELECT COUNT(*) FROM report_votes WHERE report_id=? AND vote='cleared') "
            "WHERE report_id=?", (report_id, report_id, report_id))

    def vote(self, report_id: int, user_id: int, vote: str) -> dict:
        """'still' (still flooded) or 'cleared'. Returns the report afterwards."""
        if vote not in ("still", "cleared"):
            raise ReportError("Unknown vote.")
        now = self._now()
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT * FROM flood_reports WHERE report_id=?", (report_id,)).fetchone()
            if row is None or row["status"] != "active" or row["expires_at"] <= _iso(now):
                raise ReportError("This report has already closed.")
            if vote == "still":
                if row["user_id"] == user_id:
                    raise ReportError("You made this report; others can confirm it.")
                self._confirm(conn, report_id, user_id, now)
            else:
                conn.execute(
                    "INSERT INTO report_votes (report_id, user_id, vote, voted_at) VALUES (?,?,?,?) "
                    "ON CONFLICT(report_id, user_id) DO UPDATE SET vote=excluded.vote, voted_at=excluded.voted_at",
                    (report_id, user_id, "cleared", _iso(now)))
                self._recount(conn, report_id)
                after = conn.execute("SELECT * FROM flood_reports WHERE report_id=?", (report_id,)).fetchone()
                closes = (row["user_id"] == user_id
                          or (after["clears"] >= CLEARS_TO_CLOSE and after["clears"] > after["confirmations"]))
                conn.execute("UPDATE flood_reports SET updated_at=?, status=? WHERE report_id=?",
                             (_iso(now), "cleared" if closes else "active", report_id))
            return dict(conn.execute("SELECT * FROM flood_reports WHERE report_id=?", (report_id,)).fetchone())

    def verify(self, report_id: int, moderator_id: int) -> dict:
        with self._lock, self._conn() as conn:
            conn.execute("UPDATE flood_reports SET verified_by=?, updated_at=? WHERE report_id=? AND status='active'",
                         (moderator_id, _iso(self._now()), report_id))
            row = conn.execute("SELECT * FROM flood_reports WHERE report_id=?", (report_id,)).fetchone()
        if row is None:
            raise ReportError("Report not found.")
        return dict(row)

    def remove(self, report_id: int, moderator_id: int) -> dict:
        with self._lock, self._conn() as conn:
            conn.execute("UPDATE flood_reports SET status='removed', updated_at=? WHERE report_id=?",
                         (_iso(self._now()), report_id))
            row = conn.execute("SELECT * FROM flood_reports WHERE report_id=?", (report_id,)).fetchone()
        if row is None:
            raise ReportError("Report not found.")
        return dict(row)


# ------------------------------------------------------------------ shared --
_store: ReportStore | None = None
_store_lock = threading.Lock()


def default_path() -> Path:
    return Path(os.environ.get("TWIN_STATE_DIR", "data")) / "reports.db"


def get_reports() -> ReportStore:
    """The application's report store, created on first use in TWIN_STATE_DIR."""
    global _store
    with _store_lock:
        if _store is None:
            _store = ReportStore(default_path())
        return _store


def set_reports(store: ReportStore | None) -> None:
    """Swap the shared store (tests use a temporary file)."""
    global _store
    with _store_lock:
        _store = store
