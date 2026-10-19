"""
src.auth.store
==============
User accounts, the audit log and alert subscriptions.

Implements the identity, audit and subscription entities of the Chapter 4 ERD
(`users`, `audit_log`, `alert_subscriptions` in src/persistence/schema.sql) in
a local SQLite file beside the scenario history, so the dashboard runs with
nothing installed. The column names match schema.sql.

IDENTITY
--------
An account is identified by its email. There are no passwords: people sign
in with Google or a link emailed to them (both through Firebase), and the
verified email opens the account. The username is generated from the email
and is only a short display handle.

SECOND FACTOR
-------------
Administrators also enrol an authenticator app (src/auth/totp.py): the TOTP
secret, the last accepted time step (to refuse replays) and one-time recovery
codes, stored only as salted hashes, are kept here. The password_hash column
remains in the table for older databases but is always empty: hashes left from
before password sign-in was removed are erased on start-up.

AUDIT
-----
`audit_log` is append-only (FR-15, DR-09): this module inserts into it and
never updates or deletes a row.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.auth import totp

ROLES = ("citizen", "planner", "responder", "admin")
#: How the account was first created. "local" survives only in older databases,
#: from when password sign-in existed.
PROVIDERS = ("google", "email_link")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_HANDLE_CHARS = re.compile(r"[^a-z0-9._-]")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id         INTEGER PRIMARY KEY,
    username        TEXT NOT NULL UNIQUE,
    email           TEXT UNIQUE,
    display_name    TEXT,
    password_hash   TEXT,
    auth_provider   TEXT NOT NULL DEFAULT 'local'
                    CHECK (auth_provider IN ('local','google','email_link')),
    role            TEXT NOT NULL DEFAULT 'citizen'
                    CHECK (role IN ('citizen','planner','responder','admin')),
    active          INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL,
    last_login_at   TEXT,
    totp_secret     TEXT,
    totp_last_step  INTEGER,
    recovery_codes  TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id         INTEGER PRIMARY KEY,
    occurred_at      TEXT NOT NULL,
    user_id          INTEGER REFERENCES users(user_id),
    action           TEXT NOT NULL,
    model_version_id INTEGER,
    detail           TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_log(occurred_at);
CREATE TABLE IF NOT EXISTS alert_subscriptions (
    subscription_id INTEGER PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    sub_county      TEXT NOT NULL,
    active          INTEGER NOT NULL DEFAULT 1,
    label           TEXT,
    UNIQUE (user_id, sub_county)
);
"""

#: Columns added after the first release, for databases created before them.
_ADDED_COLUMNS = {"totp_secret": "TEXT", "totp_last_step": "INTEGER", "recovery_codes": "TEXT"}
_PRIVATE = ("password_hash", "totp_secret", "totp_last_step", "recovery_codes")


class AuthError(Exception):
    """A request the store refuses, with a message safe to show the user."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _public(row: sqlite3.Row | None) -> dict | None:
    """A user row without secrets."""
    if row is None:
        return None
    user = dict(row)
    user["totp_enabled"] = bool(user.get("totp_secret"))
    for key in _PRIVATE:
        user.pop(key, None)
    user["active"] = bool(user["active"])
    return user


def normalise_email(email: str | None) -> str:
    value = (email or "").strip().lower()
    if not _EMAIL.match(value):
        raise AuthError("Enter a valid email address.")
    return value


class AuthStore:
    """All account, audit and subscription storage, behind one lock."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            have = {r[1] for r in conn.execute("PRAGMA table_info(users)")}
            for col, kind in _ADDED_COLUMNS.items():
                if col not in have:
                    conn.execute(f"ALTER TABLE users ADD COLUMN {col} {kind}")
            subs_cols = {r[1] for r in conn.execute("PRAGMA table_info(alert_subscriptions)")}
            if "label" not in subs_cols:
                conn.execute("ALTER TABLE alert_subscriptions ADD COLUMN label TEXT")
        self._widen_provider_check()
        with self._conn() as conn:
            # Password sign-in was removed; no password hash is kept.
            conn.execute("UPDATE users SET password_hash = NULL WHERE password_hash IS NOT NULL")

    def _widen_provider_check(self) -> None:
        """
        Databases created before email-link sign-in only allow 'local' and
        'google'. SQLite cannot alter a CHECK, so the table is rebuilt once with
        every row copied unchanged.
        """
        conn = sqlite3.connect(self.path, timeout=5.0)
        try:
            ddl = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='users'").fetchone()[0]
            if "email_link" in ddl:
                return
            cols = [r[1] for r in conn.execute("PRAGMA table_info(users)")]
            new_ddl = _SCHEMA.split("CREATE TABLE IF NOT EXISTS audit_log")[0].replace(
                "CREATE TABLE IF NOT EXISTS users", "CREATE TABLE users_new")
            conn.execute("PRAGMA foreign_keys = OFF")
            with conn:
                conn.execute(new_ddl)
                names = ", ".join(c for c in cols)
                conn.execute(f"INSERT INTO users_new ({names}) SELECT {names} FROM users")
                conn.execute("DROP TABLE users")
                conn.execute("ALTER TABLE users_new RENAME TO users")
            logger.info("Account store upgraded to allow email-link sign-in.")
        finally:
            conn.close()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _row(self, conn, user_id: int) -> sqlite3.Row | None:
        return conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()

    def _free_username(self, conn, email: str) -> str:
        base = _HANDLE_CHARS.sub("", email.split("@")[0])[:24] or "user"
        if len(base) < 3:
            base = (base + "user")[:24]
        name, n = base, 1
        while conn.execute("SELECT 1 FROM users WHERE username=?", (name,)).fetchone():
            n += 1
            name = f"{base}{n}"
        return name

    # --------------------------------------------------------------- users --
    def create_user(self, email: str, role: str = "citizen", *, provider: str = "email_link",
                    display_name: str | None = None) -> dict:
        """
        A new account. Whatever `provider` records (how it was first created), the
        person signs in with Google or an emailed link for this address.
        """
        email = normalise_email(email)
        if role not in ROLES:
            raise AuthError(f"Unknown role '{role}'.")
        if provider not in PROVIDERS:
            raise AuthError(f"Unknown sign-in method '{provider}'.")
        with self._lock, self._conn() as conn:
            if conn.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
                raise AuthError("An account with that email already exists.")
            cur = conn.execute(
                "INSERT INTO users (username, email, display_name, auth_provider, role, active, created_at)"
                " VALUES (?,?,?,?,?,1,?)",
                (self._free_username(conn, email), email, display_name, provider, role, _utc_now()),
            )
            return _public(self._row(conn, cur.lastrowid))

    def get_user(self, user_id: int) -> dict | None:
        with self._conn() as conn:
            return _public(self._row(conn, user_id))

    def get_by_email(self, email: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE email=?", ((email or "").strip().lower(),)).fetchone()
            return _public(row)

    def find(self, identifier: str) -> dict | None:
        """By email, or by username (the command line accepts either)."""
        ident = (identifier or "").strip().lower()
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE email=? OR username=?", (ident, ident)).fetchone()
            return _public(row)

    def touch_login(self, user_id: int) -> None:
        with self._lock, self._conn() as conn:
            conn.execute("UPDATE users SET last_login_at=? WHERE user_id=?", (_utc_now(), user_id))

    def set_display_name(self, user_id: int, name: str | None) -> None:
        if not name:
            return
        with self._lock, self._conn() as conn:
            conn.execute("UPDATE users SET display_name=? WHERE user_id=? AND display_name IS NULL",
                         (name[:80], user_id))

    def list_users(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM users ORDER BY role, email").fetchall()
            return [_public(r) for r in rows]

    def count_active_admins(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM users WHERE role='admin' AND active=1").fetchone()[0]

    def update_user(self, user_id: int, *, role: str | None = None, active: bool | None = None) -> dict:
        """Change a role or active flag, refusing to leave no active admin."""
        if role is not None and role not in ROLES:
            raise AuthError(f"Unknown role '{role}'.")
        with self._lock, self._conn() as conn:
            row = self._row(conn, user_id)
            if row is None:
                raise AuthError("No such user.")
            was_admin = row["role"] == "admin" and row["active"]
            new_role = role if role is not None else row["role"]
            new_active = int(active) if active is not None else row["active"]
            if was_admin and not (new_role == "admin" and new_active):
                admins = conn.execute(
                    "SELECT COUNT(*) FROM users WHERE role='admin' AND active=1").fetchone()[0]
                if admins <= 1:
                    raise AuthError("This is the last active administrator; promote another first.")
            conn.execute("UPDATE users SET role=?, active=? WHERE user_id=?", (new_role, new_active, user_id))
            return _public(self._row(conn, user_id))

    # ------------------------------------------------------- second factor --
    def enable_totp(self, user_id: int, secret: str, recovery_codes: list[str], first_step: int) -> None:
        hashes = [totp.hash_recovery_code(c) for c in recovery_codes]
        with self._lock, self._conn() as conn:
            conn.execute("UPDATE users SET totp_secret=?, totp_last_step=?, recovery_codes=? WHERE user_id=?",
                         (secret, first_step, json.dumps(hashes), user_id))

    def disable_totp(self, user_id: int) -> None:
        with self._lock, self._conn() as conn:
            conn.execute("UPDATE users SET totp_secret=NULL, totp_last_step=NULL, recovery_codes=NULL "
                         "WHERE user_id=?", (user_id,))

    def check_second_factor(self, user_id: int, code: str) -> str | None:
        """
        'totp' or 'recovery' if the code is accepted (and consumed), else None.
        A TOTP code is refused once its time step has been used.
        """
        with self._lock, self._conn() as conn:
            row = self._row(conn, user_id)
            if row is None or not row["totp_secret"]:
                return None
            step = totp.matching_step(row["totp_secret"], code, row["totp_last_step"])
            if step is not None:
                conn.execute("UPDATE users SET totp_last_step=? WHERE user_id=?", (step, user_id))
                return "totp"
            hashes = json.loads(row["recovery_codes"] or "[]")
            for i, h in enumerate(hashes):
                if totp.recovery_code_matches(code, h):
                    del hashes[i]
                    conn.execute("UPDATE users SET recovery_codes=? WHERE user_id=?",
                                 (json.dumps(hashes), user_id))
                    return "recovery"
            return None

    def recovery_codes_left(self, user_id: int) -> int:
        with self._conn() as conn:
            row = self._row(conn, user_id)
            return len(json.loads(row["recovery_codes"] or "[]")) if row else 0

    # --------------------------------------------------------------- audit --
    def audit(self, action: str, user_id: int | None = None, detail: str | None = None) -> None:
        """Append one audit entry (FR-15). Never raises: auditing must not break a request."""
        try:
            with self._lock, self._conn() as conn:
                conn.execute(
                    "INSERT INTO audit_log (occurred_at, user_id, action, detail) VALUES (?,?,?,?)",
                    (_utc_now(), user_id, action, (detail or "")[:500]),
                )
        except (sqlite3.Error, OSError) as exc:
            logger.warning(f"audit write failed ({action}): {exc}")

    def recent_audit(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT a.audit_id, a.occurred_at, a.action, a.detail, u.email "
                "FROM audit_log a LEFT JOIN users u ON u.user_id = a.user_id "
                "ORDER BY a.audit_id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------- figures --
    def role_counts(self) -> dict[str, int]:
        """Active accounts per role, every role present (zero when none)."""
        with self._conn() as conn:
            rows = conn.execute("SELECT role, COUNT(*) FROM users WHERE active=1 GROUP BY role").fetchall()
        counts = {r: 0 for r in ROLES}
        counts.update({role: n for role, n in rows})
        return counts

    def count_actions(self, actions: tuple[str, ...], since_iso: str) -> int:
        marks = ",".join("?" * len(actions))
        with self._conn() as conn:
            return conn.execute(
                f"SELECT COUNT(*) FROM audit_log WHERE occurred_at >= ? AND action IN ({marks})",
                (since_iso, *actions)).fetchone()[0]

    def actions_by_day(self, actions: tuple[str, ...], since_iso: str) -> dict[str, int]:
        """UTC date (YYYY-MM-DD) -> count of these audit actions since `since_iso`."""
        marks = ",".join("?" * len(actions))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT substr(occurred_at, 1, 10), COUNT(*) FROM audit_log "
                f"WHERE occurred_at >= ? AND action IN ({marks}) GROUP BY 1",
                (since_iso, *actions)).fetchall()
        return {d: n for d, n in rows}

    def subscriber_count(self) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(DISTINCT user_id) FROM alert_subscriptions WHERE active=1").fetchone()[0]

    # ------------------------------------------------------- subscriptions --
    # An alert area is a neighbourhood (sub_county): flood warnings are issued per
    # neighbourhood. `label` keeps what the person actually picked - a landmark
    # such as "Strathmore University" - so the alert can be shown in their words.

    def subscriptions(self, user_id: int) -> list[str]:
        """The neighbourhoods this user gets alerts for."""
        return [item["area"] for item in self.subscription_items(user_id)]

    def subscription_items(self, user_id: int) -> list[dict]:
        """[{"area": neighbourhood, "label": what they picked}] for this user."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT sub_county, label FROM alert_subscriptions WHERE user_id=? AND active=1 "
                "ORDER BY sub_county", (user_id,)).fetchall()
            return [{"area": area, "label": label or area} for area, label in rows]

    def set_subscriptions(self, user_id: int, places: list) -> list[str]:
        """
        Replace this user's alert areas. Each item is a neighbourhood name, or a
        {"area", "label"} dict when a landmark was picked; one row per neighbourhood
        (the first label wins). Returns the neighbourhoods saved.
        """
        chosen: dict[str, str] = {}
        for item in places or []:
            area, label = ((item.get("area"), item.get("label")) if isinstance(item, dict) else (item, item))
            area = (area or "").strip()
            if area and area not in chosen:
                chosen[area] = (label or area).strip() or area
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM alert_subscriptions WHERE user_id=?", (user_id,))
            conn.executemany(
                "INSERT INTO alert_subscriptions (user_id, sub_county, active, label) VALUES (?,?,1,?)",
                [(user_id, area, label) for area, label in sorted(chosen.items())],
            )
        return sorted(chosen)


# ------------------------------------------------------------------ shared --
_store: AuthStore | None = None
_store_lock = threading.Lock()


def default_path() -> Path:
    return Path(os.environ.get("TWIN_STATE_DIR", "data")) / "accounts.db"


def get_store() -> AuthStore:
    """The application's store, created on first use in TWIN_STATE_DIR."""
    global _store
    with _store_lock:
        if _store is None:
            _store = AuthStore(default_path())
        return _store


def set_store(store: AuthStore | None) -> None:
    """Swap the shared store (tests use a temporary file)."""
    global _store
    with _store_lock:
        _store = store


def bootstrap_admin(store: AuthStore) -> None:
    """
    Create the first administrator account for TWIN_ADMIN_EMAIL when no active
    admin exists. They sign in with Google or an emailed link to that address and
    enrol an authenticator app. (TWIN_ADMIN_EMAILS, or
    `python -m src.auth.cli create-user`, do the same.)
    """
    email = os.environ.get("TWIN_ADMIN_EMAIL")
    if store.count_active_admins() > 0 or not email:
        return
    try:
        admin = store.create_user(email, "admin")
        store.audit("admin_bootstrapped", admin["user_id"], "created from environment")
        logger.info(f"Created administrator {admin['email']} from TWIN_ADMIN_EMAIL.")
    except AuthError as exc:
        logger.warning(f"Could not create the bootstrap administrator: {exc}")
