"""
src.auth.routes
===============
Sign-in, registration and sign-out endpoints on the Flask server under Dash.

    POST /auth/firebase  {"idToken": ...} from Firebase: Google popup or email link
    POST /auth/2fa       authenticator code (or recovery code) for administrators
    GET  /auth/logout
    GET  /auth/firebase-config   public Firebase web config, or {} when unset

There are no passwords. Every account signs in with the Google account of its
email or a link Firebase emails to that address; a first sign-in creates a
citizen account. Administrators then pass an authenticator code
(src/auth/totp.py); the first time, they enrol the app on the /2fa page.

The code form carries a per-session CSRF token. Repeated wrong codes lock the
attempt for a few minutes. Every outcome is written to the audit log.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlencode

from flask import jsonify, redirect, request, session
from loguru import logger

from src.auth import google, totp
from src.auth.access import (csrf_ok, current_user, mfa_subject, needs_mfa, sign_in, sign_out,
                             start_pending)
from src.auth.store import AuthError, bootstrap_admin, get_store

MAX_FAILURES = 5
LOCKOUT_SEC = 5 * 60
_failures: dict[str, list[float]] = {}
_failures_lock = threading.Lock()


def _locked(key: str) -> bool:
    now = time.time()
    with _failures_lock:
        recent = [t for t in _failures.get(key, []) if now - t < LOCKOUT_SEC]
        _failures[key] = recent
        return len(recent) >= MAX_FAILURES


def _record_failure(key: str) -> None:
    with _failures_lock:
        _failures.setdefault(key, []).append(time.time())


def _clear_failures(key: str) -> None:
    with _failures_lock:
        _failures.pop(key, None)


def _back(path: str, error: str, **extra):
    return redirect(f"{path}?" + urlencode({"error": error, **extra}))


def _after_first_step(user: dict, method: str) -> str:
    """Finish a Google or email-link sign-in; returns where the browser goes next."""
    store = get_store()
    if needs_mfa(user):
        start_pending(user, method)
        store.audit("login_first_step", user["user_id"], f"{method}; authenticator code required")
        return "/2fa"
    sign_in(user)
    store.audit("login", user["user_id"], method)
    return "/"


def _secret_key() -> str:
    """TWIN_SECRET_KEY, else a key kept in the state directory, else a per-process key."""
    env = os.environ.get("TWIN_SECRET_KEY")
    if env:
        return env
    path = Path(os.environ.get("TWIN_STATE_DIR", "data")) / "secret_key"
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        path.parent.mkdir(parents=True, exist_ok=True)
        key = secrets.token_hex(32)
        path.write_text(key, encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return key
    except OSError as exc:
        logger.warning(f"Could not store a session key ({exc}); sessions end when the server restarts.")
        return secrets.token_hex(32)


def init_auth(server) -> None:
    """Configure sessions and register the /auth routes on the Flask server."""
    server.secret_key = _secret_key()
    server.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("TWIN_COOKIE_SECURE") == "1",
        PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
    )
    try:
        bootstrap_admin(get_store())
    except Exception as exc:                                # noqa: BLE001
        logger.warning(f"Account store unavailable at start-up: {exc}")

    @server.post("/auth/firebase")
    def auth_firebase():
        store = get_store()
        body = request.get_json(silent=True) or {}
        try:
            claims, method = google.verify_token(body.get("idToken") or "")
            user, created = google.user_from_claims(store, claims, method)
        except AuthError as exc:
            store.audit("login_failed", None, f"firebase: {exc}")
            return jsonify({"ok": False, "error": str(exc)}), 401
        label = google.METHOD_LABELS[method]
        if created:
            store.audit("register", user["user_id"], label)
        nxt = _after_first_step(user, label)
        if created and nxt == "/":
            nxt = "/account"
        return jsonify({"ok": True, "next": nxt})

    @server.post("/auth/2fa")
    def auth_2fa():
        store = get_store()
        if not csrf_ok(request.form.get("csrf")):
            return _back("/login", "Your session expired. Please sign in again.")
        user = mfa_subject()
        if user is None:
            return _back("/login", "Your sign-in expired. Please sign in again.")
        key = f"2fa|{user['user_id']}"
        if _locked(key):
            sign_out()
            store.audit("2fa_locked", user["user_id"])
            return _back("/login", "Too many wrong codes. Sign in again in a few minutes.")
        code = request.form.get("code") or ""

        if user["totp_enabled"]:
            kind = store.check_second_factor(user["user_id"], code)
            if kind is None:
                _record_failure(key)
                store.audit("2fa_failed", user["user_id"])
                return _back("/2fa", "That code is not correct, or it has already been used.")
            _clear_failures(key)
            method = session.get("pending_method") or "session"
            sign_in(user, mfa=True)
            left = store.recovery_codes_left(user["user_id"])
            store.audit("login", user["user_id"], f"{method} + {'recovery code' if kind == 'recovery' else 'authenticator'}"
                        + (f"; {left} recovery codes left" if kind == "recovery" else ""))
            return redirect("/admin")

        # First time: confirm the app shows the right code before switching 2FA on.
        secret = session.get("totp_setup")
        step = totp.matching_step(secret, code, None) if secret else None
        if step is None:
            _record_failure(key)
            return _back("/2fa", "That code does not match. Check the time on your phone and try again.")
        _clear_failures(key)
        codes = totp.new_recovery_codes()
        store.enable_totp(user["user_id"], secret, codes, step)
        method = session.get("pending_method") or "session"
        sign_in(user, mfa=True)
        session["recovery_codes_once"] = codes
        store.audit("2fa_enrolled", user["user_id"], f"after {method} sign-in")
        return redirect("/2fa-codes")

    @server.get("/auth/logout")
    def auth_logout():
        user = current_user() or mfa_subject()
        if user:
            get_store().audit("logout", user["user_id"])
        sign_out()
        return redirect("/")

    @server.get("/auth/firebase-config")
    def auth_firebase_config():
        return jsonify(google.firebase_config() or {})
