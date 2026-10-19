"""
src.auth.access
===============
Who may do what (NFR-04), and who is signed in.

Every permission is checked on the server, inside the callback or route that
performs the action. Hiding a button in the page is a convenience for the user,
never the control: a request that skips the interface is refused the same way.

The public (signed-out) user sees the live map, outlook, warnings and routing,
as FR-01 requires. Everything below needs an account.

TWO-STEP SIGN-IN FOR ADMINISTRATORS
-----------------------------------
A Google or email-link sign-in is the first step for everyone. For an
administrator it only opens a *pending* session: no permissions, valid for
five minutes, good only for the authenticator-code page. The session becomes a
full one after the code is checked (session["mfa"] is then True).

An account promoted to admin while signed in, or one whose session predates
its second factor, is treated as a citizen until it completes the code step.
"""

from __future__ import annotations

import secrets
import time

from flask import has_request_context, session

from src.auth.store import get_store

PENDING_SEC = 5 * 60

#: Permission -> roles allowed.
PERMISSIONS: dict[str, frozenset[str]] = {
    "subscribe_alerts": frozenset({"citizen", "planner", "responder", "admin"}),
    "run_whatif":       frozenset({"planner", "admin"}),
    "save_scenario":    frozenset({"planner", "admin"}),
    "compare":          frozenset({"planner", "admin"}),
    "export_map":       frozenset({"planner", "responder", "admin"}),
    "emergency_report": frozenset({"responder", "admin"}),
    "manage_users":     frozenset({"admin"}),
    "view_audit":       frozenset({"admin"}),
    "refresh_data":     frozenset({"admin"}),
}

ROLE_LABELS = {
    "citizen": "Citizen",
    "planner": "Urban planner",
    "responder": "Emergency responder",
    "admin": "Administrator",
}

#: Roles that must pass an authenticator code after the first sign-in step.
MFA_ROLES = frozenset({"admin"})


def can(user: dict | None, permission: str) -> bool:
    """True if this user's role holds the permission. Unknown permissions are denied."""
    if not user or not user.get("active", False):
        return False
    return user.get("role") in PERMISSIONS.get(permission, frozenset())


def needs_mfa(user: dict | None) -> bool:
    return bool(user) and user.get("role") in MFA_ROLES


def current_user() -> dict | None:
    """
    The signed-in user for this request, re-read so a disabled account or a role
    change takes effect at once. An admin without a completed second factor is
    returned as a citizen with `mfa_pending` set.
    """
    if not has_request_context():
        return None
    uid = session.get("uid")
    if uid is None:
        return None
    user = get_store().get_user(int(uid))
    if not user or not user["active"]:
        session.pop("uid", None)
        return None
    if needs_mfa(user) and not session.get("mfa"):
        return {**user, "role": "citizen", "real_role": user["role"], "mfa_pending": True}
    return user


def sign_in(user: dict, *, mfa: bool = False) -> None:
    """Start a full session (mfa=True once the second factor has been checked)."""
    session.clear()
    session.permanent = True
    session["uid"] = user["user_id"]
    session["mfa"] = bool(mfa)
    session["csrf"] = secrets.token_urlsafe(24)
    get_store().touch_login(user["user_id"])


def start_pending(user: dict, method: str) -> None:
    """First step passed for an account that also needs an authenticator code."""
    session.clear()
    session["pending_uid"] = user["user_id"]
    session["pending_until"] = time.time() + PENDING_SEC
    session["pending_method"] = method
    session["csrf"] = secrets.token_urlsafe(24)


def mfa_subject() -> dict | None:
    """
    The account the code page is for: a valid pending sign-in, or a signed-in
    user whose role needs a second factor that this session has not passed.
    """
    if not has_request_context():
        return None
    store = get_store()
    uid = session.get("pending_uid")
    if uid is not None:
        if time.time() > float(session.get("pending_until") or 0):
            for key in ("pending_uid", "pending_until", "pending_method", "totp_setup"):
                session.pop(key, None)
            return None
        user = store.get_user(int(uid))
        return user if user and user["active"] else None
    uid = session.get("uid")
    if uid is not None and not session.get("mfa"):
        user = store.get_user(int(uid))
        if user and user["active"] and needs_mfa(user):
            return user
    return None


def sign_out() -> None:
    session.clear()


def csrf_token() -> str:
    """Per-session token embedded in every form that posts to /auth/*."""
    token = session.get("csrf")
    if not token:
        token = session["csrf"] = secrets.token_urlsafe(24)
    return token


def csrf_ok(value: str | None) -> bool:
    expected = session.get("csrf")
    return bool(expected) and secrets.compare_digest(str(value or ""), expected)
