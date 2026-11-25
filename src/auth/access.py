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

#: Staff roles: everyone except the public (citizen).
STAFF = frozenset({"planner", "responder", "admin"})
_ANALYSIS = frozenset({"planner", "admin"})
_OPERATIONS = frozenset({"responder", "admin"})

#: Permission -> roles allowed.
PERMISSIONS: dict[str, frozenset[str]] = {
    "subscribe_alerts": frozenset({"citizen"}) | STAFF,
    "run_whatif":       _ANALYSIS,
    # Past storms replayed through the model. Staff only: a replay's "flooding
    # now" warnings could be mistaken for the live situation by the public.
    "view_replays":     STAFF,
    "save_scenario":    _ANALYSIS,
    "compare":          _ANALYSIS,
    "export_map":       STAFF,
    # Community flood reports: any signed-in user reports and confirms;
    # responders and administrators verify or remove.
    "report_flood":     frozenset({"citizen"}) | STAFF,
    "moderate_reports": _OPERATIONS,
    "emergency_report": _OPERATIONS,
    "manage_users":     frozenset({"admin"}),
    "view_audit":       frozenset({"admin"}),
    "refresh_data":     frozenset({"admin"}),
    # Operations console.
    "view_ops":         STAFF,
    "view_sensors":     STAFF,
    "view_alert_record": _OPERATIONS,
    "view_infrastructure": STAFF,
    # Sensitive: aggregated beneficiary figures and emergency stock.
    "view_beneficiaries": _OPERATIONS,
    "view_supplies":    _OPERATIONS,
    "generate_reports": STAFF,
    "export_readings":  _ANALYSIS,
    "view_system_status": frozenset({"admin"}),
}

ROLE_LABELS = {
    "citizen": "Citizen",
    "planner": "Urban planner",
    "responder": "Emergency responder",
    "admin": "Admin",
}

#: What each role is for, shown when an administrator assigns roles.
ROLE_DUTIES = {
    "citizen": "Views public flood information, published warnings and reports flooding.",
    "planner": "Runs what-if scenarios, compares flood maps and reviews trends for planning.",
    "responder": "Monitors live conditions, alerts and reports, and supports field response.",
    "admin": "Manages users, configuration and data feeds.",
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
