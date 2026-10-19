"""
tests.test_auth
===============
Accounts, roles, sign-in and access control.

Traceability: FR-12 (authenticate and authorise), NFR-04 (role-restricted
functions, tested as an access-control matrix), NFR-05 (salted password hashes),
FR-15 / DR-09 (append-only audit log), FR-03 (alert subscriptions), FR-11
(emergency report).

Each test uses its own temporary account database. Firebase tokens (Google and
email link) are faked by replacing the verifier, so no network call is made.
Authenticator codes are computed from the stored secret, as the phone app would.
"""

from __future__ import annotations

import re
import sqlite3

import pytest
from flask import Flask

from src.auth import access, google, routes, totp
from src.auth.store import AuthError, AuthStore, set_store
from tests.conftest import needs_model

PW = "correct horse"


@pytest.fixture()
def store(tmp_path):
    s = AuthStore(tmp_path / "accounts.db")
    set_store(s)
    yield s
    set_store(None)


@pytest.fixture()
def client(store, monkeypatch, tmp_path):
    monkeypatch.setenv("TWIN_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("TWIN_ADMIN_EMAIL", raising=False)
    monkeypatch.delenv("TWIN_ADMIN_EMAILS", raising=False)
    routes._failures.clear()
    app = Flask(__name__)
    routes.init_auth(app)

    @app.get("/whoami")
    def whoami():
        u = access.current_user()
        return {"email": u["email"] if u else None, "role": u["role"] if u else None,
                "mfa_pending": bool(u and u.get("mfa_pending"))}

    @app.get("/csrf")
    def csrf():
        return {"csrf": access.csrf_token()}

    @app.get("/setup-secret")
    def setup_secret():
        # What the /2fa page does on first visit: a secret held in the session.
        from flask import session
        session.setdefault("totp_setup", totp.new_secret())
        return {"secret": session["totp_setup"]}

    return app.test_client()


def _token(client):
    return client.get("/csrf").get_json()["csrf"]


def _login(client, email, password):
    return client.post("/auth/login", data={"email": email, "password": password, "csrf": _token(client)})


def _code(client, code):
    return client.post("/auth/2fa", data={"code": code, "csrf": _token(client)})


def _fake_firebase(monkeypatch, provider="google.com", **claims):
    base = {"email": "jane@example.com", "email_verified": True, "name": "Jane",
            "firebase": {"sign_in_provider": provider}}
    base.update(claims)
    monkeypatch.setenv("FIREBASE_PROJECT_ID", "test-project")
    import google.oauth2.id_token as g
    monkeypatch.setattr(g, "verify_firebase_token", lambda token, req, audience=None: base)


# ------------------------------------------------------------- passwords --
def test_passwords_are_stored_as_salted_hashes(store):
    """NFR-05: no plaintext; the same password gives different hashes."""
    store.create_user("alice@example.com", PW, "planner")
    store.create_user("bob@example.com", PW, "citizen")
    hashes = [r[0] for r in sqlite3.connect(store.path).execute("SELECT password_hash FROM users")]
    assert all(PW not in h for h in hashes)
    assert hashes[0] != hashes[1], "a fresh salt per password"
    assert all(h.startswith(("scrypt:", "pbkdf2:")) for h in hashes)


def test_password_check_accepts_only_the_right_password(store):
    store.create_user("Alice@Example.com", PW, "planner")
    assert store.verify_password("alice@example.com", PW)["email"] == "alice@example.com"
    assert store.verify_password("alice@example.com", "wrong password") is None
    assert store.verify_password("nobody@example.com", PW) is None


def test_invalid_accounts_are_rejected(store):
    for email, pw, role in [("alice@example.com", "short", "citizen"), ("not-an-email", PW, "citizen"),
                            ("carol@example.com", PW, "superuser")]:
        with pytest.raises(AuthError):
            store.create_user(email, pw, role)
    store.create_user("dup@example.com", PW)
    with pytest.raises(AuthError):
        store.create_user("DUP@example.com", PW)


def test_passwordless_accounts_have_no_password(store):
    u = store.create_user("jane@example.com", None, "planner", provider="google")
    assert not u["has_password"]
    row = sqlite3.connect(store.path).execute("SELECT password_hash FROM users").fetchone()
    assert row[0] is None
    assert store.verify_password("jane@example.com", "") is None


# ------------------------------------------------------- access control --
MATRIX = {
    # permission:        citizen planner responder admin
    "subscribe_alerts": (True,   True,   True,     True),
    "run_whatif":       (False,  True,   False,    True),
    "save_scenario":    (False,  True,   False,    True),
    "compare":          (False,  True,   False,    True),
    "export_map":       (False,  True,   True,     True),
    "emergency_report": (False,  False,  True,     True),
    "manage_users":     (False,  False,  False,    True),
    "view_audit":       (False,  False,  False,    True),
    "refresh_data":     (False,  False,  False,    True),
}


@pytest.mark.parametrize("permission", sorted(MATRIX))
def test_access_control_matrix(permission):
    """NFR-04: every restricted function attempted under every role."""
    for role, expected in zip(("citizen", "planner", "responder", "admin"), MATRIX[permission]):
        assert access.can({"role": role, "active": True}, permission) is expected, f"{role} / {permission}"
    assert access.can(None, permission) is False, "signed-out users hold no permission"


def test_disabled_accounts_and_unknown_permissions_are_denied():
    assert access.can({"role": "admin", "active": False}, "manage_users") is False
    assert access.can({"role": "admin", "active": True}, "launch_rockets") is False


def test_admin_console_is_not_rendered_for_other_roles(store, client):
    from src.dashboard.admin_dashboard import render_admin
    for role in ("citizen", "planner", "responder"):
        for path in ("/admin", "/admin/users", "/admin/activity", "/admin/system"):
            page = str(render_admin({"role": role, "active": True, "user_id": 0}, path))
            assert "Users and roles" not in page and "Audit log" not in page and "dash-shell" not in page


def test_dashboard_figures_count_roles_sign_ins_and_failures(store):
    from datetime import datetime, timezone
    store.create_user("root@example.com", PW, "admin")
    p = store.create_user("p@example.com", PW, "planner")
    store.create_user("c@example.com", PW, "citizen")
    store.set_subscriptions(p["user_id"], ["Mathare"])
    store.audit("login", p["user_id"])
    store.audit("login", p["user_id"])
    store.audit("login_failed", None, "email=x")
    assert store.role_counts() == {"citizen": 1, "planner": 1, "responder": 0, "admin": 1}
    assert store.count_actions(("login",), "2000-01-01") == 2
    assert store.count_actions(("login_failed", "2fa_failed"), "2000-01-01") == 1
    today = datetime.now(timezone.utc).date().isoformat()
    assert store.actions_by_day(("login",), "2000-01-01") == {today: 2}
    assert store.subscriber_count() == 1


def test_user_search_role_filter_and_activity_filter(store):
    from src.dashboard.admin_dashboard import audit_table, users_table
    store.create_user("root@example.com", PW, "admin")
    store.create_user("planner@example.com", PW, "planner")
    store.audit("login", None)
    store.audit("export_geojson", None, "What-if 60 mm")
    assert "planner@example.com" in str(users_table("plann")) and "root@" not in str(users_table("plann"))
    assert "root@example.com" in str(users_table(None, "admin")) and "planner@" not in str(users_table(None, "admin"))
    assert "No users match" in str(users_table("nobody"))
    data_only = str(audit_table(None, "data"))
    assert "Flood map exported" in data_only and "Signed in" not in data_only
    assert "Flood map exported" in str(audit_table("60 mm", "all"))


# ------------------------------------------------------- password sign-in --
def test_login_starts_a_session_and_logout_ends_it(store, client):
    store.create_user("alice@example.com", PW, "planner")
    resp = _login(client, "alice@example.com", PW)
    assert resp.status_code == 302 and resp.headers["Location"].endswith("/")
    assert client.get("/whoami").get_json()["role"] == "planner"
    client.get("/auth/logout")
    assert client.get("/whoami").get_json()["email"] is None


def test_wrong_password_is_refused_with_one_message_and_audited(store, client):
    store.create_user("alice@example.com", PW, "planner")
    store.create_user("jane@example.com", None, "planner", provider="google")
    wrong = _login(client, "alice@example.com", "wrong password").headers["Location"]
    nopw = _login(client, "jane@example.com", "anything at all").headers["Location"]
    unknown = _login(client, "ghost@example.com", "anything at all").headers["Location"]
    assert wrong == nopw == unknown, "the form must not reveal which emails exist or how they sign in"
    assert client.get("/whoami").get_json()["email"] is None
    assert store.recent_audit()[0]["action"] == "login_failed"


def test_login_without_csrf_token_is_refused(store, client):
    store.create_user("alice@example.com", PW, "planner")
    resp = client.post("/auth/login", data={"email": "alice@example.com", "password": PW})
    assert "/login?error=" in resp.headers["Location"]
    assert client.get("/whoami").get_json()["email"] is None


def test_repeated_failures_lock_the_account_temporarily(store, client):
    store.create_user("alice@example.com", PW, "planner")
    for _ in range(routes.MAX_FAILURES):
        _login(client, "alice@example.com", "wrong password")
    assert "Too+many" in _login(client, "alice@example.com", PW).headers["Location"]
    assert client.get("/whoami").get_json()["email"] is None


def test_self_registration_always_creates_a_citizen(store, client):
    client.post("/auth/register", data={"email": "dan@example.com", "password": "long enough pw",
                                        "confirm": "long enough pw", "role": "admin", "csrf": _token(client)})
    assert store.get_by_email("dan@example.com")["role"] == "citizen"
    assert client.get("/whoami").get_json()["role"] == "citizen"


def test_disabled_user_loses_access_immediately(store, client):
    store.create_user("root@example.com", PW, "admin")
    alice = store.create_user("alice@example.com", PW, "planner")
    _login(client, "alice@example.com", PW)
    store.update_user(alice["user_id"], active=False)
    assert client.get("/whoami").get_json()["email"] is None


# --------------------------------------------- Firebase: Google and email link --
@pytest.mark.parametrize("provider", ["google.com", "password"])
def test_first_firebase_sign_in_creates_a_citizen(store, client, monkeypatch, provider):
    """Google and email-link sign-ins both create a passwordless citizen account."""
    _fake_firebase(monkeypatch, provider)
    resp = client.post("/auth/firebase", json={"idToken": "fake"})
    assert resp.status_code == 200 and resp.get_json() == {"ok": True, "next": "/account"}
    user = store.get_by_email("jane@example.com")
    assert user["role"] == "citizen" and not user["has_password"]
    assert user["auth_provider"] == ("google" if provider == "google.com" else "email_link")


@pytest.mark.parametrize("provider", ["google.com", "password"])
def test_firebase_sign_in_opens_the_account_the_admin_created(store, client, monkeypatch, provider):
    """The role comes from this system: a pre-created responder signs in as a responder either way."""
    store.create_user("jane@example.com", None, "responder", provider="google")
    _fake_firebase(monkeypatch, provider)
    assert client.post("/auth/firebase", json={"idToken": "fake"}).get_json()["next"] == "/"
    assert client.get("/whoami").get_json() == {"email": "jane@example.com", "role": "responder",
                                                "mfa_pending": False}
    assert len(store.list_users()) == 1


def test_password_account_can_also_use_google(store, client, monkeypatch):
    store.create_user("jane@example.com", PW, "planner")
    _fake_firebase(monkeypatch)
    client.post("/auth/firebase", json={"idToken": "fake"})
    assert client.get("/whoami").get_json()["role"] == "planner"


def test_unverified_email_is_refused(store, client, monkeypatch):
    _fake_firebase(monkeypatch, "password", email_verified=False)
    assert client.post("/auth/firebase", json={"idToken": "fake"}).status_code == 401
    assert store.get_by_email("jane@example.com") is None


def test_other_firebase_methods_and_forged_tokens_are_refused(store, client, monkeypatch):
    _fake_firebase(monkeypatch, "anonymous")
    assert client.post("/auth/firebase", json={"idToken": "fake"}).status_code == 401
    import google.oauth2.id_token as g

    def reject(*_a, **_k):
        raise ValueError("Token has wrong audience")
    monkeypatch.setattr(g, "verify_firebase_token", reject)
    assert client.post("/auth/firebase", json={"idToken": "forged"}).status_code == 401
    assert client.get("/whoami").get_json()["email"] is None


def test_firebase_options_are_hidden_when_not_configured(monkeypatch):
    monkeypatch.delenv("FIREBASE_PROJECT_ID", raising=False)
    monkeypatch.delenv("FIREBASE_API_KEY", raising=False)
    assert google.firebase_config() is None


# ------------------------------------------- administrator two-step sign-in --
def test_totp_codes_follow_rfc_6238():
    """RFC 6238 Appendix B test vector (SHA-1, secret '12345678901234567890'), last 6 digits."""
    import base64
    secret = base64.b32encode(b"12345678901234567890").decode().rstrip("=")
    assert totp.code_now(secret, at=59) == "287082"
    assert totp.code_now(secret, at=1111111109) == "081804"


def test_admin_password_is_only_the_first_step(store, client):
    store.create_user("root@example.com", PW, "admin")
    resp = _login(client, "root@example.com", PW)
    assert resp.headers["Location"].endswith("/2fa")
    assert client.get("/whoami").get_json()["email"] is None, "no session until the code is checked"


class _Clock:
    """A controllable clock for the authenticator, so tests never straddle a 30 s step."""
    def __init__(self, t):
        self.t = t

    def time(self):
        return self.t


def test_admin_enrols_authenticator_then_needs_codes(store, client, monkeypatch):
    clock = _Clock(1_800_000_000.0)
    monkeypatch.setattr(totp, "time", clock)
    store.create_user("root@example.com", PW, "admin")
    _login(client, "root@example.com", PW)
    secret = client.get("/setup-secret").get_json()["secret"]
    assert "/2fa?error=" in _code(client, "000000").headers["Location"]
    enrol_code = totp.code_now(secret)
    resp = _code(client, enrol_code)
    assert resp.headers["Location"].endswith("/2fa-codes")
    assert client.get("/whoami").get_json() == {"email": "root@example.com", "role": "admin", "mfa_pending": False}
    assert store.get_by_email("root@example.com")["totp_enabled"]

    # Next sign-in: the app's code is required, and the enrolment code cannot be reused.
    client.get("/auth/logout")
    _login(client, "root@example.com", PW)
    assert "/2fa?error=" in _code(client, enrol_code).headers["Location"]
    clock.t += totp.STEP_SEC
    assert _code(client, totp.code_now(secret)).headers["Location"].endswith("/admin")
    assert client.get("/whoami").get_json()["role"] == "admin"


def test_admin_google_and_email_link_sign_ins_also_need_the_code(store, client, monkeypatch):
    store.create_user("jane@example.com", None, "admin", provider="google")
    for provider in ("google.com", "password"):
        client.get("/auth/logout")
        _fake_firebase(monkeypatch, provider)
        assert client.post("/auth/firebase", json={"idToken": "fake"}).get_json()["next"] == "/2fa"
        assert client.get("/whoami").get_json()["email"] is None


def test_recovery_codes_work_once(store, client):
    admin = store.create_user("root@example.com", PW, "admin")
    secret, codes = totp.new_secret(), totp.new_recovery_codes()
    store.enable_totp(admin["user_id"], secret, codes, 0)
    stored = sqlite3.connect(store.path).execute("SELECT recovery_codes FROM users").fetchone()[0]
    assert codes[0] not in stored, "recovery codes are stored only as hashes"
    _login(client, "root@example.com", PW)
    assert _code(client, codes[0].upper()).headers["Location"].endswith("/admin")
    assert store.recovery_codes_left(admin["user_id"]) == len(codes) - 1
    client.get("/auth/logout")
    _login(client, "root@example.com", PW)
    assert "/2fa?error=" in _code(client, codes[0]).headers["Location"]


def test_wrong_codes_lock_the_sign_in(store, client):
    admin = store.create_user("root@example.com", PW, "admin")
    store.enable_totp(admin["user_id"], totp.new_secret(), totp.new_recovery_codes(), 0)
    _login(client, "root@example.com", PW)
    for _ in range(routes.MAX_FAILURES):
        _code(client, "123456")
    assert "Too+many" in _code(client, "123456").headers["Location"]


def test_promoted_admin_keeps_citizen_powers_until_the_code_step(store, client):
    store.create_user("root@example.com", PW, "admin")
    alice = store.create_user("alice@example.com", PW, "planner")
    _login(client, "alice@example.com", PW)
    store.update_user(alice["user_id"], role="admin")
    me = client.get("/whoami").get_json()
    assert me["role"] == "citizen" and me["mfa_pending"], "no admin rights without a second factor"


def test_pending_sign_in_expires(store, client, monkeypatch):
    store.create_user("root@example.com", PW, "admin")
    _login(client, "root@example.com", PW)
    monkeypatch.setattr(access.time, "time", lambda: 10 ** 10)
    assert "/login?error=" in _code(client, "123456").headers["Location"]


# ------------------------------------------------------- administration --
def test_last_admin_cannot_be_demoted_or_disabled(store):
    root = store.create_user("root@example.com", PW, "admin")
    with pytest.raises(AuthError):
        store.update_user(root["user_id"], role="planner")
    with pytest.raises(AuthError):
        store.update_user(root["user_id"], active=False)
    store.create_user("second@example.com", PW, "admin")
    assert store.update_user(root["user_id"], role="planner")["role"] == "planner"


def test_bootstrap_admin_from_environment(store, monkeypatch):
    from src.auth.store import bootstrap_admin
    monkeypatch.setenv("TWIN_ADMIN_EMAIL", "root@example.com")
    monkeypatch.setenv("TWIN_ADMIN_PASSWORD", PW)
    bootstrap_admin(store)
    bootstrap_admin(store)                  # idempotent once an admin exists
    assert [(u["email"], u["role"]) for u in store.list_users()] == [("root@example.com", "admin")]


def test_old_databases_are_upgraded_in_place(tmp_path):
    """A database from before email links and 2FA keeps its users and gains the new fields."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE users (user_id INTEGER PRIMARY KEY, username TEXT NOT NULL UNIQUE, email TEXT UNIQUE,
            display_name TEXT, password_hash TEXT,
            auth_provider TEXT NOT NULL DEFAULT 'local' CHECK (auth_provider IN ('local','google')),
            role TEXT NOT NULL DEFAULT 'citizen', active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL, last_login_at TEXT);
        INSERT INTO users (username, email, role, created_at) VALUES ('old', 'old@example.com', 'planner', 'x');
    """)
    conn.commit()
    conn.close()
    s = AuthStore(path)
    assert s.get_by_email("old@example.com")["role"] == "planner"
    assert s.create_user("new@example.com", None, provider="email_link")["auth_provider"] == "email_link"


def test_audit_log_is_append_only_in_order(store):
    """FR-15 / DR-09: entries accumulate; nothing in the store updates or deletes them."""
    store.audit("first")
    store.audit("second", detail="x" * 1000)
    assert [a["action"] for a in store.recent_audit()] == ["second", "first"]
    assert len(store.recent_audit()[0]["detail"]) == 500


def test_alert_subscriptions_are_saved_per_user(store):
    """FR-03: a citizen's chosen areas are recorded against their profile."""
    u = store.create_user("alice@example.com", PW, "citizen")
    store.set_subscriptions(u["user_id"], ["Mathare", "Kibera", "Mathare", " "])
    assert store.subscriptions(u["user_id"]) == ["Kibera", "Mathare"]
    store.set_subscriptions(u["user_id"], ["Eastleigh"])
    assert store.subscriptions(u["user_id"]) == ["Eastleigh"]


# ------------------------------------------------- server-side enforcement --
@needs_model
def test_whatif_is_refused_on_the_server_for_roles_without_it(store):
    """A What-if request that bypasses the hidden control is still held to live mode."""
    from flask import session
    from src.dashboard.callbacks import _allowed_mode
    app = Flask(__name__)
    app.secret_key = "test"
    planner = store.create_user("alice@example.com", PW, "planner")
    citizen = store.create_user("bob@example.com", PW, "citizen")
    with app.test_request_context():
        assert _allowed_mode("WHATIF") == "LIVE", "signed out"
        session["uid"] = citizen["user_id"]
        assert _allowed_mode("WHATIF") == "LIVE", "citizen"
        session["uid"] = planner["user_id"]
        assert _allowed_mode("WHATIF") == "WHATIF", "planner"


def test_emergency_report_contains_the_scenario_and_escapes_names():
    """FR-11: parameters, extent, affected settlements and a timestamp; names are HTML-escaped."""
    from src.dashboard.auth_pages import emergency_report_html
    summary = {"label": "What-if scenario, 60 mm over 3 days", "mode": "WHATIF", "rain": 60.0,
               "area_km2": 13.5, "flooded_pct": 2.7, "pop": 366100, "model": "unet.pth",
               "regions": [{"name": "Mathare", "level": "CRITICAL", "peak_pct": 97.0, "flooded_pct": 41.0},
                           {"name": "<script>x</script>", "level": "HIGH", "peak_pct": 70.0, "flooded_pct": 5.0}]}
    page = emergency_report_html(summary, {"username": "rita", "email": "rita@example.com",
                                           "role": "responder"}, None)
    assert "60 mm" in page and "13.50 km²" in page and "366,100" in page and "Mathare" in page
    assert "Generated" in page and "rita@example.com" in page
    assert "<script>x</script>" not in page and "&lt;script&gt;" in page
    assert re.search(r"\d{2} \w{3} \d{4}, \d{2}:\d{2} EAT", page), "generation timestamp"
