"""
tests.test_auth
===============
Accounts, roles, sign-in and access control.

Traceability: FR-12 (authenticate and authorise), NFR-04 (role-restricted
functions, tested as an access-control matrix), NFR-05 (no recoverable
credentials stored), FR-15 / DR-09 (append-only audit log), FR-03 (alert
subscriptions), FR-11 (emergency report).

There are no passwords: every sign-in goes through Firebase (Google or an
emailed link). Each test uses its own temporary account database, and Firebase
tokens are faked by replacing the verifier, so no network call is made.
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

EMAIL_LINK, GOOGLE = "password", "google.com"     # Firebase sign_in_provider values


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


def _fake_firebase(monkeypatch, provider=EMAIL_LINK, **claims):
    base = {"email": "jane@example.com", "email_verified": True, "name": "Jane",
            "firebase": {"sign_in_provider": provider}}
    base.update(claims)
    monkeypatch.setenv("FIREBASE_PROJECT_ID", "test-project")
    import google.oauth2.id_token as g
    monkeypatch.setattr(g, "verify_firebase_token", lambda token, req, audience=None: base)


def _sign_in(client, monkeypatch, email="jane@example.com", provider=EMAIL_LINK):
    """Sign in through /auth/firebase as this email; returns the JSON reply."""
    _fake_firebase(monkeypatch, provider, email=email)
    return client.post("/auth/firebase", json={"idToken": "fake"}).get_json()


def _code(client, code):
    token = client.get("/csrf").get_json()["csrf"]
    return client.post("/auth/2fa", data={"code": code, "csrf": token})


class _Clock:
    """A controllable clock for the authenticator, so tests never straddle a 30 s step."""
    def __init__(self, t):
        self.t = t

    def time(self):
        return self.t


# ------------------------------------------------------ no passwords at all --
def test_password_routes_do_not_exist(store, client):
    for path in ("/auth/login", "/auth/register"):
        assert client.post(path, data={"email": "a@example.com", "password": "anything"}).status_code == 404


def test_no_credentials_are_stored(store):
    """NFR-05: accounts hold no password; nothing recoverable is kept."""
    store.create_user("alice@example.com", "planner")
    assert sqlite3.connect(store.path).execute("SELECT password_hash FROM users").fetchone()[0] is None


def test_password_hashes_from_older_databases_are_erased(tmp_path):
    path = tmp_path / "old.db"
    AuthStore(path).create_user("old@example.com", "planner")
    conn = sqlite3.connect(path)
    conn.execute("UPDATE users SET password_hash='scrypt:32768:8:1$salt$hash'")
    conn.commit()
    conn.close()
    AuthStore(path)                                   # next start-up
    assert sqlite3.connect(path).execute("SELECT password_hash FROM users").fetchone()[0] is None


def test_invalid_accounts_are_rejected(store):
    for email, role, provider in [("not-an-email", "citizen", "email_link"),
                                  ("carol@example.com", "superuser", "email_link"),
                                  ("dave@example.com", "citizen", "local")]:
        with pytest.raises(AuthError):
            store.create_user(email, role, provider=provider)
    store.create_user("dup@example.com")
    with pytest.raises(AuthError):
        store.create_user("DUP@example.com")


# ------------------------------------------------------- access control --
MATRIX = {
    # permission:        citizen planner responder admin
    "subscribe_alerts": (True,   True,   True,     True),
    "run_whatif":       (False,  True,   False,    True),
    "view_replays":     (False,  True,   True,     True),
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


# --------------------------------------------- sign-in: Google and email link --
@pytest.mark.parametrize("provider", [GOOGLE, EMAIL_LINK])
def test_first_sign_in_creates_a_citizen_account(store, client, monkeypatch, provider):
    """Someone with no account yet gets a citizen account and is sent to choose alert areas."""
    assert _sign_in(client, monkeypatch, provider=provider) == {"ok": True, "next": "/account"}
    user = store.get_by_email("jane@example.com")
    assert user["role"] == "citizen"
    assert user["auth_provider"] == ("google" if provider == GOOGLE else "email_link")
    assert client.get("/whoami").get_json()["role"] == "citizen"
    assert [a["action"] for a in store.recent_audit()][:2] == ["login", "register"]


@pytest.mark.parametrize("provider", [GOOGLE, EMAIL_LINK])
def test_sign_in_opens_the_account_the_admin_created(store, client, monkeypatch, provider):
    """The role comes from this system: a pre-created responder signs in as a responder either way."""
    store.create_user("jane@example.com", "responder")
    assert _sign_in(client, monkeypatch, provider=provider)["next"] == "/"
    assert client.get("/whoami").get_json() == {"email": "jane@example.com", "role": "responder",
                                                "mfa_pending": False}
    assert len(store.list_users()) == 1


def test_google_and_email_link_open_the_same_account(store, client, monkeypatch):
    _sign_in(client, monkeypatch, provider=GOOGLE)
    client.get("/auth/logout")
    _sign_in(client, monkeypatch, provider=EMAIL_LINK)
    assert len(store.list_users()) == 1


def test_sign_out_ends_the_session(store, client, monkeypatch):
    _sign_in(client, monkeypatch)
    client.get("/auth/logout")
    assert client.get("/whoami").get_json()["email"] is None


def test_unverified_email_is_refused(store, client, monkeypatch):
    _fake_firebase(monkeypatch, EMAIL_LINK, email_verified=False)
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


def test_disabled_account_cannot_sign_in_and_loses_its_session(store, client, monkeypatch):
    store.create_user("root@example.com", "admin")
    jane = store.create_user("jane@example.com", "planner")
    _sign_in(client, monkeypatch)
    store.update_user(jane["user_id"], active=False)
    assert client.get("/whoami").get_json()["email"] is None
    _fake_firebase(monkeypatch)
    assert client.post("/auth/firebase", json={"idToken": "fake"}).status_code == 401


def test_sign_in_page_explains_when_firebase_is_not_configured(monkeypatch):
    from src.dashboard.auth_pages import render_login
    monkeypatch.delenv("FIREBASE_PROJECT_ID", raising=False)
    monkeypatch.delenv("FIREBASE_API_KEY", raising=False)
    assert google.firebase_config() is None
    app = Flask(__name__)
    app.secret_key = "test"
    with app.test_request_context():
        page = str(render_login(""))
    assert "not set up" in page and "btn-google" not in page and "password" not in page.lower()


def test_sign_in_page_offers_only_google_and_email_link(monkeypatch):
    from src.dashboard.auth_pages import render_login
    monkeypatch.setenv("FIREBASE_PROJECT_ID", "p")
    monkeypatch.setenv("FIREBASE_API_KEY", "k")
    app = Flask(__name__)
    app.secret_key = "test"
    with app.test_request_context():
        page = str(render_login(""))
    assert "btn-google" in page and "btn-email-link" in page
    assert "/auth/login" not in page and "'password'" not in page and "Register" not in page


# ------------------------------------------- administrator two-step sign-in --
def test_totp_codes_follow_rfc_6238():
    """RFC 6238 Appendix B test vector (SHA-1, secret '12345678901234567890'), last 6 digits."""
    import base64
    secret = base64.b32encode(b"12345678901234567890").decode().rstrip("=")
    assert totp.code_now(secret, at=59) == "287082"
    assert totp.code_now(secret, at=1111111109) == "081804"


@pytest.mark.parametrize("provider", [GOOGLE, EMAIL_LINK])
def test_admin_sign_in_is_only_the_first_step(store, client, monkeypatch, provider):
    store.create_user("jane@example.com", "admin")
    assert _sign_in(client, monkeypatch, provider=provider)["next"] == "/2fa"
    assert client.get("/whoami").get_json()["email"] is None, "no session until the code is checked"


def test_email_on_admin_list_becomes_admin_and_needs_the_code(store, client, monkeypatch):
    monkeypatch.setenv("TWIN_ADMIN_EMAILS", "jane@example.com")
    assert _sign_in(client, monkeypatch)["next"] == "/2fa"
    assert store.get_by_email("jane@example.com")["role"] == "admin"


def test_admin_enrols_authenticator_then_needs_codes(store, client, monkeypatch):
    clock = _Clock(1_800_000_000.0)
    monkeypatch.setattr(totp, "time", clock)
    store.create_user("jane@example.com", "admin")
    _sign_in(client, monkeypatch)
    secret = client.get("/setup-secret").get_json()["secret"]
    assert "/2fa?error=" in _code(client, "000000").headers["Location"]
    enrol_code = totp.code_now(secret)
    assert _code(client, enrol_code).headers["Location"].endswith("/2fa-codes")
    assert client.get("/whoami").get_json() == {"email": "jane@example.com", "role": "admin", "mfa_pending": False}
    assert store.get_by_email("jane@example.com")["totp_enabled"]

    # Next sign-in: the app's code is required, and the enrolment code cannot be reused.
    client.get("/auth/logout")
    _sign_in(client, monkeypatch, provider=GOOGLE)
    assert "/2fa?error=" in _code(client, enrol_code).headers["Location"]
    clock.t += totp.STEP_SEC
    assert _code(client, totp.code_now(secret)).headers["Location"].endswith("/admin")
    assert client.get("/whoami").get_json()["role"] == "admin"


def test_recovery_codes_work_once(store, client, monkeypatch):
    admin = store.create_user("jane@example.com", "admin")
    codes = totp.new_recovery_codes()
    store.enable_totp(admin["user_id"], totp.new_secret(), codes, 0)
    stored = sqlite3.connect(store.path).execute("SELECT recovery_codes FROM users").fetchone()[0]
    assert codes[0] not in stored, "recovery codes are stored only as hashes"
    _sign_in(client, monkeypatch)
    assert _code(client, codes[0].upper()).headers["Location"].endswith("/admin")
    assert store.recovery_codes_left(admin["user_id"]) == len(codes) - 1
    client.get("/auth/logout")
    _sign_in(client, monkeypatch)
    assert "/2fa?error=" in _code(client, codes[0]).headers["Location"]


def test_wrong_codes_lock_the_sign_in(store, client, monkeypatch):
    admin = store.create_user("jane@example.com", "admin")
    store.enable_totp(admin["user_id"], totp.new_secret(), totp.new_recovery_codes(), 0)
    _sign_in(client, monkeypatch)
    for _ in range(routes.MAX_FAILURES):
        _code(client, "123456")
    assert "Too+many" in _code(client, "123456").headers["Location"]


def test_promoted_admin_keeps_citizen_powers_until_the_code_step(store, client, monkeypatch):
    store.create_user("root@example.com", "admin")
    jane = store.create_user("jane@example.com", "planner")
    _sign_in(client, monkeypatch)
    store.update_user(jane["user_id"], role="admin")
    me = client.get("/whoami").get_json()
    assert me["role"] == "citizen" and me["mfa_pending"], "no admin rights without a second factor"


def test_pending_sign_in_expires(store, client, monkeypatch):
    store.create_user("jane@example.com", "admin")
    _sign_in(client, monkeypatch)
    monkeypatch.setattr(access.time, "time", lambda: 10 ** 10)
    assert "/login?error=" in _code(client, "123456").headers["Location"]


def test_code_form_without_csrf_token_is_refused(store, client, monkeypatch):
    store.create_user("jane@example.com", "admin")
    _sign_in(client, monkeypatch)
    assert "/login?error=" in client.post("/auth/2fa", data={"code": "123456"}).headers["Location"]


# ------------------------------------------------------- administration --
def test_last_admin_cannot_be_demoted_or_disabled(store):
    root = store.create_user("root@example.com", "admin")
    with pytest.raises(AuthError):
        store.update_user(root["user_id"], role="planner")
    with pytest.raises(AuthError):
        store.update_user(root["user_id"], active=False)
    store.create_user("second@example.com", "admin")
    assert store.update_user(root["user_id"], role="planner")["role"] == "planner"


def test_bootstrap_admin_from_environment(store, monkeypatch):
    from src.auth.store import bootstrap_admin
    monkeypatch.setenv("TWIN_ADMIN_EMAIL", "root@example.com")
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
        INSERT INTO users (username, email, password_hash, role, created_at)
            VALUES ('old', 'old@example.com', 'scrypt:x', 'planner', 'x');
    """)
    conn.commit()
    conn.close()
    s = AuthStore(path)
    assert s.get_by_email("old@example.com")["role"] == "planner"
    assert s.create_user("new@example.com", provider="email_link")["auth_provider"] == "email_link"
    assert sqlite3.connect(path).execute("SELECT password_hash FROM users WHERE username='old'").fetchone()[0] is None


def test_audit_log_is_append_only_in_order(store):
    """FR-15 / DR-09: entries accumulate; nothing in the store updates or deletes them."""
    store.audit("first")
    store.audit("second", detail="x" * 1000)
    assert [a["action"] for a in store.recent_audit()] == ["second", "first"]
    assert len(store.recent_audit()[0]["detail"]) == 500


def test_alert_subscriptions_are_saved_per_user(store):
    """FR-03: a citizen's chosen areas are recorded against their profile."""
    u = store.create_user("alice@example.com")
    store.set_subscriptions(u["user_id"], ["Mathare", "Kibera", "Mathare", " "])
    assert store.subscriptions(u["user_id"]) == ["Kibera", "Mathare"]
    store.set_subscriptions(u["user_id"], ["Eastleigh"])
    assert store.subscriptions(u["user_id"]) == ["Eastleigh"]


def test_dashboard_figures_count_roles_sign_ins_and_failures(store):
    from datetime import datetime, timezone
    store.create_user("root@example.com", "admin")
    p = store.create_user("p@example.com", "planner")
    store.create_user("c@example.com", "citizen")
    store.set_subscriptions(p["user_id"], ["Mathare"])
    store.audit("login", p["user_id"])
    store.audit("login", p["user_id"])
    store.audit("login_failed", None, "firebase: unverified")
    assert store.role_counts() == {"citizen": 1, "planner": 1, "responder": 0, "admin": 1}
    assert store.count_actions(("login",), "2000-01-01") == 2
    assert store.count_actions(("login_failed", "2fa_failed"), "2000-01-01") == 1
    today = datetime.now(timezone.utc).date().isoformat()
    assert store.actions_by_day(("login",), "2000-01-01") == {today: 2}
    assert store.subscriber_count() == 1


def test_user_search_role_filter_and_activity_filter(store):
    from src.dashboard.admin_dashboard import audit_table, users_table
    store.create_user("root@example.com", "admin")
    store.create_user("planner@example.com", "planner")
    store.audit("login", None)
    store.audit("export_geojson", None, "What-if 60 mm")
    assert "planner@example.com" in str(users_table("plann")) and "root@" not in str(users_table("plann"))
    assert "root@example.com" in str(users_table(None, "admin")) and "planner@" not in str(users_table(None, "admin"))
    assert "No users match" in str(users_table("nobody"))
    data_only = str(audit_table(None, "data"))
    assert "Flood map exported" in data_only and "Signed in" not in data_only
    assert "Flood map exported" in str(audit_table("60 mm", "all"))


# ------------------------------------------------- server-side enforcement --
@needs_model
def test_whatif_is_refused_on_the_server_for_roles_without_it(store):
    """A What-if request that bypasses the hidden control is still held to live mode."""
    from flask import session
    from src.dashboard.callbacks import _allowed_mode
    app = Flask(__name__)
    app.secret_key = "test"
    planner = store.create_user("alice@example.com", "planner")
    citizen = store.create_user("bob@example.com", "citizen")
    with app.test_request_context():
        assert _allowed_mode("WHATIF") == "LIVE", "signed out"
        session["uid"] = citizen["user_id"]
        assert _allowed_mode("WHATIF") == "LIVE", "citizen"
        session["uid"] = planner["user_id"]
        assert _allowed_mode("WHATIF") == "WHATIF", "planner"


@needs_model
def test_replays_are_staff_only_in_the_menu_and_on_the_server(store):
    """The public and citizens get the live forecast only; a replay request from them is refused."""
    from flask import session
    from src.dashboard.callbacks import _allowed_source, source_options
    from src.forecast.nowcast import SOURCES
    replay = next(k for k in SOURCES if k != "live")
    app = Flask(__name__)
    app.secret_key = "test"
    citizen = store.create_user("bob@example.com", "citizen")
    responder = store.create_user("cara@example.com", "responder")
    assert [o["value"] for o in source_options(None)] == ["live"]
    assert [o["value"] for o in source_options(citizen)] == ["live"]
    assert [o["value"] for o in source_options(responder)] == list(SOURCES)
    with app.test_request_context():
        assert _allowed_source(replay) == "live", "signed out"
        session["uid"] = citizen["user_id"]
        assert _allowed_source(replay) == "live", "citizen"
        assert _allowed_source("not-a-source") == "live"
        session["uid"] = responder["user_id"]
        assert _allowed_source(replay) == replay, "responder"


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


def test_alert_areas_keep_the_place_people_picked(store):
    """A landmark is followed through its neighbourhood; the person's pick is kept as the label."""
    u = store.create_user("alice@example.com")
    store.set_subscriptions(u["user_id"], [{"area": "Siwaka Estate", "label": "Strathmore University"},
                                           {"area": "Siwaka Estate", "label": "Strathmore Library"},
                                           "Mathare"])
    assert store.subscriptions(u["user_id"]) == ["Mathare", "Siwaka Estate"]
    assert store.subscription_items(u["user_id"]) == [
        {"area": "Mathare", "label": "Mathare"},
        {"area": "Siwaka Estate", "label": "Strathmore University"}]


def test_alert_subscriptions_table_gains_a_label_column(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE alert_subscriptions (subscription_id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL,
            sub_county TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1, UNIQUE (user_id, sub_county));
        INSERT INTO alert_subscriptions (user_id, sub_county) VALUES (1, 'Kibera');
    """)
    conn.commit()
    conn.close()
    assert AuthStore(path).subscription_items(1) == [{"area": "Kibera", "label": "Kibera"}]


@needs_model
def test_alert_search_finds_landmarks_and_the_server_works_out_their_neighbourhood(store):
    """Places the trip planner knows (e.g. Strathmore University) can be followed for alerts."""
    from src.dashboard.auth_pages import alert_area_search, resolve_alert_value
    from src.routing.flood_router import place_name_at
    from src.routing.geocoder import decode_value
    opts = alert_area_search("strathmore university", [])
    pick = next(o for o in opts if o["value"].startswith("loc:") and "Strathmore University" in o["label"])
    lat, lon, _name = decode_value(pick["value"])
    resolved = resolve_alert_value(pick["value"])
    assert resolved == {"area": place_name_at(lat, lon), "label": "Strathmore University"}
    assert f"alerts for {resolved['area']}" in pick["label"]
    assert resolve_alert_value("sub:Not A Real Place|x") is None, "made-up neighbourhoods are refused"
    assert resolve_alert_value("loc:5.0,40.0|Somewhere far") is None, "places outside the map are refused"
    assert resolve_alert_value("sub:Mathare|Mathare") == {"area": "Mathare", "label": "Mathare"}


def test_alerts_show_the_picked_place(store):
    from src.dashboard.auth_pages import _alerts_for
    u = store.create_user("alice@example.com")
    store.set_subscriptions(u["user_id"], [{"area": "Siwaka Estate", "label": "Strathmore University"}])
    nowcast = {"warnings": [{"kind": "starts", "hour": 2, "level": "MODERATE", "text": "Moderate flooding",
                             "places": ["Siwaka Estate", "Kibera"], "places_text": "Siwaka Estate, Kibera"}]}
    page = str(_alerts_for(u, nowcast))
    assert "Siwaka Estate (Strathmore University)" in page and "Kibera" not in page
