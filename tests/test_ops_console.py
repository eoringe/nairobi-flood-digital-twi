"""
tests.test_ops_console
======================
Specification roles, the permission matrix, the account-store upgrade, and
the operations console pages (docs/capstone/TASKLIST.md Phase 2).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from src.auth.access import PERMISSIONS, ROLE_DUTIES, ROLE_LABELS, can
from src.auth.store import ROLES, AuthStore
from tests.conftest import sqlite_only
from src.integrations import base, registry
from src.integrations.store import OpsStore, set_ops_store



def _user(role: str, uid: int = 1) -> dict:
    return {"user_id": uid, "role": role, "active": True, "email": f"{role}@example.org"}


def test_the_four_roles_exist_with_a_label_and_duty():
    assert set(ROLES) == {"citizen", "planner", "responder", "admin"}
    assert set(ROLES) == set(ROLE_LABELS) == set(ROLE_DUTIES)
    for roles in PERMISSIONS.values():
        assert roles <= set(ROLES), "a permission names an unknown role"


@pytest.mark.parametrize("role, permission, allowed", [
    ("citizen", "view_ops", False),
    ("responder", "view_ops", True),
    ("planner", "view_ops", True),
    ("admin", "view_system_status", True),
    ("planner", "view_system_status", False),
    ("planner", "export_readings", True),
    ("responder", "export_readings", False),
    ("responder", "view_beneficiaries", True),
    ("planner", "view_beneficiaries", False),
    ("planner", "view_infrastructure", True),
    ("planner", "run_whatif", True),
    ("responder", "run_whatif", False),
    ("responder", "moderate_reports", True),
    ("admin", "manage_users", True),
    ("responder", "manage_users", False),
])
def test_permission_matrix(role, permission, allowed):
    assert can(_user(role), permission) is allowed


def test_a_disabled_account_has_no_permissions():
    assert not can({**_user("admin"), "active": False}, "view_ops")


@sqlite_only
def test_accounts_on_retired_roles_move_to_the_four_roles(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    retired = ",".join(f"'{r}'" for r in ("citizen", "planner", "responder", "admin", "monitoring_officer",
                                          "hydrologist", "data_analyst", "emergency_officer", "county_officer",
                                          "infrastructure_officer"))
    conn.executescript(f"""
        CREATE TABLE users (user_id INTEGER PRIMARY KEY, username TEXT NOT NULL UNIQUE, email TEXT UNIQUE,
            display_name TEXT, password_hash TEXT,
            auth_provider TEXT NOT NULL DEFAULT 'local' CHECK (auth_provider IN ('local','google','email_link')),
            role TEXT NOT NULL DEFAULT 'citizen' CHECK (role IN ({retired})),
            active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, last_login_at TEXT,
            totp_secret TEXT, totp_last_step INTEGER, recovery_codes TEXT, assigned_areas TEXT);
        INSERT INTO users (username, email, role, created_at) VALUES ('p', 'p@example.org', 'planner', '2026-01-01');
        INSERT INTO users (username, email, role, created_at) VALUES ('h', 'h@example.org', 'hydrologist', '2026-01-01');
        INSERT INTO users (username, email, role, created_at, assigned_areas)
            VALUES ('c', 'c@example.org', 'county_officer', '2026-01-01', '["Kibra"]');
    """)
    conn.commit(); conn.close()
    store = AuthStore(path)
    roles = {u["email"]: u["role"] for u in store.list_users()}
    assert roles == {"p@example.org": "planner", "h@example.org": "planner", "c@example.org": "responder"}
    assert store.get_by_email("c@example.org")["assigned_areas"] == []
    with pytest.raises(Exception):
        store.create_user("x@example.org", "hydrologist")
    ddl = sqlite3.connect(path).execute("SELECT sql FROM sqlite_master WHERE name='users'").fetchone()[0]
    assert "hydrologist" not in ddl, "the retired roles are no longer accepted by the table"


# ----------------------------------------------------------------- pages --
class _Fake(base.DataSource):
    """A source returning fixed records; `records` may map each request to its own list."""

    def __init__(self, key, state=base.OK, records=None, observed=None, demo=False):
        self.key, self.name, self.system = key, key.title(), f"{key} system"
        self.state, self.records, self.observed, self.is_demo = state, records, observed, demo
        self.provides = ("x",)

    def fetch(self, what, **params):
        recs = self.records.get(what, []) if isinstance(self.records, dict) else (self.records or [])
        return base.SourceResult(self.key, self.state, recs, observed_at=self.observed,
                                 freshness=base.LIVE if self.observed else base.UNAVAILABLE, demo=self.is_demo,
                                 message="" if self.state == base.OK else "not connected")


@pytest.fixture
def fake_sources(tmp_path, monkeypatch):
    set_ops_store(OpsStore(tmp_path / "ops.db"))
    now = datetime.now(timezone.utc)
    nc = {"warnings": [{"level": "HIGH", "places": ["Mathare", "Mabatini"], "text": "High flooding expected"},
                       {"level": "MODERATE", "places": ["Kibera"], "text": "Moderate flooding expected"}]}
    fakes = {
        "prediction": _Fake("prediction", observed=now, records={
            "outlook": [nc], "chance": [{"p": 0.41, "call": "possible", "call_text": "Flooding possible"}]}),
        "iot": _Fake("iot", state=base.NOT_CONNECTED),
        "forecast": _Fake("forecast", records=[{"rain_mm": 2.5, "rain_72h_mm": 31.0, "forecast": False}], observed=now),
        "outlook_alerts": _Fake("outlook_alerts", records=[{"areas": ["Mathare", "Kibera"]}], observed=now),
        "alerts": _Fake("alerts", state=base.NOT_CONNECTED),
        "community": _Fake("community", observed=now),
        "infrastructure_system": _Fake("infrastructure_system", state=base.NOT_CONNECTED),
        "infrastructure": _Fake("infrastructure", observed=now, records=[
            {"name": "Mathare North Health Centre", "category": "health_centre", "level": 2}]),
    }
    monkeypatch.setattr(registry, "SOURCES", fakes)
    monkeypatch.setattr(registry, "check_unknown", lambda: None)
    monkeypatch.setattr(registry, "status_rows", lambda: [])
    yield fakes
    set_ops_store(None)


def _text(component) -> str:
    """All text in a Dash component tree."""
    if component is None:
        return ""
    if isinstance(component, (str, int, float)):
        return str(component)
    if isinstance(component, (list, tuple)):
        return " ".join(_text(c) for c in component)
    return _text(getattr(component, "children", None))


def test_overview_hides_unconnected_partners_and_labels_model_output(fake_sources):
    from src.dashboard import ops_console
    page = _text(ops_console.overview(_user("responder"), "dark"))
    assert "Monitoring stations" not in page and "Not connected" not in page, "unconnected partners are left out"
    assert "41%" in page and "Flooding possible" in page
    assert "Mathare" in page and "High" in page, "high-risk areas are listed with the specification's class"
    assert "Model output, not observed flooding" in page
    assert "model warnings, 12-hour outlook" in page, "model warnings are not passed off as published alerts"


def test_the_console_is_closed_to_the_public(fake_sources):
    from src.dashboard import ops_console
    assert "operations console is for" in _text(ops_console.render_ops(_user("citizen"), "/ops"))
    assert "Your role does not include" in _text(ops_console.render_ops(_user("planner"), "/ops/status"))


def test_overview_counts_interim_infrastructure_from_openstreetmap(fake_sources):
    from src.dashboard import ops_console
    page = _text(ops_console.overview(_user("responder"), "dark"))
    assert "(OpenStreetMap)" in page and "Exposed infrastructure" in page


def test_area_and_risk_filters_narrow_the_risk_table(fake_sources):
    from src.dashboard import ops_console
    from src.geo import boundaries
    mathare = boundaries.ids_by_name(["Mathare"])
    page = _text(ops_console.overview(_user("responder"), "dark", {"areas": mathare, "min_risk": 0}))
    assert "Mabatini" in page and "Kibera" not in page
    page = _text(ops_console.overview(_user("responder"), "dark", {"areas": [], "min_risk": 3}))
    assert "at this risk level" in page, "no Very High places in the fake outlook"


def test_staff_see_the_whole_city_by_default():
    from src.dashboard.ops_console import effective_filter
    assert effective_filter(_user("planner"), None, 0)["areas"] == []


def test_boundaries_place_lookup():
    from src.geo import boundaries
    ward = boundaries.place_areas()["Mabatini"]["ward"]
    assert boundaries.label(ward) == "Mabatini ward, Mathare"
    assert boundaries.place_in("Mabatini", boundaries.ids_by_name(["Mathare"]))
    assert not boundaries.place_in("Mabatini", boundaries.ids_by_name(["Kibra"]))
    assert len(boundaries.load()["sub_county"]) == 19


def test_an_admin_can_remove_a_user_but_not_the_last_admin(tmp_path):
    from src.auth.store import AuthError, AuthStore, set_store
    from src.dashboard import admin_dashboard
    store = AuthStore(tmp_path / "acc.db")
    set_store(store)
    try:
        admin = store.create_user("admin@example.org", "admin")
        gone = store.create_user("gone@example.org", "planner")
        store.audit("login", gone["user_id"], "Google")
        text = str(admin_dashboard.users_table())
        assert "del-user" in text and "Assigned areas" not in text
        assert store.delete_user(gone["user_id"])["email"] == "gone@example.org"
        assert store.get_by_email("gone@example.org") is None
        assert any(a["action"] == "login" for a in store.recent_audit(10)), "the activity log is kept"
        with pytest.raises(AuthError):
            store.delete_user(admin["user_id"])
    finally:
        set_store(None)
