"""
tests.test_reports
==================
Community flood reports, routing around reported flooding, and the satellite
flood layer.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from src.reports.store import (CLEARS_TO_CLOSE, CONFIRM_TTL, RATE_LIMIT, REPORT_TTL, ReportError, ReportStore)
from tests.conftest import needs_roads

KICC = (-1.2884, 36.8233)


class _Clock:
    def __init__(self):
        self.t = datetime(2026, 4, 24, 10, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.t

    def advance(self, **kw):
        self.t += timedelta(**kw)


@pytest.fixture
def clock():
    return _Clock()


@pytest.fixture
def reports(tmp_path, clock):
    return ReportStore(tmp_path / "reports.db", now=clock)


def test_a_report_is_active_until_it_expires(reports, clock):
    r, merged = reports.add(1, *KICC, "water", "Central Business District", "ankle deep")
    assert not merged
    assert [a["report_id"] for a in reports.active()] == [r["report_id"]]
    clock.advance(seconds=REPORT_TTL.total_seconds() + 60)
    assert reports.active() == []


def test_a_nearby_report_confirms_the_existing_one_and_keeps_the_worse_kind(reports):
    first, _ = reports.add(1, *KICC, "water", "CBD")
    second, merged = reports.add(2, KICC[0] + 0.0005, KICC[1], "impassable", "CBD")   # ~55 m away
    assert merged and second["report_id"] == first["report_id"]
    assert second["confirmations"] == 1 and second["kind"] == "impassable"
    far, merged = reports.add(3, KICC[0] + 0.01, KICC[1], "water", "Ngara")            # ~1.1 km away
    assert not merged and far["report_id"] != first["report_id"]


def test_reporting_the_same_place_again_is_not_a_self_confirmation(reports):
    first, _ = reports.add(1, *KICC, "water", "CBD")
    again, merged = reports.add(1, *KICC, "water", "CBD")
    assert merged and again["report_id"] == first["report_id"]
    assert again["confirmations"] == 0


def test_confirmations_keep_a_report_alive(reports, clock):
    r, _ = reports.add(1, *KICC, "water", "CBD")
    clock.advance(hours=5)
    reports.vote(r["report_id"], 2, "still")
    clock.advance(seconds=(REPORT_TTL - timedelta(hours=5)).total_seconds() + 60)   # past the first expiry
    assert reports.active(), "a confirmation keeps it for CONFIRM_TTL from that moment"
    clock.advance(seconds=CONFIRM_TTL.total_seconds())
    assert reports.active() == []


def test_the_reporter_cannot_confirm_their_own_report(reports):
    r, _ = reports.add(1, *KICC, "water", "CBD")
    with pytest.raises(ReportError):
        reports.vote(r["report_id"], 1, "still")


def test_cleared_closes_a_report(reports):
    own, _ = reports.add(1, *KICC, "water", "CBD")
    assert reports.vote(own["report_id"], 1, "cleared")["status"] == "cleared", "the reporter can close it"

    other, _ = reports.add(1, KICC[0] + 0.02, KICC[1], "water", "Ngara")
    for voter in range(2, 2 + CLEARS_TO_CLOSE - 1):
        assert reports.vote(other["report_id"], voter, "cleared")["status"] == "active"
    assert reports.vote(other["report_id"], 99, "cleared")["status"] == "cleared"


def test_clears_do_not_outvote_more_confirmations(reports):
    r, _ = reports.add(1, *KICC, "homes", "CBD")
    for voter in (2, 3, 4):
        reports.vote(r["report_id"], voter, "still")
    for voter in (5, 6):
        reports.vote(r["report_id"], voter, "cleared")
    assert reports.get(r["report_id"])["status"] == "active"


def test_rate_limit(reports, clock):
    for i in range(RATE_LIMIT):
        reports.add(1, KICC[0] + 0.01 * i, KICC[1], "water", "somewhere")
    with pytest.raises(ReportError):
        reports.add(1, KICC[0] - 0.05, KICC[1], "water", "somewhere")
    clock.advance(hours=1, minutes=1)
    reports.add(1, KICC[0] - 0.05, KICC[1], "water", "somewhere")


def test_moderators_verify_and_remove(reports):
    r, _ = reports.add(1, *KICC, "water", "CBD")
    assert reports.verify(r["report_id"], 7)["verified_by"] == 7
    reports.remove(r["report_id"], 7)
    assert reports.active() == []


def test_unknown_kind_is_refused_and_notes_are_trimmed(reports):
    with pytest.raises(ReportError):
        reports.add(1, *KICC, "tsunami", "CBD")
    r, _ = reports.add(1, *KICC, "water", "CBD", "x" * 500)
    assert len(r["note"]) == 140


def test_report_permissions():
    from src.auth.access import can
    for role in ("citizen", "planner", "responder", "admin"):
        assert can({"role": role, "active": True}, "report_flood")
    assert not can(None, "report_flood")
    assert [r for r in ("citizen", "planner", "responder", "admin") if can({"role": r, "active": True}, "moderate_reports")] \
        == ["responder", "admin"]


def test_map_rows_never_carry_the_reporter(reports, monkeypatch):
    from src.dashboard import reports_ui
    reports.add(42, *KICC, "impassable", "CBD", "matatus turning back")
    rows = reports_ui.map_rows(reports.active())
    assert len(rows) == 1 and rows[0]["title"] == "Road impassable"
    assert "42" not in str(rows[0]) and "user" not in str(rows[0]).lower()


# ---------------------------------------------------------------- routing --
@needs_roads
def test_route_avoids_a_road_reported_impassable(router, places):
    dry = router.route(places["Eastleigh"], places["Kibera"], None)
    fast = dry["fastest"]
    # Report the middle of the fastest route as impassable.
    lon, lat = fast.coordinates[len(fast.coordinates) // 2]
    res = router.route(places["Eastleigh"], places["Kibera"], None,
                       reports=[{"lat": lat, "lon": lon, "kind": "impassable"}])
    assert res["fastest_reported"]
    assert not res["safe_reported"], "the flood-aware route should go around the reported road"
    assert res["safe"].node_path != res["fastest"].node_path


@needs_roads
def test_no_reports_changes_nothing(router):
    assert not router.report_probability([]).any()
    assert not router.report_probability(None).any()


# -------------------------------------------------------------- satellite --
def test_satellite_render_marks_flooding_and_what_radar_cannot_see():
    from PIL import Image
    import io
    from src.reports import satellite
    flood = np.zeros((10, 10, 4), np.uint8)
    flood[2:4, 2:4] = (112, 173, 71, 255)                 # observed flooding
    mask = np.zeros((10, 10, 4), np.uint8)
    mask[5:, :] = (133, 134, 134, 255)                    # built-up: radar cannot see
    png, flooded, visible = satellite.render(flood, mask)
    out = np.asarray(Image.open(io.BytesIO(png)))
    assert flooded == 4.0 and visible == 50.0
    assert tuple(out[3, 3]) == satellite.WATER_RGBA
    assert tuple(out[7, 7]) == satellite.UNSEEN_RGBA
    assert out[0, 0, 3] == 0


def test_latest_pass_is_the_newest_day_with_an_acquisition(monkeypatch):
    from src.reports import satellite
    passes = {date(2024, 4, 24), date(2024, 5, 1)}

    def fake_map(layer, day, w, h):
        a = np.zeros((h, w, 4), np.uint8)
        if day in passes:
            a[0, 0] = (133, 134, 134, 255)
        return a
    monkeypatch.setattr(satellite, "_get_map", fake_map)
    assert satellite.latest_pass(date(2024, 5, 3)) == date(2024, 5, 1)
    assert satellite.latest_pass(date(2024, 4, 30)) == date(2024, 4, 24)
    assert satellite.latest_pass(date(2024, 4, 20), days=3) is None
