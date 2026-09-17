"""
src.forecast.nowcast
====================
A 12-hour flood outlook: "expect moderate flooding in Mathare in about 2 hours".

HOW THE TIMING IS PRODUCED
--------------------------
The U-Net has no clock. It was trained on daily CHIRPS rainfall and maps a
3-day rainfall total to flood extent. It cannot, by itself, say when water
arrives. The timing here comes from the rainfall forecast instead:

1. Fetch hourly rainfall for Nairobi from Open-Meteo: the last 10 days and the
   next 2 days for a live outlook, or archived hours for a replay.
2. For each hour from now to +12 h, total the rain over the 72 hours ending at
   that hour. That is the model's input quantity, evaluated hour by hour.
3. Run the model on each total and find the first hour at which each place's
   flooding reaches Moderate, High or Critical.

So "moderate flooding in about 2 hours" means: at the current forecast, the
72-hour rainfall total reaches the level at which the model shows moderate
flooding there in about 2 hours. It is not a simulation of water flowing through
streets, and it is only as good as the hourly rainfall forecast, which for
tropical convective storms is weak (LIMITATIONS.md section 11).

Places are assigned a band when at least MIN_CELLS grid cells in their area reach
it, on the same smoothed field the map draws, so a single speckle cell does not
raise a warning.

NO FABRICATED FALLBACK
----------------------
If the forecast cannot be fetched the outlook says so and carries no warnings.
It never substitutes invented rainfall figures.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.ndimage import gaussian_filter

from src.forecast.rain_bias_correction import correct_3day_total, is_available as bias_correction_available
from src.routing.flood_router import LEVEL_NAMES, MAP_SMOOTHING_SIGMA, level_of, place_raster

NAIROBI_LAT, NAIROBI_LON = -1.2864, 36.8172
EAT = timezone(timedelta(hours=3))            # Nairobi, no daylight saving
HORIZON_H = 12
WINDOW_H = 72
#: ~0.03 km2. A place is warned only when this many cells reach a band.
MIN_CELLS = 6
LIVE_TTL_SEC = 20 * 60

FORECAST_URL = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                "&hourly=precipitation&past_days=10&forecast_days=2&timezone=Africa%2FNairobi")
#: Replays use Open-Meteo's archive of its own FORECASTS, the same feed live mode
#: reads, so a replay exercises the real pipeline, bias correction included. The
#: reanalysis archive (ERA5) is a different product with a different bias.
ARCHIVE_URL = ("https://historical-forecast-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               "&start_date={start}&end_date={end}&hourly=precipitation&timezone=Africa%2FNairobi")

#: Outlook sources. Replays re-run the outlook at a past moment using archived
#: hourly rainfall, so the warning logic can be demonstrated on a real storm.
#: In a replay the "forecast" hours are what actually fell - a perfect forecast
#: no live system would have - and the interface labels it as such.
#: Replay moments are chosen on documented Nairobi floods (validation/
#: validate_documented_events.py), at the hour the bias-corrected 72-hour total
#: was still below the flooding level but about to cross it.
SOURCES = {
    "live": {"label": "Now - live forecast", "as_of": None},
    "replay-2026-03-06": {"label": "Replay - 6 Mar 2026, 06:00 (Nairobi River floods)",
                          "as_of": datetime(2026, 3, 6, 6)},
    "replay-2024-04-24": {"label": "Replay - 24 Apr 2024, 10:00 (Mathare floods)",
                          "as_of": datetime(2024, 4, 24, 10)},
    "replay-2023-11-12": {"label": "Replay - 12 Nov 2023, 17:00 (El Nino rains)",
                          "as_of": datetime(2023, 11, 12, 17)},
}


@dataclass
class Nowcast:
    source: str
    label: str
    is_replay: bool
    as_of: datetime | None = None
    provider: str = ""
    error: str | None = None
    notice: str | None = None                                  # data present but degraded
    hours: list[datetime] = field(default_factory=list)        # as_of + 0..HORIZON_H
    acc72_mm: list[float] = field(default_factory=list)        # per outlook hour, CHIRPS scale
    acc72_raw_mm: list[float] = field(default_factory=list)    # per outlook hour, as forecast
    bias_corrected: bool = False
    chart_times: list[datetime] = field(default_factory=list)  # -24 h .. +12 h
    chart_rain_mm: list[float] = field(default_factory=list)
    chart_acc72_mm: list[float] = field(default_factory=list)
    place_levels: np.ndarray | None = None                     # (hours, places)
    flooded_pct: list[float] = field(default_factory=list)
    city_level: list[int] = field(default_factory=list)
    trend: str = ""
    warnings: list[dict] = field(default_factory=list)
    headline: str = ""
    detail: str = ""
    fetched_at: float = 0.0

    def rain_key(self, hour: int) -> float:
        return float(round(self.acc72_mm[hour]))

    def to_client(self) -> dict:
        """JSON-safe subset for a dcc.Store."""
        return {
            "source": self.source, "label": self.label, "is_replay": self.is_replay,
            "error": self.error, "notice": self.notice, "provider": self.provider,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "hours": [h.strftime("%H:%M") for h in self.hours],
            "acc72_mm": [round(a, 1) for a in self.acc72_mm],
            "acc72_raw_mm": [round(a, 1) for a in self.acc72_raw_mm],
            "bias_corrected": self.bias_corrected,
            "city_level": self.city_level, "flooded_pct": self.flooded_pct,
            "trend": self.trend, "headline": self.headline, "detail": self.detail,
            "warnings": self.warnings, "fetched_at": self.fetched_at,
            "chart_offsets": [int((t - self.as_of).total_seconds() // 3600) for t in self.chart_times],
            "chart_labels": [t.strftime("%H:%M") for t in self.chart_times],
            "chart_rain_mm": [round(v, 2) for v in self.chart_rain_mm],
            "chart_acc72_mm": [round(v, 1) for v in self.chart_acc72_mm],
            "as_of_label": self.as_of.strftime("%d %b %Y, %H:%M") if self.as_of else None,
        }


# ------------------------------------------------------------------ fetch --
def _get_json(url: str, timeout: float = 15.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "nairobi-flood-digital-twin/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


#: Archived rainfall for the replays, committed so a replay never needs the
#: internet - it is a fixed past event, and a presentation should not depend on
#: a third-party API answering. Written on first successful fetch.
REPLAY_FILE = Path("data/processed/replay_rainfall.json")
#: Last successful live forecast, used when Open-Meteo cannot be reached.
LIVE_CACHE_FILE = Path(os.environ.get("TWIN_STATE_DIR", "data/raw")) / "live_rainfall_cache.json"
#: How old a cached live forecast may be before it is refused. Beyond this the
#: 12-hour outlook would describe hours that have already passed.
LIVE_CACHE_MAX_AGE_H = 6
#: After a failed fetch, wait this long before trying again, so every page load
#: during an outage does not sit on a network timeout.
RETRY_AFTER_SEC = 60


def _parse(data: dict) -> dict[datetime, float]:
    # Open-Meteo precipitation at time T is the sum over the preceding hour.
    return {datetime.fromisoformat(t): float(v) for t, v in
            zip(data["hourly"]["time"], data["hourly"]["precipitation"]) if v is not None}


def _save_json(path: Path, payload: dict) -> None:
    """
    Best-effort cache write. On a host whose storage is read-only or owned by
    another user (a Railway volume mounts as root while the app runs as a
    non-root user), failing to cache must not take the forecast down with it.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError as exc:
        logger.warning(f"Could not write {path} ({exc}); continuing without caching it.")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        return {}


def _hourly_series(source: str, as_of: datetime | None):
    """
    Hourly rainfall keyed by local hour-ending time.

    Returns (series, as_of, provider, notice). `notice` is set when the data did
    not come from a fresh fetch, so the interface can say so.
    """
    if as_of is not None:                                      # replay
        stored = _read_json(REPLAY_FILE).get(source)
        if stored:
            return _parse(stored["data"]), as_of, stored["provider"], None
        start = (as_of - timedelta(days=10)).date().isoformat()
        end = (as_of + timedelta(days=2)).date().isoformat()
        data = _get_json(ARCHIVE_URL.format(lat=NAIROBI_LAT, lon=NAIROBI_LON, start=start, end=end), 60.0)
        provider = "Open-Meteo archived forecasts (hourly)"
        all_replays = _read_json(REPLAY_FILE)
        all_replays[source] = {"provider": provider, "as_of": as_of.isoformat(),
                               "data": {"hourly": {"time": data["hourly"]["time"],
                                                   "precipitation": data["hourly"]["precipitation"]}}}
        _save_json(REPLAY_FILE, all_replays)
        return _parse(data), as_of, provider, None

    try:
        data = _get_json(FORECAST_URL.format(lat=NAIROBI_LAT, lon=NAIROBI_LON), 10.0)
    except Exception as exc:                                   # noqa: BLE001
        cached = _read_json(LIVE_CACHE_FILE)
        if cached:
            issued = datetime.fromisoformat(cached["as_of"])
            now = datetime.now(EAT).replace(tzinfo=None)
            age_h = (now - issued).total_seconds() / 3600
            if age_h <= LIVE_CACHE_MAX_AGE_H:
                logger.warning(f"Live forecast fetch failed ({exc}); using the forecast issued "
                               f"{age_h:.1f} h ago")
                return (_parse(cached["data"]), issued, cached["provider"],
                        f"Offline: Open-Meteo could not be reached, so this is the forecast issued at "
                        f"{issued:%H:%M} ({age_h:.1f} h ago). Hours already past are no longer a forecast.")
        raise
    as_of = datetime.now(EAT).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    provider = "Open-Meteo forecast (hourly)"
    _save_json(LIVE_CACHE_FILE, {"as_of": as_of.isoformat(), "provider": provider,
                                 "data": {"hourly": {"time": data["hourly"]["time"],
                                                     "precipitation": data["hourly"]["precipitation"]}}})
    return _parse(data), as_of, provider, None


def _acc72(series: dict[datetime, float], end: datetime) -> tuple[float, int]:
    vals = [series.get(end - timedelta(hours=k)) for k in range(WINDOW_H)]
    return float(sum(v for v in vals if v is not None)), sum(v is None for v in vals)


# ---------------------------------------------------------------- analyse --
def _place_levels(probability: np.ndarray, place_idx: np.ndarray, n_places: int) -> np.ndarray:
    """Band level per place: highest band reached by at least MIN_CELLS cells."""
    lv = level_of(gaussian_filter(probability.astype(np.float32), sigma=MAP_SMOOTHING_SIGMA)).ravel()
    out = np.zeros(n_places, dtype=np.int8)
    for band in (1, 2, 3):
        counts = np.bincount(place_idx, weights=(lv >= band), minlength=n_places)
        out[counts >= MIN_CELLS] = band
    return out


def _trend(series: dict[datetime, float], as_of: datetime) -> str:
    past3 = sum(series.get(as_of - timedelta(hours=k), 0.0) for k in range(3))
    next3 = sum(series.get(as_of + timedelta(hours=k), 0.0) for k in range(1, 4))
    # The whole outlook window, not just the next three hours: a storm arriving in
    # hour 4 was otherwise described as "little or no rain is forecast" beside a
    # warning that flooding starts in four hours.
    later = sum(series.get(as_of + timedelta(hours=k), 0.0) for k in range(4, HORIZON_H + 1))
    if next3 >= 2.0 and next3 > 1.5 * past3:
        return "intensifying"
    if later >= 5.0 and later > 2.0 * next3:
        return "arriving"
    if past3 >= 2.0 and next3 < 0.5 * past3:
        return "easing"
    if next3 >= 1.0 or past3 >= 1.0:
        return "steady"
    return "dry"


def _in_hours(h: int) -> str:
    return "now" if h == 0 else ("in about 1 hour" if h == 1 else f"in about {h} hours")


def _list_places(names: list[str], k: int = 3) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) <= k:
        return f"{', '.join(names[:-1])} and {names[-1]}"
    return f"{', '.join(names[:k])} and {len(names) - k} more"


def _build_warnings(nc: Nowcast, names: list[str], sizes: np.ndarray) -> None:
    """
    Group places that share a warning: same kind, same starting band and time,
    same peak band. The time a place peaks is not part of the key - splitting on
    it produced a dozen one-place groups differing by an hour - so each group
    reports the earliest peak among its members.
    """
    L = nc.place_levels
    groups: dict[tuple, list[tuple[int, int]]] = {}
    for p in np.flatnonzero(L.max(axis=0) > 0):
        now, peak = int(L[0, p]), int(L[:, p].max())
        first = int(np.argmax(L[:, p] > 0))
        first_lvl = int(L[first, p])
        t_peak = int(np.argmax(L[:, p] == peak))
        if now > 0 and peak == now:
            key = ("ongoing", now, 0, now)
        elif now > 0:
            key = ("rising", now, 0, peak)
        elif peak > first_lvl:
            key = ("expected-rising", first_lvl, first, peak)
        else:
            key = ("expected", first_lvl, first, peak)
        groups.setdefault(key, []).append((p, t_peak))

    warnings = []
    for (kind, lvl, t_first, peak), members in groups.items():
        t_peak = min(tp for _p, tp in members)
        members.sort(key=lambda m: -sizes[m[0]])
        places = [names[p] for p, _tp in members]
        hh = lambda h: nc.hours[h].strftime("%H:%M")
        if kind == "ongoing":
            text = f"{LEVEL_NAMES[lvl].title()} flooding likely now"
        elif kind == "rising":
            text = (f"{LEVEL_NAMES[lvl].title()} flooding now, rising to "
                    f"{LEVEL_NAMES[peak].lower()} by {hh(t_peak)} ({_in_hours(t_peak)})")
        elif kind == "expected-rising":
            text = (f"{LEVEL_NAMES[lvl].title()} flooding expected from {hh(t_first)} "
                    f"({_in_hours(t_first)}), {LEVEL_NAMES[peak].lower()} by {hh(t_peak)}")
        else:
            text = f"{LEVEL_NAMES[lvl].title()} flooding expected from {hh(t_first)} ({_in_hours(t_first)})"
        warnings.append({
            "kind": kind, "level": LEVEL_NAMES[peak], "level_now": LEVEL_NAMES[lvl],
            "hour": t_first if kind.startswith("expected") else 0,
            "peak_hour": t_peak, "text": text,
            "places": places, "places_text": _list_places(places),
            "place_count": len(places),
        })
    # Most severe first; among equals, soonest first.
    warnings.sort(key=lambda w: (-list(LEVEL_NAMES.values()).index(w["level"]), w["hour"], -w["place_count"]))
    nc.warnings = warnings


def _events(nc: Nowcast) -> dict[tuple[int, int], list[str]]:
    """
    Future changes, keyed by (hour, band reached): an onset where a place starts
    flooding, or an escalation where it reaches its peak band. Ongoing flooding
    is not an event - nothing is about to change.
    """
    ev: dict[tuple[int, int], list[str]] = {}
    rank = {v: k for k, v in LEVEL_NAMES.items()}
    for w in nc.warnings:
        if w["kind"].startswith("expected"):
            ev.setdefault((w["hour"], rank[w["level_now"]]), []).extend(w["places"])
        if w["kind"] in ("rising", "expected-rising"):
            ev.setdefault((w["peak_hour"], rank[w["level"]]), []).extend(w["places"])
    return ev


def _headline(nc: Nowcast) -> None:
    """
    Lead with the soonest change, since that is what a person can still act on,
    then name the worst change still to come if it is a different one.
    """
    peak_acc = max(nc.acc72_mm)
    trend_phrase = {"intensifying": "rainfall is intensifying", "easing": "rainfall is easing",
                    "steady": "rain is falling steadily", "arriving": "heavy rain is forecast later in the outlook",
                    "dry": "little or no rain is forecast"}[nc.trend]
    if not nc.warnings:
        nc.headline = "No flooding expected in the next 12 hours"
        nc.detail = (f"72-hour rainfall {nc.acc72_mm[0]:.0f} mm now, peaking at {peak_acc:.0f} mm "
                     f"in this window; {trend_phrase}.")
        return

    events = _events(nc)
    if not events:
        worst = nc.warnings[0]
        nc.headline = f"{worst['level'].title()} flooding likely now in {worst['places_text']}"
        nc.detail = (f"No further rise expected in the next 12 hours; 72-hour rainfall "
                     f"{nc.acc72_mm[0]:.0f} mm and {trend_phrase}.")
        return

    (h, lvl), places = min(events.items(), key=lambda kv: (kv[0][0], -kv[0][1]))
    nc.headline = f"Expect {LEVEL_NAMES[lvl].lower()} flooding in {_list_places(places, 2)} {_in_hours(h)}"
    detail = (f"{trend_phrase.capitalize()}: the 72-hour total reaches {nc.acc72_mm[h]:.0f} mm by "
              f"{nc.hours[h].strftime('%H:%M')}, where the model shows this flooding.")
    (wh, wl), wplaces = max(events.items(), key=lambda kv: (kv[0][1], -kv[0][0]))
    if (wh, wl) != (h, lvl) and wl > lvl:
        detail += (f" Worst ahead: {LEVEL_NAMES[wl].lower()} in {_list_places(wplaces, 2)} "
                   f"by {nc.hours[wh].strftime('%H:%M')}.")
    nc.headline, nc.detail = nc.headline, detail


def build_nowcast(source: str, predictor) -> Nowcast:
    """Fetch rainfall and evaluate the model across the outlook window."""
    spec = SOURCES.get(source, SOURCES["live"])
    nc = Nowcast(source=source, label=spec["label"], is_replay=spec["as_of"] is not None,
                 fetched_at=time.time())
    try:
        series, as_of, provider, notice = _hourly_series(source, spec["as_of"])
    except Exception as exc:                                    # noqa: BLE001
        logger.warning(f"Nowcast {source}: rainfall fetch failed: {exc}")
        what = "Archived rainfall for this replay" if nc.is_replay else "The live rainfall forecast"
        nc.error = (f"{what} could not be downloaded from Open-Meteo, and no saved copy exists. "
                    f"No outlook is shown rather than a guessed one; it retries automatically.")
        nc.headline = "Forecast unavailable"
        return nc
    nc.notice = notice

    nc.as_of, nc.provider = as_of, provider
    nc.hours = [as_of + timedelta(hours=h) for h in range(HORIZON_H + 1)]
    missing = 0
    for t in nc.hours:
        acc, miss = _acc72(series, t)
        nc.acc72_raw_mm.append(acc)
        # The model learned CHIRPS totals; the forecast feed reports about half of
        # CHIRPS in storms. See src/forecast/rain_bias_correction.py.
        nc.acc72_mm.append(correct_3day_total(acc))
        missing = max(missing, miss)
    if missing > 6:
        nc.notice = (nc.notice + " " if nc.notice else "") + f"Rainfall record incomplete ({missing} of 72 hours missing); totals may be low."

    nc.chart_times = [as_of + timedelta(hours=h) for h in range(-24, HORIZON_H + 1)]
    nc.chart_rain_mm = [series.get(t, 0.0) for t in nc.chart_times]
    nc.chart_acc72_mm = [correct_3day_total(_acc72(series, t)[0]) for t in nc.chart_times]
    nc.bias_corrected = bias_correction_available()
    if not nc.bias_corrected:
        nc.notice = (nc.notice + " " if nc.notice else "") + (
            "Rainfall is NOT bias-corrected, so storms are underestimated by about half.")
    nc.trend = _trend(series, as_of)

    place_idx, names = place_raster()
    inside = place_idx[:-1]
    sizes = np.bincount(inside, minlength=len(names))
    levels, flooded = [], []
    for h in range(HORIZON_H + 1):
        prob = predictor.probability_for(nc.rain_key(h), scenario_date=as_of.date())
        levels.append(_place_levels(prob, inside, len(names)))
        flooded.append(round(100.0 * float((prob > 0.5).mean()), 2))
    nc.place_levels = np.vstack(levels)
    nc.flooded_pct = flooded
    nc.city_level = [int(l.max()) for l in levels]

    # Names can repeat in the gazetteer (a place tagged as both node and way a
    # little apart); merge them so a warning does not list "Mathare" twice.
    uniq: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        uniq.setdefault(n, []).append(i)
    if len(uniq) < len(names):
        cols = list(uniq.values())
        nc.place_levels = np.stack([nc.place_levels[:, c].max(axis=1) for c in cols], axis=1)
        sizes = np.array([sizes[c].sum() for c in cols])
        names = list(uniq.keys())

    _build_warnings(nc, names, sizes)
    _headline(nc)
    logger.info(f"Nowcast {source} as of {as_of:%Y-%m-%d %H:%M}: 72h {nc.acc72_mm[0]:.0f} -> "
                f"{max(nc.acc72_mm):.0f} mm, {len(nc.warnings)} warning groups. {nc.headline}")
    return nc


_CACHE: dict[str, Nowcast] = {}
_LOCK = threading.Lock()


def get_nowcast(source: str, predictor, refresh: bool = False) -> Nowcast:
    """
    Cached outlook. Live outlooks expire after LIVE_TTL_SEC; replays never change.
    Locked so the startup warm-up and a page load cannot build the same one twice.
    """
    with _LOCK:
        nc = _CACHE.get(source)
        age = time.time() - nc.fetched_at if nc is not None else 0.0
        stale = nc is not None and (
            (nc.error is not None and age > RETRY_AFTER_SEC)
            or (nc.notice is not None and not nc.is_replay and age > RETRY_AFTER_SEC)
            or (not nc.is_replay and age > LIVE_TTL_SEC))
        if nc is None or refresh or stale:
            nc = build_nowcast(source, predictor)
            _CACHE[source] = nc
        return nc


def cached_nowcast(source: str) -> Nowcast | None:
    """The outlook already built for a source, without fetching or computing."""
    return _CACHE.get(source)
