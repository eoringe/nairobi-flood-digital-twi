"""
src.integrations.base
=====================
What every data source looks like to the dashboard (specification §2.15, §8.2,
§9, §10.2).

A source is fetched through `DataSource.fetch(what, **params)`, which always
returns a `SourceResult` and never raises. The result says where the records
came from, when they were observed, how fresh they are, and whether the source
was reachable, so a screen can show an honest state instead of a blank or, worse,
a stale value presented as live.

FRESHNESS
---------
    live         newest observation within the source's live window
    delayed      older than that, but within its stale window
    stale        older than the stale window: shown, marked, never as current
    historical   an archive (e.g. past news reports): its age is the point
    unavailable  no observation at all

A missing reading is a missing reading. It is never turned into zero.

PARTNER SYSTEMS
---------------
`HttpSource` is the base for systems reached over HTTP (the IoT platform, the
alert system and so on). It reads its base URL and key from environment
variables. Unset, the source is *not connected* and every call returns that
state at once, without a network request.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

LIVE, DELAYED, STALE, HISTORICAL, UNAVAILABLE = "live", "delayed", "stale", "historical", "unavailable"
#: Result states.
OK, NOT_CONNECTED, ERROR = "ok", "not_connected", "error"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(t: datetime | str | None) -> datetime | None:
    """Timestamps are compared in UTC. A naive time is taken as Nairobi local (UTC+3)."""
    if t is None:
        return None
    if isinstance(t, str):
        t = datetime.fromisoformat(t.replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone(timedelta(hours=3)))
    return t.astimezone(timezone.utc)


def freshness(observed_at: datetime | str | None, live_within: timedelta, stale_after: timedelta,
              now: datetime | None = None) -> str:
    """Classify how current an observation is."""
    t = as_utc(observed_at)
    if t is None:
        return UNAVAILABLE
    age = (now or utc_now()) - t
    if age <= live_within:
        return LIVE
    if age <= stale_after:
        return DELAYED
    return STALE


@dataclass
class SourceResult:
    """Records from one source with everything needed to display them honestly."""
    source: str
    state: str                                   # ok | not_connected | error
    records: list[dict] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=utc_now)
    observed_at: datetime | None = None          # newest source timestamp in the records
    freshness: str = UNAVAILABLE
    message: str = ""
    demo: bool = False

    @property
    def ok(self) -> bool:
        return self.state == OK

    def to_dict(self) -> dict:
        return {
            "source": self.source, "state": self.state, "freshness": self.freshness,
            "fetched_at": self.fetched_at.isoformat(),
            "observed_at": self.observed_at.isoformat() if self.observed_at else None,
            "message": self.message, "demo": self.demo, "count": len(self.records),
            "records": self.records,
        }


class DataSource:
    """
    One integrated data source. Subclasses set the class attributes and
    implement `_fetch`, which returns (records, newest observation time).
    """
    key: str = ""
    name: str = ""
    system: str = ""                  # the specification's name for the component
    kind: str = "internal"            # internal | partner | external
    description: str = ""
    provides: tuple[str, ...] = ()    # what `fetch` accepts
    live_within: timedelta = timedelta(hours=1)
    stale_after: timedelta = timedelta(hours=6)
    archive: bool = False             # historical records: never "live"
    probe: str | None = None          # the cheapest request, used to check the source is up

    def classify(self, observed_at) -> str:
        if observed_at is None:
            return UNAVAILABLE
        return HISTORICAL if self.archive else freshness(observed_at, self.live_within, self.stale_after)

    def configured(self) -> bool:
        return True

    def demo(self) -> bool:
        return False

    def fetch(self, what: str, **params: Any) -> SourceResult:
        """Fetch, time and record the call; never raises."""
        from src.integrations.store import get_ops_store
        if what not in self.provides:
            return SourceResult(self.key, ERROR, message=f"{self.name} does not provide '{what}'.")
        if not self.configured() and not self.demo():
            return SourceResult(self.key, NOT_CONNECTED, message=self.not_connected_message())
        started = time.perf_counter()
        try:
            records, observed_at = self._fetch(what, **params)
        except Exception as exc:                                    # noqa: BLE001
            ms = (time.perf_counter() - started) * 1000
            logger.warning(f"{self.key}.{what} failed: {exc}")
            get_ops_store().record_call(self.key, ok=False, latency_ms=ms, error=str(exc)[:300])
            return SourceResult(self.key, ERROR, message=f"{self.name} could not be reached: {exc}")
        ms = (time.perf_counter() - started) * 1000
        get_ops_store().record_call(self.key, ok=True, latency_ms=ms, observed_at=observed_at)
        observed = as_utc(observed_at)
        return SourceResult(self.key, OK, records=records, observed_at=observed,
                            freshness=self.classify(observed),
                            demo=self.demo())

    def not_connected_message(self) -> str:
        return f"{self.system or self.name} is not connected yet."

    def _fetch(self, what: str, **params: Any) -> tuple[list[dict], datetime | None]:
        raise NotImplementedError

    def info(self) -> dict:
        return {"key": self.key, "name": self.name, "system": self.system, "kind": self.kind,
                "description": self.description, "provides": list(self.provides),
                "configured": self.configured(), "demo": self.demo()}


class HttpSource(DataSource):
    """A partner system reached over HTTP, configured by environment variables."""
    kind = "partner"
    env_url: str = ""
    env_key: str = ""
    timeout_s: float = 10.0
    retries: int = 2

    def base_url(self) -> str:
        return os.environ.get(self.env_url, "").rstrip("/")

    def configured(self) -> bool:
        return bool(self.base_url())

    def demo(self) -> bool:
        return not self.configured() and os.environ.get("INTEGRATION_DEMO") == "1"

    def not_connected_message(self) -> str:
        return (f"{self.system} is not connected yet. Set {self.env_url} (and {self.env_key}) "
                f"when its API is available.")

    def _get(self, path: str, params: dict | None = None) -> Any:
        """GET with a timeout and retries with backoff on network errors and 5xx."""
        import requests
        headers = {"Accept": "application/json", "User-Agent": "nairobi-flood-dashboard/1.0"}
        if os.environ.get(self.env_key):
            headers["Authorization"] = f"Bearer {os.environ[self.env_key]}"
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.get(f"{self.base_url()}{path}", params=params, headers=headers, timeout=self.timeout_s)
            except Exception as exc:                                # noqa: BLE001  network error: retry
                last = exc
            else:
                if r.status_code < 400:
                    return r.json()
                if r.status_code < 500:                             # our request is wrong: retrying won't help
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                last = RuntimeError(f"HTTP {r.status_code}")
            if attempt < self.retries:
                time.sleep(0.5 * 2 ** attempt)
        raise RuntimeError(str(last))
