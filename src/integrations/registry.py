"""
src.integrations.registry
=========================
Every data source the dashboard reads (specification §8 DataSourceRegistry,
§9), and their current status for the system status screen (§6.12, UC-012).
"""

from __future__ import annotations

import os

from src.integrations.base import DataSource, as_utc
from src.integrations.internal import (CommunityReports, ForecastFeed, NewsFloods, OutlookAlerts,
                                       PredictionModel, SatelliteFloodMap)
from src.integrations.partners import (AutomatedAlertSystem, BeneficiarySystem, InfrastructureRiskSystem,
                                       IoTSensorPlatform, SupplyChainSystem)

SOURCES: dict[str, DataSource] = {s.key: s for s in (
    ForecastFeed(), PredictionModel(), OutlookAlerts(), CommunityReports(), NewsFloods(), SatelliteFloodMap(),
    IoTSensorPlatform(), AutomatedAlertSystem(), InfrastructureRiskSystem(), SupplyChainSystem(),
    BeneficiarySystem(),
)}


def get(key: str) -> DataSource:
    return SOURCES[key]


def register(source: DataSource) -> None:
    """Add a source (used by modules that bring their own, e.g. infrastructure)."""
    SOURCES[source.key] = source


def demo_mode() -> bool:
    return os.environ.get("INTEGRATION_DEMO") == "1"


def status_rows() -> list[dict]:
    """One row per source: configuration, last success/failure, failures in 24 h, freshness."""
    from src.integrations.store import get_ops_store
    store = get_ops_store()
    rows = []
    for s in SOURCES.values():
        st = store.status(s.key)
        if not s.configured() and not s.demo():
            state = "not_connected"
        elif st.get("last_failure") and (not st.get("last_success") or st["last_failure"] > st["last_success"]):
            state = "down"
        elif st.get("last_success"):
            state = "ok"
        else:
            state = "unknown"                 # configured but not called yet
        rows.append({
            **s.info(), "state": state,
            "last_success": st.get("last_success"), "last_failure": st.get("last_failure"),
            "last_error": st.get("last_error"), "latency_ms": st.get("last_latency_ms"),
            "last_observed": st.get("last_observed"),
            "freshness": s.classify(as_utc(st.get("last_observed"))),
            "failures_24h": st["failures_24h"], "calls_24h": st["calls_24h"],
        })
    return rows
