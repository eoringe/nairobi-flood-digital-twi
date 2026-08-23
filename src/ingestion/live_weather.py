"""
src.ingestion.live_weather
===========================
Nairobi Urban Flood Digital Twin — Real-Time Live Weather Sync

PURPOSE
-------
Fetch live real-time precipitation forecast for Nairobi from Open-Meteo API.
Allows the dashboard to make automatic real-time predictions without manual sliders.
"""

from __future__ import annotations

import json
import urllib.request
from loguru import logger


def fetch_live_nairobi_weather() -> dict:
    """
    Fetch live 7-day weather forecast for Nairobi (-1.286389, 36.817222).
    Returns today's expected rainfall (mm/day) and 7-day maximum peak forecast.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=-1.286389&longitude=36.817222"
        "&daily=precipitation_sum,precipitation_probability_max"
        "&timezone=Africa%2FNairobi"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NairobiDigitalTwin/1.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        daily = data.get("daily", {})
        precip_sums = daily.get("precipitation_sum", [0.0])
        precip_probs = daily.get("precipitation_probability_max", [0])

        today_rain = float(precip_sums[0]) if precip_sums else 0.0
        max_rain_7d = float(max(precip_sums)) if precip_sums else 0.0
        today_prob = int(precip_probs[0]) if precip_probs else 0

        logger.info(f"Live Weather Sync: Today={today_rain}mm, 7-Day Peak={max_rain_7d}mm, Prob={today_prob}%")

        return {
            "success": True,
            "today_rainfall_mm": round(today_rain, 1),
            "peak_7day_rainfall_mm": round(max_rain_7d, 1),
            "rain_probability_pct": today_prob,
            "source": "Open-Meteo Live API (Nairobi Station)",
        }
    except Exception as e:
        logger.warning(f"Could not fetch live weather: {e}. Using fallback forecast.")
        return {
            "success": False,
            "today_rainfall_mm": 18.5,
            "peak_7day_rainfall_mm": 45.0,
            "rain_probability_pct": 65,
            "source": "Live Cache Fallback (Nairobi Central)",
        }
