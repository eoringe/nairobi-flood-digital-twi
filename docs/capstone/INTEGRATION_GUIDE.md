# Integration Guide for Partner Systems

How the Flood Visualization Dashboard connects to the other components of the
Integrated Flood Monitoring System, and what each team needs to provide.

The dashboard only **reads** from partner systems. It never creates, approves,
publishes or cancels anything in them (specification §2.8).

## How a connection works

Each partner system is reached through one adapter in
`src/integrations/partners.py`. An adapter is switched on by two environment
variables on the dashboard server:

| system | base URL | key |
|---|---|---|
| IoT Sensor Data Platform | `IOT_API_URL` | `IOT_API_KEY` |
| Automated Alert System | `ALERTS_API_URL` | `ALERTS_API_KEY` |
| Infrastructure Risk Mapping System | `INFRA_API_URL` | `INFRA_API_KEY` |
| Supply Chain Management System | `SUPPLY_API_URL` | `SUPPLY_API_KEY` |
| Student Beneficiary Management System | `BENEFICIARY_API_URL` | `BENEFICIARY_API_KEY` |

- **Requests.** Every request is an HTTPS `GET` with `Accept: application/json`
  and `Authorization: Bearer <key>`, when a key is set.
- **Timeout and retries.** Timeout is 10 s. Network errors and `5xx` responses
  are retried twice, with 0.5 s and 1 s back-off. A `4xx` response is not
  retried.
- **When a system is down or not configured.** The dashboard shows it as down
  or not connected. The rest of the dashboard keeps working (§10.2), and
  nothing is filled in.
- **Timestamps.** Use ISO 8601 with a time zone, for example
  `2026-11-01T09:40:00+03:00`. A time without a zone is read as Nairobi time.
- **Missing values.** Send `null`, never `0`. The dashboard shows `null` as a
  gap.

If your API's shape differs from what follows, tell the dashboard team. Only
your adapter changes; no screen does.

## IoT Sensor Data Platform

**`GET /stations`**: every registered station.

```json
[
  {
    "id": "WL-017",
    "name": "Nairobi River at Museum Hill",
    "types": ["water_level", "rainfall"],
    "lat": -1.2745, "lon": 36.8155,
    "status": "active",
    "health": "ok",
    "last_communication": "2026-11-01T09:40:00+03:00",
    "warning_m": 2.5, "critical_m": 3.5,
    "latest": {
      "water_level": {"value": 1.08, "unit": "m", "observed_at": "2026-11-01T09:40:00+03:00"},
      "rainfall": {"value": null, "unit": "mm/h", "observed_at": "2026-11-01T09:40:00+03:00"}
    }
  }
]
```

| field | rules |
|---|---|
| `status` | `active` or `inactive` |
| `health` | `ok`, `degraded`, `offline` or `unknown` |
| `type` | A single `type` string is accepted in place of `types` |
| `warning_m`, `critical_m` | Water-level thresholds in metres. The dashboard displays them and does not set them. |

**`GET /stations/health`**: the same shape. It may contain only `id`,
`health`, `status` and `last_communication`.

**`GET /stations/{id}/readings?from=&to=&type=`**: one station's readings.

- `from` and `to` are ISO times.
- `type` is `rainfall` or `water_level`.

```json
[
  {"station_id": "WL-017", "kind": "water_level", "observed_at": "2026-11-01T08:00:00+03:00",
   "value": 1.04, "unit": "m", "quality": "valid"},
  {"station_id": "WL-017", "kind": "water_level", "observed_at": "2026-11-01T09:00:00+03:00",
   "value": null, "unit": "m", "quality": "missing"}
]
```

`quality` is `valid`, `suspect` or `missing`.

**Freshness.** A station reading is shown as live within 30 minutes, delayed up
to 3 hours, and stale after that.

## Automated Alert System

**`GET /alerts?status=active`**: the active alerts.

**`GET /alerts/{id}`**: the full record of one alert.

```json
[
  {
    "id": "A-2026-0412",
    "title": "Flood warning: Mathare River corridor",
    "severity": "high",
    "status": "published",
    "areas": ["Mathare", "Mabatini", "Kiamaiko"],
    "created_at": "2026-11-01T12:10:00+03:00",
    "published_at": "2026-11-01T12:30:00+03:00",
    "description": "River expected to overtop at Juja Road bridge.",
    "recommended_actions": ["Move valuables above floor level", "Avoid crossing the river"],
    "acknowledged": 14,
    "recipients": 120
  }
]
```

| field | rules |
|---|---|
| `severity` | `low`, `moderate`, `high` or `critical` |
| `areas` | Place, ward or sub-county names. The dashboard maps them to wards for the alert-areas layer. |

**Until this system is connected.** The dashboard shows the digital twin's own
model warnings in its place. They are clearly labelled as not published, and
have no publication time or acknowledgements.

## Infrastructure Risk Mapping System

**`GET /assets?bbox=&category=`**: assets in the grid.

- `bbox` is `west,south,east,north`.

```json
[
  {"id": "H-0091", "name": "Mathare North Health Centre", "category": "health_centre",
   "lat": -1.2551, "lon": 36.8602, "risk_class": "high"}
]
```

| field | rules |
|---|---|
| `category` | `school`, `hospital`, `health_centre`, `bridge`, `public_building` or `emergency_service`. Other values are shown as given. |
| `risk_class` | Your system's own classification. Shown beside the dashboard's predicted risk. |

**Until this system is connected.** The dashboard uses 3,290 facilities from
OpenStreetMap, labelled as an interim source.

## Supply Chain Management System

Operational figures only. Do not send procurement or financial data
(§9.5).

**`GET /warehouses`**

```json
[{"id": "WH-1", "name": "Industrial Area depot", "lat": -1.306, "lon": 36.85,
  "stock": {"tarpaulins": 400, "water_kits": 900, "food_packs": 1200}}]
```

**`GET /distributions?status=`**

```json
[{"id": "D-77", "destination": "Mathare", "status": "in_transit",
  "items": {"food_packs": 200}, "updated_at": "2026-11-01T10:00:00+03:00"}]
```

Only emergency management officers, response teams and administrators see these
figures.

## Student Beneficiary Management System

Aggregates only (§9.6). The dashboard keeps just `area`, `level` and
`beneficiaries` from each row. Any other field is dropped before it is used, so
personal records are never stored or shown, even if they are sent by mistake.

**`GET /beneficiaries/aggregate?area_level=sub_county`**

```json
[{"area": "Mathare", "level": "sub_county", "beneficiaries": 420}]
```

Only emergency management officers and administrators see these figures.

## Reading from the dashboard

Partner systems can also read from the dashboard through its REST API:

- **Documentation:** OpenAPI 3 at `/api/v1/openapi.json`, browsable at
  `/api/docs`.
- **Access:** machine clients use a bearer token issued by the dashboard
  administrator. Tokens are set in `TWIN_API_TOKENS` as `token:role` pairs, and
  each token acts with its role's permissions.
- **Responses:** every response lists its sources with state and freshness, and
  marks predictions as model output.

## Testing before your API is ready

With `INTEGRATION_DEMO=1` and no URL set, each adapter serves fixed demo
records. They are named "DEMO …" and every response carries `"demo": true`.
Use demo mode to check the screens; never use it to make decisions. Once your
URL is set, demo records for your system stop.

## Checking a connection

The **System status** screen (`/ops/status`) shows the following for each
source:

- connected, down, or not connected;
- the last success and last failure, with the error;
- failed requests in the last 24 hours;
- response time;
- the age of the newest data.

The same information is at `GET /api/v1/system/integration-status`.
