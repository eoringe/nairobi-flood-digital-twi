# Flood Visualization Dashboard: Capstone Task List

How the Nairobi Flood Digital Twin will meet the *Flood Visualization Dashboard
Specification v1.0* (Integrated Flood Monitoring System). Every task lists the
requirement IDs it closes (FVD-FR-xxx, use case UC-FVD-xxx, acceptance criterion
AC-n from §11 of the specification).

**Status key**

| mark | meaning |
|---|---|
| ✅ | already in the app |
| 🟡 | partly there, task finishes it |
| ⬜ | to build |
| 🔌 | placeholder: the screen, API and data model are built, waiting for a partner system's API |

**Architecture decision.** The specification recommends React, FastAPI and
PostGIS "subject to the existing project architecture". The app is Dash on
Flask with SQLite, deployed in Docker, and it already meets much of the
specification. Rewriting it would cost the remaining time for no functional
gain. The plan therefore keeps Dash and Flask, adds a versioned REST API
(`/api/v1`, OpenAPI documented) on the same Flask server, and keeps the storage
layer behind small store classes so it can move to PostgreSQL/PostGIS without
touching screens.

**Partner systems not yet available.** IoT Sensor Data Platform, Automated
Alert System, Infrastructure Risk Mapping System, Supply Chain Management
System, Student Beneficiary Management System. Each gets an adapter that reads
its base URL and key from environment variables. Until then, the adapter
reports *Not connected*, every screen that depends on it shows an explicit
empty state naming the missing system, and nothing is invented. An optional
`INTEGRATION_DEMO=1` mode loads clearly labelled demo fixtures, watermarked
*DEMO DATA*, so screens can be shown and acceptance-tested before the real APIs
exist.

---

## What already exists

| spec area | what the app has | status |
|---|---|---|
| 2.2 Interactive map | 3D WebGL map, zoom/pan/tilt, feature hover and click details, legends, layer switches (prediction, satellite, past floods, community reports) | 🟡 |
| 2.6 Prediction visualization | U-Net flood extent, 12-hour outlook with hours, issue time, 3-day flood chance with 95% call, labelled as model output | 🟡 |
| 2.7 Risk visualization | Moderate / High / Critical bands per place | 🟡 |
| 2.8 Alerts | Outlook warnings, *My flood alerts* for subscribed areas | 🟡 |
| 2.10 Incidents | Community flood reports (status, confirm, clear, verify), 26 years of news-reported floods | 🟡 |
| 2.12 Search | Place and building search | 🟡 |
| 2.14 Access | Google/email sign-in, 2FA for admins, roles, server-side permission checks, audit log | 🟡 |
| 2.15 Refresh and status | Auto-refreshing outlook, offline fallback, admin health panel | 🟡 |
| 10.5 Usability | Responsive layout, light/dark theme, risk colours with text labels | ✅ |

---

## Phase 1: Foundation and integration layer

| # | task | closes | status |
|---|---|---|---|
| 1.1 | `src/integrations/`: a base adapter with timeout, retry with backoff, and status recording (last success, last failure, latency, error); a registry of all data sources | FR-023, FR-017, §2.15, §10.2 | ✅ |
| 1.2 | Internal adapters over existing components: forecast feed, prediction model, flood chance, community reports, news-reported floods, satellite flood map | FR-023 | ✅ |
| 1.3 | 🔌 Partner adapters: IoT sensors, alert system, infrastructure risk, supply chain, beneficiary. Environment-variable configuration, *Not connected* state, contract documented | FR-023, §9 | ✅ |
| 1.4 | Demo fixtures behind `INTEGRATION_DEMO=1`, watermarked in every view and API response (`"demo": true`) | AC-20 testing | ✅ |
| 1.5 | Data model: `data_source_status`, `prediction_runs`, `map_layers`, `user_preferences`, `report_definitions`, `generated_reports`, `geographic_boundaries` (SQLite, PostGIS-ready schema) | §8 | ✅ |
| 1.6 | Freshness rules: every value carries its source timestamp and is classed live / delayed / stale / unavailable; missing readings are never zero | §8.2, §10.6, FR-024 | ✅ |

## Phase 2: Roles, operations console and main dashboard

| # | task | closes | status |
|---|---|---|---|
| 2.1 | Spec roles (assigning areas in the UI and filtering by them: 3.7, 7.5): system administrator, monitoring officer, hydrologist, data analyst, emergency management officer, county/regional officer (with assigned areas), infrastructure officer, disaster response team, public. Existing roles mapped; permission matrix updated | §4, FR-019, AC-19 | ✅ |
| 2.2 | Operations console at `/ops` with role-filtered sidebar, sharing the admin console's shell | §6, §2.14 | ✅ |
| 2.3 | Main dashboard: summary cards (stations total/active/inactive, latest rainfall, latest water level, areas under alert, active predictions, high-risk areas, reported incidents, exposed infrastructure), recent activity, freshness badge per card | §2.1, §6.1, FR-001, UC-001, AC-2 | ✅ |
| 2.4 | Global refresh bar: last successful update, manual refresh, configurable auto-refresh interval saved per user | FR-016, FR-017, §2.15 | ✅ |
| 2.5 | System notifications strip: data warnings (source down, stale, demo mode) | FR-020 | ✅ |

## Phase 3: Map completeness and geographical filtering

| # | task | closes | status |
|---|---|---|---|
| 3.1 | Administrative boundaries: Nairobi's sub-counties and wards (geoBoundaries, CC BY 4.0), drawn on the map and available as filters | §2.2, FR-013, UC-007, AC-12 | ✅ |
| 3.2 | River network layer (from flow accumulation already in the repo) and permanent water bodies | §2.2 | ✅ |
| 3.3 | Station markers layer, health-coloured, click for details | FR-003, UC-002, AC-3 | ✅ layer · 🔌 data |
| 3.4 | Active alert area layer | FR-009 | ✅ |
| 3.5 | Infrastructure layer: schools, hospitals and health centres, bridges, public buildings from OpenStreetMap as an interim source until the Infrastructure Risk Mapping System connects | §2.9, FR-010, AC-14 | ✅ |
| 3.6 | Layer manager: every layer listed with a description and source; visibility saved per user | FR-014, AC-13 | ✅ |
| 3.7 | Filters applied across map, cards and tables: sub-county, ward, risk level, alert severity, date; *Reset filters* | §2.12, FR-013 | ✅ |
| 3.8 | Risk classes aligned to the specification (the console uses Low/Moderate/High/Very High; the map legend keeps Critical, its Very High): Low, Moderate, High, Very High (Critical shown as Very High), with filtering by class | §2.7, FR-008 | ✅ |

## Phase 4: Sensors, rainfall and water levels

| # | task | closes | status |
|---|---|---|---|
| 4.1 | Sensor monitoring screen: station list, status, latest readings, last communication time, health, search by station | §2.3, §6.3, FR-003, FR-006, FR-015, AC-4, AC-7 | 🔌 |
| 4.2 | Station detail: time-series chart with date range, threshold lines, missing data shown as gaps | UC-002, FR-024 | 🔌 |
| 4.3 | Rainfall analytics: accumulated rainfall, intensity categories (KMD thresholds), trend charts with configurable periods, historical comparison. Works now on forecast-feed and CHIRPS rainfall; stations added when connected | §2.4, §6.4, FR-004, UC-003, AC-5 | 🟡 |
| 4.4 | Water-level analytics: river-level charts, warning/critical thresholds, rising/stable/falling trend, station comparison | §2.5, §6.5, FR-005, UC-004, AC-6 | 🔌 |
| 4.5 | TAHMO gauge ingestion once approved (feeds 4.1–4.3 through the IoT adapter interface) | §9.1 | ⬜ |

## Phase 5: Predictions, risk and alerts

| # | task | closes | status |
|---|---|---|---|
| 5.1 | Persist every prediction run (issue time, horizon, inputs, flood chance, place levels, model version) | §2.6, FR-007 | ⬜ |
| 5.2 | Prediction screen: risk by area, horizons, generation time, confidence and uncertainty, model details, run history | §6.6, UC-005, AC-8, AC-9 | 🟡 |
| 5.3 | Prediction vs observation: pick a run and area, show predicted level beside community reports, news reports, satellite and (when connected) gauges; hit/miss indicators | FR-021, UC-010, AC-16 | ⬜ |
| 5.4 | Observed vs predicted styling: predictions hatched and labelled *Model output*; observations solid | §2.6, AC-10 | 🟡 |
| 5.5 | Active alerts screen: list with severity, status, affected areas, created/published times, recommended actions, acknowledgement status, full record for authorised users, filters. Internal outlook warnings now; Automated Alert System when connected | §2.8, §6.7, FR-009, UC-006, AC-11 | 🟡 🔌 |

## Phase 6: Infrastructure, incidents, history

| # | task | closes | status |
|---|---|---|---|
| 6.1 | Infrastructure risk screen: facilities in or near predicted and observed flood areas, counts by type, details | §2.9, §6.8, UC-008, AC-14 | ⬜ |
| 6.2 | Vulnerable communities and beneficiary statistics, aggregated, restricted roles only | §2.9, §9.6 | 🔌 |
| 6.3 | Emergency supplies panel (warehouses, stock, distribution), restricted | §9.5 | 🔌 |
| 6.4 | Incident screen: community + news incidents with severity, type, status, date filters, distribution charts | §2.10, §6.9, FR-011, UC-009, AC-15 | 🟡 |
| 6.5 | Historical analysis: rainfall and water-level history, past incidents, historical alerts, period comparison, summary statistics | §2.11, §6.10, FR-012 | 🟡 |

## Phase 7: Reports, API and status

| # | task | closes | status |
|---|---|---|---|
| 7.1 | Reports screen: flood summary, station report, risk summary, alert and incident summary; preview; PDF and CSV export; generated reports recorded and audited | §2.13, §6.11, FR-018, UC-011, AC-18 | ⬜ |
| 7.2 | REST API `/api/v1`: all endpoints in §7, session or token auth, role checks, pagination, `source` / `timestamp` / `freshness` / `demo` on every record | §7, FR-023 | ⬜ |
| 7.3 | OpenAPI document and Swagger UI at `/api/docs` | §12 | ⬜ |
| 7.4 | System status screen: each source's state, last success, failed requests, freshness warnings | §6.12, FR-017, UC-012, AC-17 | ✅ |
| 7.5 | User management screen: roles, assigned areas, access status, audit history | §6.13, FR-022 | 🟡 |

## Phase 8: Testing, documentation, deployment

| # | task | closes | status |
|---|---|---|---|
| 8.1 | Tests: adapters (success, timeout, not configured), freshness rules, permissions per role, API contract, report generation | §14 Phase 6 | ⬜ |
| 8.2 | Resilience test: each partner source down at a time, dashboard still usable | §10.2, AC-22 | ⬜ |
| 8.3 | Acceptance checklist mapping AC-1 to AC-22 to evidence (screenshots, tests) | §11 | ⬜ |
| 8.4 | Integration guide for partner teams: the contract each adapter expects | §9 | ⬜ |
| 8.5 | Docker and environment variables updated | §12 | ⬜ |

---

## Integration contracts expected from partner teams

Each adapter is written against the endpoint shape below. If a partner's API
differs, only its adapter changes.

| system | env vars | endpoints the adapter calls |
|---|---|---|
| IoT Sensor Data Platform | `IOT_API_URL`, `IOT_API_KEY` | `GET /stations`, `GET /stations/{id}/readings?from&to&type`, `GET /stations/health` |
| Automated Alert System | `ALERTS_API_URL`, `ALERTS_API_KEY` | `GET /alerts?status=active`, `GET /alerts/{id}` |
| Infrastructure Risk Mapping | `INFRA_API_URL`, `INFRA_API_KEY` | `GET /assets?bbox&category`, `GET /assets/{id}/risk` |
| Supply Chain Management | `SUPPLY_API_URL`, `SUPPLY_API_KEY` | `GET /warehouses`, `GET /distributions?status` |
| Student Beneficiary Management | `BENEFICIARY_API_URL`, `BENEFICIARY_API_KEY` | `GET /beneficiaries/aggregate?area_level` (aggregates only) |
