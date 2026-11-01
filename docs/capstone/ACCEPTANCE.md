# Acceptance Checklist: Flood Visualization Dashboard

Evidence for each acceptance criterion in §11 of the *Flood Visualization
Dashboard Specification v1.0*. "Screen" is where a reviewer can see the
criterion met. "Test" is the automated check in `tests/`.

**Status key**

| mark | meaning |
|---|---|
| ✅ | met |
| ✅ 🔌 | met, with that system's data arriving once the partner system publishes its API (until then the screen says the system is not connected, or shows labelled demo data with `INTEGRATION_DEMO=1`) |

**How to review.** Start the app with `INTEGRATION_DEMO=1`, sign in with a
staff role, and open **Operations** (`/ops`).

| # | criterion | status | screen | test |
|---|---|---|---|---|
| 1 | Authorized users can access the dashboard | ✅ | Google or email-link sign-in, 2FA for admins; `/ops` for staff roles | `test_auth.py`, `test_ops_console.py::test_the_console_is_closed_to_the_public` |
| 2 | Main dashboard displays key indicators | ✅ | `/ops`: stations, rainfall, water levels, flood chance, areas under alert, high-risk areas, incidents, exposed infrastructure, each with source and freshness | `test_ops_console.py::test_overview_*` |
| 3 | Stations displayed at their locations | ✅ 🔌 | Live map, *Monitoring stations* layer; `/ops/sensors` | `test_integrations.py::test_a_connected_partner_is_normalised` |
| 4 | Station details and readings can be inspected | ✅ 🔌 | `/ops/sensors` → *Readings*: period, thresholds, gaps, CSV export | `test_ops_monitoring.py` |
| 5 | Rainfall shown with charts and geographical filters | ✅ | `/ops/rainfall`: hourly rainfall and 72 h accumulation, intensity classes, seasonal comparison 2015–2026, unusually wet days; area filter | `test_ops_monitoring.py::test_unusual_days_*` |
| 6 | Water levels shown with thresholds | ✅ 🔌 | `/ops/water-levels`: warning and critical lines, status, 3 h trend, stale readings marked | `test_ops_monitoring.py::test_water_level_trend_*`, `::test_an_old_reading_is_shown_as_stale_*` |
| 7 | Sensor connectivity and health visible | ✅ 🔌 | `/ops/sensors`: health, last communication with freshness | `test_ops_monitoring.py`, `test_ops_console.py` |
| 8 | Predictions visualized on the map | ✅ | Live map flood layer; `/ops/predictions` | `test_forecast.py`, `test_model.py` |
| 9 | Prediction timestamps and horizons displayed | ✅ | `/ops/predictions`: issue time, 12-hour hourly outlook, 3-day flood chance; run history | `test_ops_predictions.py` |
| 10 | Predicted zones distinguishable from observations | ✅ | "Model output, not observed flooding" on every prediction; observations labelled observed; API marks `"kind": "model_output"` | `test_ops_console.py::test_overview_shows_missing_partners_and_model_output_honestly` |
| 11 | Active alerts with severity and status | ✅ 🔌 | `/ops/alerts`: severity, status, areas, times, actions, acknowledgements, full record; model warnings labelled until the alert system connects | `test_ops_predictions.py::test_model_warnings_are_never_presented_as_published_alerts` |
| 12 | Filter by geographical area | ✅ | Sub-county and ward filter on every console page, area box on the map; county officers limited to assigned areas | `test_ops_console.py::test_area_and_risk_filters_*`, `::test_a_county_officer_never_sees_*` |
| 13 | Map layers can be enabled and disabled | ✅ | Live map *Map layers*: boundaries, rivers, infrastructure, stations, alert areas, plus satellite and past floods; saved per user; catalogue on `/ops/status` | `test_ops_console.py` |
| 14 | Infrastructure against flood-risk layers | ✅ | Map infrastructure layer ringed by predicted risk; `/ops/infrastructure` exposed facilities (OpenStreetMap until the Infrastructure Risk Mapping System connects) | `test_ops_assets.py::test_exposed_facilities_*` |
| 15 | Historical incidents can be viewed | ✅ | `/ops/incidents` (2000–2025 news reports, community reports); `/ops/history` | `test_ops_assets.py::test_incidents_*` |
| 16 | Predictions compared with observations | ✅ | `/ops/compare`: any run or storm replay against observations by ward (April 2024: 17 of 31 predicted wards confirmed) | `test_ops_predictions.py::test_a_stored_run_is_compared_*` |
| 17 | Unavailable or delayed sources indicated | ✅ | Freshness badge on every value; warnings above each page; `/ops/status` | `test_integrations.py`, `test_ops_monitoring.py::test_the_newest_observation_*` |
| 18 | Authorized reports can be generated | ✅ | `/ops/reports`: five reports, preview, PDF and CSV, recorded and audited; `/api/v1/reports/*` | `test_api.py::test_reports_in_three_formats` |
| 19 | Role-based access enforced | ✅ | Ten roles, permission checked on the server in every callback and API endpoint | `test_ops_console.py::test_permission_matrix`, `test_api.py::test_permissions_follow_the_role` |
| 20 | Retrieves data from the agreed components | ✅ 🔌 | Adapter per component (`src/integrations`); contract in `INTEGRATION_GUIDE.md` | `test_integrations.py` |
| 21 | Desktop and mobile screen sizes | ✅ | Responsive layout, cards stack under 1,100 px, sidebar folds under 860 px | checked in a headless browser at 1,500 px and 390 px (phone) |
| 22 | Usable when one integration is unavailable | ✅ | Every page states what is missing and keeps working | `test_resilience.py` (each source down in turn, and all at once) |

## Beyond the criteria

| item | where |
|---|---|
| REST API with all §7 endpoints, OpenAPI 3, Swagger UI | `/api/v1`, `/api/docs`; `test_api.py::test_openapi_lists_every_specification_endpoint` |
| §8 dashboard entities (source status, prediction runs, layers, preferences, report definitions, generated reports, boundaries) | `src/integrations/store.py` |
| Data integrity rules (§8.2): source timestamps, live/delayed/stale/historical, missing never zero | `src/integrations/base.py` |
| Audit log of administrative actions, exports and record views | `/admin/activity` |
| Sensitive data: beneficiaries aggregated and restricted, supplies restricted, reporters never shown | `test_ops_assets.py::test_supplies_and_beneficiaries_are_restricted`, `test_integrations.py::test_beneficiary_data_is_reduced_to_aggregates` |

## Remaining dependencies

| depends on | what it adds |
|---|---|
| IoT Sensor Data Platform API | real stations and readings (criteria 3, 4, 6, 7) |
| Automated Alert System API | published alerts and acknowledgements (11) and published-alert history |
| Infrastructure Risk Mapping System API | the partner's asset register and risk classes, replacing OpenStreetMap (14) |
| Supply Chain and Beneficiary system APIs | supplies and beneficiary panels |
| TAHMO gauge data (requested) | in-city rainfall through the IoT adapter interface |
