# Test Report

Automated testing of the Nairobi Flood Digital Twin. Written to supply the
evidence Chapter 5 requires (test strategy, test cases with expected and actual
results, defects and retesting).

**Last run:** 96 tests, 96 passed, 0 failed
**Command:** `python -m pytest tests/ -q`
**Tooling:** pytest 9.1.1, Python 3.13.9, Windows 11, 8 CPU threads, CPU-only PyTorch

---

## 1. Test strategy and acceptance criteria

**Purpose.** Show that the implemented system behaves as specified, that
predictions stay within valid ranges, and that the system degrades safely when
external services or storage are unavailable.

**Levels used**

| Level | What it checks here |
|---|---|
| Unit | Individual functions: bias correction, rainfall trend, place risk levels, search parsing, road-graph snapping |
| Integration | Components working together: an outlook built from rainfall through the model, a route planned over a predicted flood field, a map document assembled from a probability grid |
| System | The running application: the page contains every control the callbacks drive, and the health endpoint answers |

**Techniques.** Mostly black-box (inputs and expected outputs), with grey-box
checks where internal behaviour matters (caching identity, calibration
monotonicity, dilation of the flood field before routing). Positive, negative,
boundary and exception cases are all included: 0 mm rainfall, a point 100 km from
any road, a one-character search, an unreachable forecast service and a
read-only disk.

**Environment and data.** Tests run offline. Rainfall is injected as synthetic
hourly series through `monkeypatch`, so results do not depend on the weather or
on a third-party service being up. Database tests use temporary files and never
touch the real one. Tests requiring the trained model or road network skip with
a clear reason if those files are absent.

**Acceptance criteria.** A test passes only if the observed result matches a
predefined expected result stated in the test. Ranges (for example, a 60 mm
storm flooding between 8 and 20 km²) come from measurements reported in
`RESULTS.md`, not from whatever the code happens to produce.

---

## 2. Test cases and results

All cases below were executed on the last run and passed. Test IDs match
`CHAPTER_5_GUIDE.md`.

| Test ID | Requirement area | Scenario and conditions | Expected result | Actual result | Verdict |
|---|---|---|---|---|---|
| TC-01 | Flood prediction | Predict for 60 mm over 3 days | 198 × 252 grid, all values in [0, 1]; flooded area 8–20 km² | Grid correct; 13.5 km² flooded | Pass |
| TC-02 | Flood prediction | Predict for 0 mm | No cell above the 0.5 threshold | 0 km² flooded | Pass |
| TC-02b | Flood prediction | Predict for 30, 60 and 100 mm | Flooded area increases with rainfall | 1.9 → 13.5 → 25.5 km² | Pass |
| TC-02c | Flood prediction | Check flooding against terrain susceptibility | Concentration ratio above 1.5 | 12.7 | Pass |
| TC-02d | Flood prediction | Inspect the calibration mapping | Monotone, within [0, 1], applied by the predictor | Monotone; model version reports calibration | Pass |
| TC-03 | Outlook | Rainfall rising through the 12-hour window | A warning naming places and the hour flooding starts | Warning issued with places and hour | Pass |
| TC-03b | Outlook | Rainfall of 0.02 mm per hour | No warnings; headline says no flooding expected | As expected | Pass |
| TC-04 | Routing | Eastleigh → Kibera during a 60 mm storm | Flood-aware route avoids all critical flooding; fastest route does not; both under 2 s | 0 km critical on the recommended route; fastest crosses it; 0.1 s | Pass |
| TC-04b | Routing | Same trip with no flooding | Both routes identical, 5–30 km | Identical, 10.7 km | Pass |
| TC-05 | Routing | Snap a point 100 km outside Nairobi | Distance above 600 m, so the request can be refused | Reported far beyond the limit | Pass |
| TC-06 | Search | Query "kencom" offline | Kencom returned in under 0.5 s | Returned in 0.003 s | Pass |
| TC-07 | Search | Coordinates outside the mapped area | Listed but disabled, labelled "outside the mapped area" | As expected | Pass |
| TC-07b | Search | Query "-1.2921, 36.8219" | Decodes to those coordinates | Exact match | Pass |
| TC-08 | Performance | Build the map document for a 60 mm storm | Under 10 s including polygon extraction | 2.2 s | Pass |
| TC-09 | Availability | Forecast service unreachable, cache present | Cached forecast returned, labelled offline | Notice begins "Offline" | Pass |
| TC-09b | Availability | Cache directory not writable | Warning logged, no exception | No exception raised | Pass |
| TC-10 | Deployment | Request `/healthz` | HTTP 200 with status and model version | 200, status ok | Pass |
| TC-10b | Interface | Build the page layout | Every control the callbacks drive is present | All 12 ids present | Pass |
| TC-11 | Persistence | Save a scenario, then list history | One row with the correct rainfall and critical count | As expected | Pass |
| TC-11b | Persistence | Create the database schema | At least 19 tables, including the ERD's core entities | 19 tables created | Pass |
| TC-12 | External data | Correct a 58 mm forecast total | Corrected above 100 mm (CHIRPS scale) | 124 mm | Pass |
| TC-12b | External data | Correct a rising series of totals | Monotone; 0 mm stays near 0 | As expected | Pass |
| TC-13 | NFR-05 / Credentials | Two users with the same password | Stored as salted hashes; no plaintext; hashes differ | scrypt hashes, all different | Pass |
| TC-14 | NFR-04 / Access control | Every restricted function under citizen, planner, responder, admin and signed out | Only the roles in the permission matrix succeed | Matrix matched for all 9 permissions | Pass |
| TC-15 | FR-12 / Sign-in | Correct password; wrong password; unknown email; passwordless account; no CSRF token; 5 failures | Session only for the correct password with a valid token; one identical message for every failure; lock after 5 failures | As expected; failures audited | Pass |
| TC-16 | FR-12 / Registration | Self-registration form posting role=admin | Account created as citizen | Citizen | Pass |
| TC-17 | FR-12 / Google and email-link sign-in | First sign-in; account pre-created by an admin; password account using Google; unverified email; other Firebase method; forged token | Citizen; the admin-assigned role; same account; refused; refused; refused | As expected for both methods (verifier replaced, no network) | Pass |
| TC-17b | FR-12 / Admin two-step sign-in | Admin signs in by password, Google or email link | No session until a valid authenticator code; enrolment by QR then recovery codes; reused code, wrong codes (locks after 5), expired attempt refused; recovery code works once | As expected; codes match the RFC 6238 test vectors | Pass |
| TC-17c | FR-12 / Privilege change | Planner promoted to admin while signed in | Citizen powers until the code step | As expected | Pass |
| TC-18 | NFR-04 / Server enforcement | What-if request from signed-out user and citizen | Held to live mode on the server | LIVE for both; WHATIF for planner | Pass |
| TC-19 | Administration | Demote or disable the last admin | Refused | Refused until a second admin exists | Pass |
| TC-20 | FR-11 / Report | Report for a scenario with a place named `<script>` | Parameters, extent, settlements, timestamp; name escaped | As expected | Pass |
| TC-21 | FR-03, FR-15 | Save alert areas; write audit entries | Areas saved per user; audit append-only, newest first | As expected | Pass |

Coverage by module:

| Test module | Tests | Covers |
|---|---|---|
| `test_model.py` | 9 | U-Net inference, calibration, caching, latency, units |
| `test_forecast.py` | 10 | Bias correction, rain trend, place levels, outlook, offline fallback |
| `test_routing.py` | 9 | Road graph, flood-aware routing, snapping, exposure over time |
| `test_geocoder.py` | 8 | Offline search, coordinates, out-of-area handling, speed |
| `test_dashboard.py` | 6 | Layout, health endpoint, map layers, themes, place summary |
| `test_persistence.py` | 5 | Scenario history, schema, model registration, simulation storage |
| `test_pipeline.py` | 3 | Processed arrays and memory guard (existing tests, retained) |
| `test_auth.py` | 46 | Password hashing, access-control matrix, password / Google / email-link sign-in, administrator two-step sign-in, lockouts, admin safeguards, database upgrade, audit log, subscriptions, report |

---

## 3. Defect log

Defects found during implementation and testing, with the corrective change and
the retest that confirmed it. All are recoverable from the repository history.

| ID | Defect | Cause | Fix | Retest |
|---|---|---|---|---|
| D-01 | Dashboard returned HTTP 500 after the model was replaced | The region summary emitted `level`/`Severe` while the renderer expected `risk_level` | `f9268cc` aligned the contract | Dashboard loads; covered by `test_flood_polygons_are_summarised_by_place` |
| D-02 | A whole flood ribbon was labelled with one place name, so Globe Roundabout showed as "Kariobangi South" | One label per polygon, though contours run for kilometres | `e056aec` split each polygon against a place partition | Hover labels verified; place summary test covers the contract |
| D-03 | The storm-progression slider showed less flooding at hour 24 than at hour 18 | The slider scaled rainfall by a bell curve that fell below the flooding threshold at both ends, which the daily model cannot represent | `6544f5f` removed the slider and the fabricated hydrograph | Replaced by the forecast-driven hour slider, covered by the outlook tests |
| D-04 | Storm replays showed "Forecast unavailable" | The archive request exceeded a 20 s timeout | `a8fe635` raised the timeout and committed the replay rainfall for offline use | All three replays load; offline path covered by `test_last_forecast_is_reused_when_the_service_is_down` |
| D-05 | Route lines disappeared when the map was zoomed out | deck.gl 9.2 drops paths with pixel width units below zoom 12 | Widths changed to metres with pixel limits (`314638a`) | Verified at zoom 10.5 in the browser |
| D-06 | Container crashed on start | `rasterio` links against a system library absent from the slim image (`libexpat.so.1`) | `libexpat1` installed in the image | Container reports healthy; `/healthz` covered by `test_health_endpoint_reports_status` |
| D-07 | Outlook said "little or no rain is forecast" beside a warning of flooding in 4 hours | The rain trend looked only 3 hours ahead | `9c58034` extended the trend across the outlook window | `test_trend_reports_rain_arriving_later_in_the_window` |
| D-08 | Saving a scenario would crash on a container with a root-owned volume | `save_scenario_run` caught database errors but not the permission error raised when creating the directory | `da840bc` treats storage failures as a warning | `test_saving_never_raises_when_storage_is_unavailable` |

D-08 was found **by** this test suite, not in use.

---

## 4. Running the tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q                      # whole suite, ~35 s
python -m pytest tests/test_routing.py -v       # one module
python -m pytest tests/ --junit-xml=results.xml # machine-readable results
```

Tests that need the trained model or the road network skip with an explanation
if those files are absent, so a fresh clone still runs the rest.

---

## 5. What the tests do not cover

Stated so the evaluation is not read as stronger than it is:

- **No user testing.** Usability and acceptance were not evaluated, so no claim
  is made about how easy the interface is to use.
- **No automated browser tests.** Page structure and callbacks are tested in
  Python. The sign-in, admin, planner, responder and citizen flows were driven
  once in headless Edge (create a user, hold and compare, export, report, alert
  areas), but that script is not part of the suite.
- **Google and email-link sign-in against a real Firebase project** are not
  covered: the token verifier is replaced in the tests, so the popup, the emailed
  link and Google's key check are exercised only when Firebase is configured.
- **No load or concurrency testing.** The application was tested with one user
  at a time.
- **Model accuracy is not retested here.** It is measured separately by
  `src/models/calibrate_v2.py` and `src/validation/`, and reported in
  `RESULTS.md`.
