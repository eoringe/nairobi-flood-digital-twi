# Limitations

Honest account of what this model does and does not establish. Every figure
here was measured from the pipeline, not estimated. Written to be defensible
under questioning rather than flattering.

---

## 1. Flood extent is susceptibility-derived, not observed

**This is the most important limitation and the one to lead with.**

The training labels are constructed, not measured:

```
flood(t)     = storm(t) AND susceptible(pixel)
storm(t)     = accumulated rainfall over [t, t+2] >= 30 mm
susceptible  = terrain  : f(HAND, slope, TWI)          -- fails validation (§9)
               drainage : built_up x channel_proximity x flat   -- passes (§9)
               permanent water excluded in both
```

No pixel in the training set was ever confirmed flooded by observation. The
model therefore does not learn *where Nairobi floods* from ground truth. It
learns to reproduce a susceptibility field, gated by a rainfall threshold.

Two susceptibility formulations exist and both are retained. The terrain one
fails spatial validation; the drainage one passes. Any reported result must
state which was used — they produce different labels and therefore different
models.

**What can honestly be claimed:** the model anticipates storm-driven flood
timing from antecedent rainfall, and distributes the resulting water according
to a susceptibility field that, in the drainage formulation, agrees with
independently mapped flood-prone areas at neighbourhood scale.

**What cannot be claimed:** that predicted flood extents correspond to real
inundation. That requires independent validation (§9).

### 1a. The circularity risk, and what mitigates it

Because the label is a function of rainfall and terrain, and rainfall and
terrain are also the model inputs, a naive setup lets the network recover the
label by arithmetic rather than learning anything — it would score a high F1
while demonstrating nothing.

The forecast framing is what prevents that. The input window and the label
window do not overlap:

| | days |
|---|---|
| input (rainfall shown to model) | *t−7 … t−1* |
| label (flood extent) | *t … t+2* |

Day *t*'s rainfall is never provided, so the storm trigger cannot be computed
from an input channel and must be *anticipated* from antecedent conditions.

This mitigates but does not eliminate the concern. The spatial component
remains partly circular: the susceptibility field is derived from HAND and
slope, and HAND and slope are input channels, so the network can in principle
learn a thresholding function over an input rather than a genuine spatial
relationship. **This should be conceded directly if raised, not defended.**

---

## 2. Rainfall has no spatial variation

`rainfall_chirps.npy` has shape `(4138, 1, 1)` — CHIRPS was fetched as a single
point over Nairobi, not as a grid. Each sample's rainfall input is 7 scalars
broadcast uniformly across all 49,896 pixels.

Consequences:

- Spatially localised convective storms — the dominant flood-producing
  mechanism in Nairobi — cannot be represented. A cell over Kibera and one over
  Karen are identical to the model.
- All spatial structure in the prediction necessarily originates from terrain.
- The project roadmap called for regridding CHIRPS to 0.05°. That was never
  done, and remains the single highest-value data improvement available.

---

## 3. The model uses no SAR data

Despite the project's framing, `dataset_metadata.json` declares the input
channels as `["rain", "dem", "slope", "twi"]`. There is no Sentinel-1
backscatter in the model inputs. The comment in the earlier
`build_segmentation_dataset.py` claiming `# SAR: first 4 channels (VV, VH,
angle)` was incorrect.

Sentinel-1 was used earlier in the project and abandoned for documented
reasons (§4). The thesis should describe SAR as an *investigated and rejected*
data source, not as a model input.

---

## 4. Two satellite data sources were tested and rejected

Both rejections are empirical findings worth reporting as results, not
failures to hide.

**Sentinel-2 optical (MNDWI).** 156 cloud-free scenes yielded zero detectable
open water, including during a 138 mm rainfall event. Urban street flooding is
too small-scale and too transient for 10 m optical imagery, and cloud cover
peaks exactly when flooding occurs.

**Sentinel-1 SAR, absolute threshold.** The 23 `s1_water_mask_*.tif` composites
were built with `VV < −16 dB`. That threshold anti-correlates with rainfall
(ρ = −0.74, p < 0.001) — it selects smooth dry surfaces rather than water. Wet
soil raises VV backscatter, masking the water signal in an urban setting.
These files are deliberately excluded from the current pipeline.

**Sentinel-1 change detection.** Backscatter drop against a dry-season baseline
also correlated negatively with rainfall (r = −0.399), for the same physical
reason.

---

## 5. Grid resolution exceeds the scale of urban flooding

The 198 × 252 grid spans 0.12° latitude × 0.18° longitude, giving roughly
**67 m × 79 m per pixel**. Urban flooding in Nairobi occurs at street and
drainage-channel scale, typically 10–30 m wide. Most real flood features are
therefore sub-pixel, and the model predicts *flood-affected areas* rather than
resolved flood extents.

---

## 6. Few independent storm events

The dataset covers **22 storm seasons** (2015–2026, long and short rains).
Although it contains 2,024 samples, consecutive samples share overlapping
7-day rainfall windows and identical terrain, so the effective independent
sample size is closer to the number of seasons than the number of rows.

The train/val/test split is event-aware — no storm season appears in more than
one set — which is the correct handling, but it means:

| split | seasons | samples | storm-positive |
|---|---|---|---|
| train | 15 | 1,380 | 203 (14.7%) |
| val | 3 | 276 | 42 (15.2%) |
| test | 4 | 368 | 37 (10.1%) |

**37 storm-positive test samples drawn from 4 seasons** is a thin basis for the
headline metric. Per season, Model B scores 0.923–0.945 and Model A 0.114–0.244
(RESULTS.md §4.11). Model B is stable; Model A's score depends heavily on the
split. Test scores will be sensitive to which seasons landed in the
test split. K-fold cross-validation across all 22 seasons is the appropriate
remedy and is computationally cheap at this dataset size.

---

## 7. Class imbalance

Flood pixels are ~0.90% of the dataset. Accuracy is meaningless here — a model
predicting "no flood" everywhere scores 99.1%. Only F1, IoU, precision and
recall are reported, and Focal Tversky loss (α = 0.7 weighting false negatives,
β = 0.3 weighting false positives) is used so that missed floods are penalised
more heavily than false alarms.

This asymmetry is deliberate and appropriate for an early-warning application,
where a missed flood costs more than a false alarm — but it does mean reported
precision will be systematically lower than recall.

---

## 8. Label parameters are chosen, not fitted

Five parameters define the labels and were set from literature and judgement
rather than calibrated against observations:

| parameter | value | basis |
|---|---|---|
| rainfall threshold | 30 mm / 3 days | documented heavy-rain threshold for Nairobi |
| saturation rainfall | 120 mm / 3 days | extent saturates at extreme totals |
| extent range | 2% → 18% of grid | plausible urban flood footprint |
| HAND decay scale | 3 m | HAND as dominant flood control |
| slope decay scale | 0.15 | flat ground pools |

Different values yield different labels and therefore different scores. A
sensitivity sweep over these parameters is required before any reported metric
can be treated as stable, and is inexpensive — a full dataset rebuild takes
seconds.

---

## 9. Independent validation: partial, temporal only

The labels have been checked against independently documented Nairobi flood
events (`src/validation/validate_documented_events.py`). Sources are Copernicus
EMS, ReliefWeb situation reports and contemporaneous news.

Events are graded by how directly they evidence flooding *in Nairobi*. Sources
reporting dated Nairobi flooding are stronger evidence than Kenya-wide episodes
in which Nairobi is merely named among affected areas — heavy rain in western
Kenya says little about this catchment.

**Nairobi-specific events**

| documented event | 3-day rainfall | labels flag it? |
|---|---|---|
| March 2026 — Nairobi River burst its banks, 37 deaths in Nairobi | 41.5 mm | yes, 2/10 days |
| April 2024 — Mathare, ~147,000 affected in Nairobi County | 62.3 mm | yes, 8/8 days |
| November 2023 — El Niño, rivers burst banks | 39.6 mm | yes, 3/15 days |

**Kenya-wide episodes including Nairobi**

| documented event | 3-day rainfall | labels flag it? |
|---|---|---|
| Oct–Dec 2019 — wettest short rains on record, ~400% of average | 77.7 mm | yes, 16/40 days |
| Apr–May 2020 — ~194 deaths, 100,000 displaced nationally | 92.3 mm | yes, 11/26 days |
| Mar–May 2018 — long rains ~145% of average, 310,000 displaced | 81.8 mm | yes, 15/40 days |

**Detected 6/6 (3/3 Nairobi-specific). Dry-season controls produced false alarms
on 3 of 126 days (2.4%).**

This supports the rainfall threshold as a flood indicator: every documented
flood coincides with days the labels flag, and dry periods stay quiet 97.6% of
the time.

**What remains unvalidated — and a negative result.** The above checks the
temporal trigger only. A separate test of the *spatial* claim
(`src/validation/validate_spatial_neighbourhoods.py`) compared the susceptibility
field against 37 flood-prone neighbourhoods mapped under the Nairobi Rivers
Regeneration Programme, plus neighbourhoods reported flooded in April 2024.

**It found no agreement.** Mapped flood-prone areas do not score higher than
control neighbourhoods at any sampling radius (single pixel −14.0 points,
*p* = 0.818; disc up to ~525 m, *p* ≥ 0.589), and the April 2024 predicted mask
covered 1 of 10 reported neighbourhoods against a chance expectation of 0.8.
Mathare — worst affected, over 7,000 displaced — sits at the 47th percentile of
susceptibility and is not predicted flooded.

Raster misalignment was ruled out: HAND correlates positively with elevation as
stored (r = +0.275) and worse under every flip.

The likely explanation is that Nairobi's flooding is driven substantially by
drainage failure — blocked storm drains, riparian encroachment, impervious
surfaces — which a terrain model cannot represent, compounded by ~70 m pixels
against river valleys 100–200 m wide.

**This was subsequently addressed.** Rebuilding susceptibility around drainage
(built-up land x channel proximity x flat ground, channels from the previously
unused `predictor_upa.npy` flow accumulation) passes the same benchmark:
separation +24.3 to +26.1 points, significant at all five sampling radii
(p = 0.027-0.037), and Mathare moves from the 47th to the 97th percentile. Both
datasets are retained so the comparison can be reported. See RESULTS.md 4.8.3,
including the multiple-comparisons caveat: eleven predictors were compared and
none clears a Bonferroni-corrected threshold.

The original negative result still stands for the terrain formulation, and the
following caveats apply to both verdicts. The test uses six controls,
approximate centroids, and a news summary rather than the source GIS layer. But
it means **"flooding occurs in these pixels" has no supporting evidence, and one
deliberate attempt to find some failed.** Treat the spatial output as
unvalidated, and do not present flood maps as operationally reliable.

Six events is still a modest sample, and three of them are Kenya-wide rather
than Nairobi-specific. Sources are recorded per event in the script so a reader
can audit them.

### 9a. A real event that shows the forecasting limit

The November 2023 flood is a worked illustration of §10. In the seven days
before the peak, only **2.2 mm** of rain fell — the antecedent record looked
like a dry spell — and then **39.6 mm** arrived. Model A sees only that
antecedent window, so it could not have anticipated this flood, and no model
restricted to rainfall history could. Model B, given the rainfall, handles it.

This single event demonstrates the argument more convincingly than the aggregate
metrics: the constraint is meteorological information, not model capacity.

---

## 10. Predicted probabilities are overconfident in the uncertain middle

The model emits a per-cell value in [0, 1] and the interface presents it as a
probability. Whether that word is earned depends on calibration: a cell the model
calls 80% should flood about 80% of the time.

Measured on the held-out storm seasons:

| predicted band | observed frequency | predicted mean | gap |
|---|---|---|---|
| 0.0–0.1 | 0.0% | 0.0% | 0 |
| 0.1–0.2 | 24.8% | 14.5% | −10 |
| 0.3–0.4 | 37.1% | 34.8% | −2 |
| 0.6–0.7 | 50.5% | 65.2% | **+15** |
| 0.7–0.8 | 55.1% | 75.2% | **+20** |
| 0.8–0.9 | 61.8% | 85.5% | **+24** |
| 0.9–1.0 | 95.5% | 99.6% | +4 |

**99.92% of cells fall in the two extreme bands, where calibration is good.** The
middle bands hold 0.08% of cells but are overconfident by 15–24 percentage
points: a cell the model calls 85% floods about 62% of the time.

**Do not quote a cell-weighted expected calibration error for this model.** It
computes to 0.000, which is an artefact: 18.2 million of 18.4 million cells sit
in the trivial 0.0–0.1 background band where the model is exactly right, and
they swamp the average. Weighting each band equally gives 0.092, and restricting
to the mid-range bands gives 0.109. The unweighted figures are the informative
ones.

This matters where it is least convenient. The well-calibrated extremes are the
cells whose classification was never in doubt; the overconfident middle is
precisely the borderline zone where a warning decision is marginal. A displayed
"85% probability" on a mid-range polygon overstates the true frequency.

Overconfidence of this kind is common in neural networks trained with a
cross-entropy component and does not indicate a defect in training.

**Corrected.** Isotonic regression is fitted on the validation seasons and
applied by the dashboard. On the test seasons it brings unweighted ECE from
0.092 to **0.020** and mid-band ECE from 0.109 to **0.025**. A cell shown as 85%
now carries a flood label about 85% of the time, and F1 is unchanged (RESULTS.md
§4.10). The correction is against the constructed labels, so it removes
overconfidence relative to the labels, not relative to observed floods (§1).

**Terminology.** "Probability" and "likelihood" are used interchangeably in
casual speech but denote different things in statistics: probability is
P(outcome | model), likelihood is L(model | data). The quantity here is
P(cell floods | rainfall, terrain), a probability. The interface previously
mixed both words for the same number; it now uses "probability" throughout.

---

## 11. Performance is bounded by rainfall predictability, not by the model

The task decomposes into a temporal half (will a storm arrive?) and a spatial
half (given a storm, which pixels flood?). Measured on the held-out test
seasons, these behave very differently:

| method | test F1 | precision | recall |
|---|---|---|---|
| predict nothing | 0.0000 | — | 0.000 |
| flood everywhere, always | 0.0145 | 0.007 | 1.000 |
| fixed stencil, ignores rainfall | 0.1434 | 0.080 | 0.702 |
| logistic regression on rainfall | 0.1592 | 0.167 | 0.152 |
| U-Net, Model A (forecast) | 0.1696 | 0.107 | 0.407 |
| **oracle: storm known, true extent** | **0.9997** | 1.000 | 0.999 |

The oracle row is the key result. **Given the rainfall, flood extent is
essentially perfectly recoverable from terrain.** The entire gap between 0.17
and 1.00 is storm forecasting, not spatial modelling.

Storm-detection skill from antecedent rainfall was measured directly by
ablation. Scene-level AUC stays at **0.58–0.68** across every feature set and
classifier tried:

| features | AUC | scene F1 |
|---|---|---|
| 7 antecedent days | 0.661 | 0.205 |
| + day-of-year (seasonality) | 0.679 | 0.345 |
| + 14/30-day antecedent totals | 0.656 | 0.321 |

Seasonality is a genuine gain (scene F1 0.205 → 0.345) and is now included in
the inputs. But AUC — which, unlike F1, is independent of the base rate —
barely moves. Apparent gains from lengthening the forecast horizon come mostly
from raising the positive rate (13.9% → 29.1%), not from added skill. That
distinction matters: **F1 can be inflated by making the task easier without
improving prediction at all.**

**Consequence for the success criterion.** The project roadmap set "validation
F1 > 0.60". That is unreachable for Model A and no loss function, architecture,
or feature set can reach it, because the information is not present in
rainfall history. The criterion has been replaced with "beats terrain-only and
linear-rainfall baselines", which Model A meets. Model B exceeds 0.60 by
construction, but answers a different question and must be reported separately
(§1a).

**Practical implication.** An operational Nairobi flood warning system should
consume a numerical weather prediction rainfall forecast (KMD, ECMWF) rather
than attempt to forecast rainfall from rainfall history. The hydrological
mapping is solved; the meteorological input is the binding constraint.

---

## 12. Temporal coverage

The CHIRPS series spans 2015-01-01 to 2026-04-30 (4,138 days). Eleven years is
modest for characterising climate variability, and the period may not represent
the rainfall regime under continued climate change. Trends in storm intensity
cannot be reliably separated from interannual variability at this record
length.

---

## 13. Flood-aware routing is a lower-risk suggestion, not a safe route

The trip planner (`src/routing/flood_router.py`) drives the OpenStreetMap road
network (29,473 roads) with each road's cost raised by the flood probability
under it: ×3 in the moderate band, ×25 in high, closed in critical. It shows
the flood-aware route beside the fastest route that ignores flooding.

What that does not establish:

- **Street scale is not validated.** Flood extent is checked at neighbourhood
  scale on ~70 m cells (§5, §9). A bridge or raised carriageway inside a flooded
  cell is avoided anyway, and a low underpass in a dry cell is not. The
  interface says "flood-aware route", never "safe route".
- **No live traffic.** Travel times use typical speeds per road class (e.g.
  35 km/h on primary roads). Google Maps reroutes on live GPS from millions of
  phones, which shows what is actually happening. This system reroutes on
  *predicted* flooding. A crowd-sourced "report flooded road" feature would add
  real observations; it is future work.
- **The penalties are chosen, not fitted.** Like the label parameters (§8), the
  ×3 / ×25 / closed weights encode a judgement about acceptable detours. There
  is no data on how drivers trade time against flood exposure.
- **Routing only covers the mapped area.** Location search finds places across
  Nairobi, but anything outside the downloaded road network (e.g. Two Rivers
  Mall, JKIA) is shown as "outside the mapped area" and cannot be routed to.
  A start or destination more than 600 m from a mapped road is refused rather
  than snapped to a distant road.
- **Outside the grid there is no prediction.** Roads beyond the 198 × 252 grid
  carry no penalty, and the route summary says when a route leaves the grid.

---

## 14. "Flooding in about 2 hours" comes from the rainfall forecast, not the model

The model has no clock. It was trained on *daily* CHIRPS rainfall and maps a
3-day total to extent. The outlook (`src/forecast/nowcast.py`) produces timing
by feeding it hourly rainfall from Open-Meteo:

1. For each hour from now to +12 h, total the rain over the 72 hours ending
   then.
2. Run the model on each total.
3. Report the first hour each place reaches moderate, high or critical.

So "moderate flooding in Mathare in about 2 hours" means: *at the current
forecast, the 72-hour total reaches the level where the model shows moderate
flooding there in about 2 hours.* Consequences:

- **Timing is only as good as the hourly rainfall forecast.** Convective storms
  over Nairobi are poorly predicted at hourly resolution (§11).
- **Water movement is not simulated.** There is no runoff lag, drainage
  capacity or flow routing. Urban flash flooding in Nairobi follows intense rain
  closely, so the rainfall crossing time is a reasonable proxy, but it has not
  been checked against observed flood times.
- **A single rainfall point** still drives the whole city (§2).
- **Antecedent rainfall uses the seasonal mean.** The model is insensitive to it:
  at 50 mm, antecedent 0 vs 20 mm/day moves flooded area from 4.04% to 4.09%.
- **The replays are not forecasts.** The April 2024 replays run the outlook at a
  past moment on archived hourly reanalysis. The "future" hours are what
  actually fell, which is a perfect forecast no live system would have. The
  interface labels them as such.
- **No invented fallback.** If the forecast cannot be fetched, the outlook says so
  and shows no warnings.

---

## 15. People at risk is an exposure count, not an impact estimate

The dashboard sums WorldPop 2025 population over cells predicted flooded
(RESULTS.md §4.12). This replaced a county-average density that understated
exposure about 4×. What the figure is not:

- **Not a count of people flooded.** A ~70 m cell is predicted flooded; not
  every household in it is. The figure is closer to an upper bound.
- **Not a census count.** WorldPop 2025 is modelled from the 2019 census and
  building footprints, and informal settlements are hard to count.
- **Scale check.** At the April 2024 rainfall it gives about 370,000 people,
  against about 147,000 reported affected. That is consistent with an
  upper-bound exposure measure.

---

## Summary for the defence

The defensible claim is narrow and should be stated narrowly:

> A U-Net trained on antecedent rainfall and terrain can anticipate
> storm-driven flood extent on held-out storm seasons, where extent is defined
> by a terrain-susceptibility model. The approach avoids the inverted-threshold
> failure that affects absolute SAR water detection in this setting, and the
> non-overlapping forecast window prevents the network from recovering the
> label arithmetically. Validation against independently documented flood
> events remains outstanding and is required before any operational use.

Anticipated questions and honest answers:

**"Isn't the model just learning your HAND threshold?"** Partly, yes — for the
spatial component. The temporal component is a genuine forecast, since day *t*
rainfall is withheld. See §1a.

**"How do you know the predictions are right?"** We do not, in the sense of
comparison against observed floods. See §9.

**"Why is precision low?"** By design — Focal Tversky weights false negatives
above false positives, which suits early warning. See §7.

**"Why not use the SAR data?"** It was used, tested, and found to be inverted
for this site (ρ = −0.74 with rainfall). See §4.

**"Can a driver trust the flood-aware route?"** As a lower-risk suggestion,
yes; as a guarantee, no. Street-scale flooding is not validated and there is no
live traffic. See §13.

**"How can a daily model say 'in 2 hours'?"** It cannot by itself. The timing
comes from the hourly rainfall forecast crossing the model's flooding level.
See §14.
