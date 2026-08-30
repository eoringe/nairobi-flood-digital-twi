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
susceptible  = f(HAND, slope, TWI), permanent water excluded
```

No pixel in the training set was ever confirmed flooded by observation. The
model therefore does not learn *where Nairobi floods* from ground truth. It
learns to reproduce a terrain-based susceptibility field, gated by a rainfall
threshold.

**What can honestly be claimed:** the model anticipates storm-driven flood
timing from antecedent rainfall, and distributes the resulting water according
to terrain controls that are standard in the flood-mapping literature (HAND as
the dominant control, modulated by slope and topographic convergence).

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
headline metric. Test scores will be sensitive to which seasons landed in the
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

## 10. Performance is bounded by rainfall predictability, not by the model

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

## 11. Temporal coverage

The CHIRPS series spans 2015-01-01 to 2026-04-30 (4,138 days). Eleven years is
modest for characterising climate variability, and the period may not represent
the rainfall regime under continued climate change. Trends in storm intensity
cannot be reliably separated from interannual variability at this record
length.

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
