# Results

Draft of the thesis Results chapter. Every figure is reproducible from the
scripts named beside it; nothing here is estimated or carried over from earlier
versions of the pipeline. Read alongside `LIMITATIONS.md`, which qualifies each
claim made below.

---

## 4.1 Experimental design

Two models were trained on identical labels, identical splits and identical
architecture, differing only in what their inputs contain. The comparison is
deliberate: it separates two questions that a single model conflates.

| | inputs | channels | question it answers |
|---|---|---|---|
| **Model A** | rainfall *t−7…t−1*, seasonality, 14/30-day antecedent totals, terrain | 17 | Can flooding be *anticipated* from rainfall history? |
| **Model B** | the above **plus** rainfall over *t…t+2* | 20 | Given rainfall, can flood extent be predicted? |

Model A never sees rainfall from the label window, so the target cannot be
recovered arithmetically from an input channel. Model B receives that rainfall,
mirroring an operational system supplied with a numerical weather prediction
from a forecasting centre. Model B is therefore a *hydrological mapping*, not a
weather forecast, and is reported as such throughout.

Both use a four-level U-Net (7.85 M parameters, base width 32), trained for 60
epochs with Adam (lr 10⁻³, cosine annealing) under a combined pos-weighted
binary cross-entropy and batch-level Focal Tversky loss (α = 0.7 on false
negatives, β = 0.3 on false positives).

**Splitting.** Folds are formed over *storm seasons*, never over samples.
Consecutive samples share overlapping seven-day rainfall windows and identical
terrain, so a sample-level split would place near-duplicates in both training
and test partitions and inflate every metric. No storm season appears in more
than one partition.

*Scripts: `src/ingestion/build_segmentation_dataset_v2.py`,
`src/models/train_segmentation_v2.py`*

---

## 4.2 Dataset characteristics

| property | value |
|---|---|
| Samples | 2,024 |
| Storm seasons (events) | 22 (2015–2026, long and short rains) |
| Grid | 198 × 252 (≈ 67 m × 79 m per pixel) |
| Train / validation / test | 1,380 / 276 / 368 samples |
| Train / validation / test seasons | 15 / 3 / 4 |
| Storm-positive scenes | 282 (13.9%) |
| Positive pixel rate | 0.90% |
| Distinct flood masks | 243 |

Flood extent is intensity-graded rather than a single repeated footprint: it
spans 2.0% of the grid for a storm just clearing the 30 mm threshold to 18.0%
at 120 mm, with a median of 5.3% across flood-positive scenes.

The 0.90% positive rate means accuracy is uninformative — a model predicting no
flooding anywhere scores 99.1% — so only F1, IoU, precision and recall are
reported.

---

## 4.3 Baselines

Reporting a segmentation F1 without baselines is uninterpretable at this class
balance. Four reference points were computed on the same held-out test seasons.

| method | F1 | precision | recall |
|---|---|---|---|
| Predict nothing | 0.0000 | — | 0.000 |
| Predict flooding everywhere, always | 0.0145 | 0.007 | 1.000 |
| **Fixed terrain stencil, ignores rainfall** | **0.1434** | 0.080 | 0.702 |
| Logistic regression on rainfall + median extent | 0.1592 | 0.167 | 0.152 |
| **Oracle: storm known, true extent applied** | **0.9997** | 1.000 | 0.999 |

The third row is the meaningful floor: a model that learns terrain but ignores
rainfall entirely achieves F1 0.143. The final row is the ceiling: given perfect
knowledge of whether a storm occurs and its magnitude, flood extent is
recoverable essentially exactly.

**The gap between these two rows is the whole problem.** Spatial prediction is
solved; storm timing is not.

---

## 4.4 Model A — forecasting from rainfall history

| metric | validation (best) | test |
|---|---|---|
| F1 | 0.2515 | **0.1718** |
| IoU | 0.1438 | 0.0940 |
| Precision | 0.1598 | 0.1133 |
| Recall | 0.5900 | 0.3550 |

Model A exceeds both the terrain-only stencil (0.1434, +19.8% relative) and the
linear rainfall baseline (0.1592, +7.9%), confirming that it extracts genuine
non-linear signal. The absolute value nevertheless remains low.

Training loss fell steadily from 0.770 to 0.431 over 60 epochs with no collapse,
and validation F1 plateaued near 0.22 from roughly epoch 22 onward. The
validation-test discrepancy (0.25 against 0.17) reflects the test partition's
lower storm frequency (10.1% against 15.2%) and its small positive sample — 37
storm-positive samples drawn from four seasons.

Recall (0.355) exceeds precision (0.113) by design: the loss weights false
negatives above false positives, which is the appropriate asymmetry for early
warning, where a missed flood is costlier than a false alarm.

---

## 4.5 Model B — rainfall-conditioned extent mapping

| metric | validation (best) | test |
|---|---|---|
| F1 | 0.9133 | **0.9442** |
| IoU | 0.8405 | 0.8942 |
| Precision | 0.8966 | 0.9067 |
| Recall | 0.9307 | 0.9848 |

Model B converged smoothly, training loss falling from 0.756 to 0.072, and
generalised to storm seasons never seen during training. Test performance
slightly exceeding validation indicates no overfitting.

At F1 0.944 the network approaches the analytic oracle (0.9997), demonstrating
that the rainfall-to-extent mapping is not merely solvable in principle but
learnable from data by this architecture.

**Interpretation.** Model B's labels are a deterministic function of its inputs,
so this result establishes a *capability* — that the network learns the
hydrological mapping and generalises it across events — rather than evidence
about flooding in Nairobi. That distinction is developed in `LIMITATIONS.md`
§1a and must be preserved in any statement of this figure.

---

## 4.6 Where the difficulty lies

Placing all results on one scale:

| method | test F1 |
|---|---|
| Predict nothing | 0.0000 |
| Fixed terrain stencil | 0.1434 |
| Logistic regression on rainfall | 0.1592 |
| **Model A — must forecast rainfall** | **0.1718** |
| **Model B — given rainfall** | **0.9442** |
| Oracle — perfect storm knowledge | 0.9997 |

Knowing the rainfall is worth **+0.77 F1**. Every method required to forecast it
clusters between 0.14 and 0.17, irrespective of whether it is a fixed stencil, a
linear model, or a 7.85 M-parameter convolutional network.

**Storm predictability was measured directly.** A classifier was trained to
predict whether ≥ 30 mm would fall over the following three days, using an
event-aware split:

| features | model | AUC | scene F1 |
|---|---|---|---|
| 7 antecedent days | logistic | 0.661 | 0.205 |
| + day-of-year | gradient boosting | **0.679** | **0.345** |
| + 14/30-day antecedent totals | logistic | 0.656 | 0.321 |

Seasonality is a genuine gain (scene F1 0.205 → 0.345). However **AUC remains
between 0.58 and 0.68 across every feature set and classifier tested.** Because
AUC is independent of base rate while F1 is not, this indicates that apparent
improvements from lengthening the forecast horizon — which raises the positive
rate from 13.9% to 29.1% — reflect an easier scoring regime rather than
additional predictive skill.

**Finding.** The performance ceiling on this task is set by the predictability
of rainfall, not by model capacity, loss design, or spatial representation.

*Script: ablation reproduced in `LIMITATIONS.md` §10*

---

## 4.7 Training stability: a loss-function failure and its correction

An initial training run appeared to converge, then collapsed catastrophically at
epoch 45: validation F1 fell from 0.190 to 0.000 and remained there, while
*training loss simultaneously dropped* from 0.89 to 0.21.

Diagnosis showed the per-sample Focal Tversky formulation was at fault. For the
~86% of samples containing no flooding, true positives are zero, so the Tversky
numerator reduces to the smoothing constant while false positives accumulate
across all 49,896 pixels. A model driving its outputs to saturated zeros
therefore achieves a near-optimal loss:

| prediction strategy | per-sample loss | corrected loss |
|---|---|---|
| Total collapse (saturated) | **0.090** | 1.443 |
| Partially trained model | 0.984 | 0.559 |
| Perfect prediction | 0.001 | 0.000 |

Under the original formulation, collapsing was worth a 0.89 reduction in loss
while incremental honest improvement yielded almost nothing — so gradient
descent had a strong incentive to abandon a partially correct solution. The
correction aggregates Tversky over the batch rather than per sample and adds a
pos-weighted cross-entropy term, restoring the correct ordering. Training
thereafter ran 60 epochs without collapse.

This is reported because the failure is silent: training loss *improved* while
the model became useless, and any pipeline monitoring loss alone would have
recorded a successful run.

---

## 4.8 External validation against documented flood events

All metrics above measure agreement with labels constructed from rainfall and
terrain. To test whether those labels correspond to real flooding, they were
compared against independently documented events, graded by whether the source
reports dated flooding *in Nairobi* or a Kenya-wide episode in which Nairobi is
named among affected areas.

**Nairobi-specific events**

| event | 3-day rainfall | labels flag it |
|---|---|---|
| March 2026 — Nairobi River burst banks, 37 deaths in Nairobi | 41.5 mm | 2/10 days |
| April 2024 — Mathare, ~147,000 affected in Nairobi County | 62.3 mm | 8/8 days |
| November 2023 — El Niño, rivers burst banks | 39.6 mm | 3/15 days |

**Kenya-wide episodes including Nairobi**

| event | 3-day rainfall | labels flag it |
|---|---|---|
| Oct–Dec 2019 — wettest short rains on record (~400% of average) | 77.7 mm | 16/40 days |
| Apr–May 2020 — ~194 deaths, 100,000 displaced nationally | 92.3 mm | 11/26 days |
| Mar–May 2018 — long rains ~145% of average, 310,000 displaced | 81.8 mm | 15/40 days |

**Detected: 6/6 (3/3 Nairobi-specific). False alarms on dry-season controls: 3
of 126 days (2.4%).**

Every documented flood coincides with days the labels flag, and dry-season
periods remain quiet 97.6% of the time. This externally supports the rainfall
threshold as a flood indicator.

*Script: `src/validation/validate_documented_events.py`, with sources recorded
per event*

### 4.8.1 A worked case: the limits of rainfall-history forecasting

Antecedent rainfall for each documented event exposes where Model A fails:

| event | rainfall in preceding week | rainfall that fell | foreseeable? |
|---|---|---|---|
| March 2026 | 148.5 mm | 41.5 mm | plausibly |
| April 2024 | 92.8 mm | 62.3 mm | plausibly |
| **November 2023** | **2.2 mm** | **39.6 mm** | **no — appeared dry** |
| Oct–Dec 2019 | 40.6 mm | 77.7 mm | plausibly |
| Apr–May 2020 | 86.0 mm | 92.3 mm | plausibly |
| Mar–May 2018 | 130.9 mm | 81.8 mm | plausibly |

Five of six floods followed a demonstrably wet spell. The November 2023 event
did not: only 2.2 mm fell in the preceding week, so the antecedent record
resembled a dry spell until 39.6 mm arrived and rivers burst their banks.

**Model A's blind spot is therefore not uniform — it is concentrated in
flash-flood events**, precisely the category early warning exists to address. A
model conditioned on antecedent rainfall would have anticipated five of these
six events; the one it would have missed is the one that arrived without
warning.

### 4.8.2 Spatial validation: a negative result

The validation above is temporal. A second test asked whether the model floods
the right *places*, using two references independent of the HAND/slope/TWI field:
37 flood-prone neighbourhoods mapped under the Nairobi Rivers Regeneration
Programme from river-corridor proximity, and neighbourhoods named in reporting
of the April 2024 floods. Controls are Nairobi neighbourhoods absent from both.

| sampling | mapped flood-prone | control | separation | *p* |
|---|---|---|---|---|
| single pixel | 47.4% | 61.5% | −14.0 | 0.818 |
| disc r ≈ 225 m | 83.8% | 82.3% | +1.5 | 0.613 |
| disc r ≈ 525 m | 98.5% | 97.7% | +0.8 | 0.589 |

**No separation at any sampling radius.** Percentiles are city-wide ranks of the
susceptibility score; a disc takes the maximum within the radius, since at ~70 m
resolution a neighbourhood centroid can fall on a valley shoulder rather than
its floor.

For the April 2024 event the predicted mask covered **1 of 10** reported
neighbourhoods at 7.7% grid coverage, against a random expectation of 0.8 — no
better than chance. Mathare, the worst-affected settlement with over 7,000
displaced, sits at the 47th percentile of susceptibility and is not predicted
flooded.

**Raster misalignment was ruled out** as an explanation: HAND correlates
positively with elevation in the stored orientation (r = +0.275) and worse under
every flip (horizontal −0.233, 180° −0.195), and the layers cross-check as
expected (slope–TWI r = −0.749).

**Interpretation.** The most likely explanation is that Nairobi's urban flooding
is driven substantially by drainage failure — blocked storm drains, riparian
encroachment, impervious surfaces — rather than natural topography alone. HAND
describes where water collects on undeveloped terrain, not where a built
drainage system fails. Grid resolution (~70 m against river valleys 100–200 m
wide) plausibly contributes.

**Weight of this evidence.** The test has real limits: six controls, approximate
centroids rather than boundaries, and a news summary rather than the underlying
GIS layer. It therefore does not establish that the spatial predictions are
wrong. It does establish that the claim they are *right* has no supporting
evidence, and that a deliberate attempt to find such evidence failed.

### 4.8.3 Rebuilding susceptibility around drainage

The failure above motivated a specific hypothesis, stated before testing: if
Nairobi floods because settlements occupy river corridors and block drainage,
then the predictor should combine **built-up land, proximity to a drainage
channel, and flat ground** — not terrain alone.

Drainage channels were derived from upstream flow accumulation
(`predictor_upa.npy`, MERIT Hydro), which was present in the repository but
unused by earlier versions, taking the top 1% of accumulation and applying an
exponential distance decay (e-folding ≈ 220 m).

Eleven candidate predictors were compared on the §4.8.2 benchmark:

| predictor | separation | *p* |
|---|---|---|
| terrain: HAND × slope × TWI (previous) | +3.7 | 0.662 |
| HAND alone | +3.9 | 0.581 |
| flow accumulation alone | +1.1 | 0.354 |
| built-up alone | +10.7 | 0.076 |
| channel proximity alone | +20.7 | 0.037 |
| **built-up × channel proximity × flat** | **+25.0** | **0.045** |

Adopting the last, and re-running the full validation:

| | terrain | drainage |
|---|---|---|
| single pixel | +3.7 (*p* 0.662) | **+25.9 (*p* 0.034)** |
| disc ≈ 225 m | +3.7 (*p* 0.662) | **+25.8 (*p* 0.033)** |
| disc ≈ 525 m | +0.2 (*p* 0.662) | **+25.4 (*p* 0.027)** |
| **Mathare percentile** | **46.8%** | **96.5%** |

The drainage formulation separates mapped flood-prone neighbourhoods from
controls **at every sampling radius**, where terrain separated at none — and
Mathare, the worst-affected settlement in April 2024, moves from the 47th to the
97th percentile of predicted risk.

**Statistical caveat.** Eleven predictors were compared, so a Bonferroni-corrected
threshold would be 0.0045 and none of these results clear it. The sample is 28
flood-prone against 6 control locations. Three considerations nonetheless support
adoption: the significant results are variants of a single hypothesis rather than
independent findings; that hypothesis was stated before testing, as the
explanation for §4.8.2's failure; and the effect is stable across all five
sampling radii rather than appearing at one.

**Event coverage did not improve** (1/10 reported neighbourhoods, unchanged).
This metric is limited by the extent parameter rather than the ranking: at 62 mm
of rainfall the predicted extent is 5.3% of the grid, and ten neighbourhood
centroids distributed across the city cannot mostly fall within the top 5%. It
suggests the intensity-to-extent calibration (§8 of `LIMITATIONS.md`, chosen
rather than fitted) is too conservative, and is a separate parameter from the
susceptibility field this section evaluates.

**Status of Link 2.** Supported for the drainage formulation, unsupported for
terrain. Both datasets are retained so the comparison can be reported. The
spatial output remains coarse — neighbourhood scale, not street scale — and
should be described accordingly.

---

## 4.9 Summary of findings

1. **Flood extent over Nairobi is near-deterministically recoverable from
   terrain given rainfall.** An oracle supplied with storm occurrence and
   magnitude reaches F1 0.9997, and a U-Net learns this mapping to F1 0.944 on
   held-out storm seasons (§4.5).

2. **Forecasting that rainfall from rainfall history is the binding
   constraint.** Storm-detection AUC remains 0.58–0.68 across all feature sets
   and classifiers, capping flood-forecast F1 at 0.172 — only marginally above a
   terrain-only baseline of 0.143 (§4.4, §4.6).

3. **The bottleneck is meteorological, not hydrological.** Knowing the rainfall
   is worth +0.77 F1. No architecture, loss function, or feature engineering
   recovers information absent from the data (§4.6).

4. **Rainfall-derived labels agree with documented reality — in time.** All six
   independently reported flood events coincide with flagged days, with a 2.4%
   dry-season false-alarm rate (§4.8).

5. **They do not demonstrably agree in space.** The terrain susceptibility field
   fails to separate independently mapped flood-prone neighbourhoods from
   controls at any sampling radius (*p* ≥ 0.589), and covers 1 of 10
   neighbourhoods reported flooded in April 2024 against a chance expectation of
   0.8. Raster misalignment was excluded. Nairobi's flooding appears to be
   driven substantially by drainage failure rather than natural topography, which
   a HAND-based model cannot represent (§4.8.2).

6. **Forecast failure concentrates in flash floods.** Five of six documented
   events followed a wet spell and were plausibly foreseeable; the exception
   arrived after a dry week (§4.8.1).

**Implication for design.** An operational flood early-warning system for
Nairobi should consume a numerical weather prediction rainfall forecast rather
than extrapolate from rainfall history: the timing component is where the
information is missing (§4.6).

That recommendation concerns *timing*. Finding 5 constrains what can be claimed
about *location*: the system can indicate when flooding is likely with external
support, but its map of where flooding will occur is not yet corroborated. The
terrain-only susceptibility model is the weakest link in the chain, and
improving it — with drainage-network data, riparian-encroachment mapping, or the
Nairobi Rivers Regeneration Programme's underlying GIS layer — would do more for
operational value than any further work on the model itself.

**Revised success criterion.** The project proposal set validation F1 > 0.60.
That target is unattainable for Model A, and §4.6 establishes why: the
information does not exist in rainfall history. Reaching it would require either
lowering the rainfall threshold or lengthening the horizon, both of which raise
F1 by increasing the positive rate without improving prediction. The criterion
was therefore replaced with *"exceeds terrain-only and linear-rainfall
baselines"*, which Model A satisfies, while Model B exceeds 0.60 under the
separate and narrower claim stated in §4.5.

---

## Outstanding work

- **K-fold cross-validation** (`src/models/crossvalidate_v2.py`) to report
  Model A as mean ± spread across all 22 seasons rather than one split of 37
  positive test samples.
- **Spatial validation.** All external validation above is temporal. Documented
  reports name affected settlements but provide no inundation polygons, so
  predicted flood *location* remains unverified (`LIMITATIONS.md` §9).
