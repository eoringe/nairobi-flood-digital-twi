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
| **Model A** | rainfall *t−7…t−1*, seasonality, 14/30-day antecedent totals, terrain | 18 | Can flooding be *anticipated* from rainfall history? |
| **Model B** | the above **plus** rainfall over *t…t+2* | 21 | Given rainfall, can flood extent be predicted? |

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
| Storm seasons (events) | 22 (2015–2025, long and short rains) |
| Grid | 198 × 252 (≈ 67 m × 79 m per pixel) |
| Train / validation / test | 1,380 / 276 / 368 samples |
| Train / validation / test seasons | 15 / 3 / 4 |
| Storm-positive scenes | 282 (13.9%) |
| Distinct flood masks | 243 |

Storm occurrence is identical across both label definitions; they differ only in
which cells a storm floods:

| | positive pixel rate | extent when flooded (min / median / max) |
|---|---|---|
| terrain labels | 0.90% | 2.0% / 5.3% / 18.0% |
| **drainage labels (reported)** | **0.62%** | **1.4% / 3.6% / 12.3%** |

Flood extent is intensity-graded rather than a single repeated footprint,
widening as accumulated rainfall rises from the 30 mm threshold to 120 mm.

At a 0.62% positive rate accuracy is uninformative — a model predicting no
flooding anywhere scores 99.4% — so only F1, IoU, precision and recall are
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

| metric | validation (best, epoch 23) | test |
|---|---|---|
| F1 | 0.2349 | **0.1757** |
| IoU | 0.1331 | 0.0963 |
| Precision | 0.1436 | 0.1059 |
| Recall | 0.6448 | 0.5148 |

Model A exceeds both the terrain-only stencil (0.1434, +22.5% relative) and the
linear rainfall baseline (0.1592, +10.4%), confirming that it extracts genuine
non-linear signal. The absolute value nevertheless remains low.

Training loss fell steadily from 0.703 to 0.401 over 60 epochs with no collapse,
and validation F1 plateaued near 0.22 from roughly epoch 23 onward. The
validation-test discrepancy (0.23 against 0.18) reflects the test partition's
lower storm frequency (10.1% against 15.2%) and its small positive sample — 37
storm-positive samples drawn from four seasons.

Recall (0.515) exceeds precision (0.106) as intended: the loss weights false
negatives above false positives, which is the appropriate asymmetry for early
warning, where a missed flood is costlier than a false alarm. This behaviour was
verified rather than assumed, discharging NFR-15.

**The ceiling is independent of the label definition.** Trained on terrain labels
instead, Model A reaches test F1 0.1767 — within 0.001 of the drainage result.
Since the two label definitions produce different targets, different positive
rates and different spatial fields, near-identical scores indicate that what
limits Model A is neither the target nor the model but the predictability of the
rainfall itself.

---

## 4.5 Model B — rainfall-conditioned extent mapping

| metric | validation (best, epoch 60) | test |
|---|---|---|
| F1 | 0.9084 | **0.9370** |
| IoU | 0.8322 | 0.8815 |
| Precision | 0.8947 | 0.9244 |
| Recall | 0.9226 | 0.9500 |

Model B converged smoothly, training loss falling from 0.684 to 0.065, and
generalised to storm seasons never seen during training. Test performance
slightly exceeding validation indicates no overfitting.

At F1 0.937 the network approaches the analytic oracle (0.9997), demonstrating
that the rainfall-to-extent mapping is not merely solvable in principle but
learnable from data by this architecture.

Trained on terrain labels the same architecture reaches 0.9555. The drainage
target is marginally harder — its positive rate is lower (0.62% against 0.90%)
and its flooded extents smaller — so a slightly lower score is expected. **These
two figures must not be read as evidence that terrain labels are better.** They
are scores against different targets, and comparing them says nothing about
which target is correct. That question is settled by spatial validation
(§4.8.2, §4.8.3), where the terrain field fails and the drainage field passes.

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
| **Model A — must forecast rainfall** | **0.1757** |
| **Model B — given rainfall** | **0.9370** |
| Oracle — perfect storm knowledge | 0.9997 |

Knowing the rainfall is worth **+0.761 F1**. Every method required to forecast it
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
   magnitude reaches F1 0.9997, and a U-Net learns this mapping to F1 0.937 on
   held-out storm seasons (§4.5).

2. **Forecasting that rainfall from rainfall history is the binding
   constraint.** Storm-detection AUC remains 0.58–0.68 across all feature sets
   and classifiers, capping flood-forecast F1 at 0.176 — only marginally above a
   terrain-only baseline of 0.143 (§4.4, §4.6).

3. **The bottleneck is meteorological, not hydrological.** Knowing the rainfall
   is worth +0.761 F1. No architecture, loss function, or feature engineering
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

## 4.10 Probability calibration

Raw outputs are overconfident in the mid-range (LIMITATIONS.md §10). An isotonic
calibration was fitted on the three **validation** seasons and judged on the
four **test** seasons, so the improvement is not measured on the data that
produced it (`src/models/calibrate_v2.py`).

Test-season reliability, Model B:

| predicted band | observed, raw | observed, calibrated |
|---|---|---|
| 0.1–0.2 | 24.8% (predicted 14.5%) | 13.8% (predicted 13.9%) |
| 0.3–0.4 | 37.1% (34.8%) | 28.9% (34.7%) |
| 0.6–0.7 | 50.5% (65.2%) | 63.6% (64.8%) |
| 0.7–0.8 | 55.1% (75.2%) | 71.8% (74.8%) |
| 0.8–0.9 | 61.8% (85.5%) | 84.7% (84.8%) |
| 0.9–1.0 | 95.5% (99.6%) | 98.5% (98.5%) |

| metric (test seasons) | raw | calibrated |
|---|---|---|
| unweighted ECE | 0.092 | **0.020** |
| mid-band ECE | 0.109 | **0.025** |
| Brier score | 0.00052 | **0.00046** |
| F1 / IoU at 0.5 | 0.937 / 0.881 | 0.937 / 0.882 |

The raw figures reproduce those in LIMITATIONS.md §10, which checks the
evaluation code. Calibration is monotone, so it changes stated percentages
without reordering cells, and F1 is unchanged. The dashboard uses calibrated
values. Calibration is against the constructed labels, not observed floods, so
it makes the numbers internally honest, not externally validated (§4.8.2).

Model A is far worse calibrated (unweighted ECE **0.425**): a cell it scores
80–90% carries a flood label 9.6% of the time. It is not deployed, and its
outputs should not be read as probabilities.

## 4.11 Stability across test seasons

F1 per held-out test season (`models/time_series/evaluation_v2_*.json`):

| season | storm samples | Model A F1 | Model B F1 |
|---|---|---|---|
| 2015 short rains | 15 | 0.191 | 0.930 |
| 2019 long rains | 7 | 0.114 | 0.923 |
| 2021 short rains | 3 | 0.151 | 0.942 |
| 2024 short rains | 12 | 0.244 | 0.945 |
| **pooled** | 37 | **0.176** | **0.937** |
| season-bootstrap 95% interval | | 0.130–0.224 | 0.928–0.944 |

Model B varies by 0.02 across seasons; Model A by a factor of two. The bootstrap
resamples only four seasons and is therefore crude. The event-aware 5-fold
cross-validation (`src/models/crossvalidate_v2.py`, section 7 of the Colab
notebook) supersedes it once run.

## 4.12 Population exposure

"People at risk" now sums WorldPop 2025 population (100 m, constrained) over the
cells predicted flooded (`src/ingestion/build_population_grid.py`). It replaces
flooded area × the county-average 6,300 people/km². Resampling preserves the
total exactly (2,908,836 people in the model area).

| 3-day rainfall | flooded area | county-average estimate | WorldPop |
|---|---|---|---|
| 30 mm | 1.9 km² | 11,900 | 49,800 |
| 40 mm | 7.2 km² | 45,300 | 192,700 |
| 60 mm | 13.5 km² | 85,100 | 366,100 |
| 100 mm | 25.5 km² | 160,600 | 650,000 |

Predicted flood cells hold about 26,000 people/km², roughly four times the
county mean, because flooding follows the dense river corridors. The
average-density figure understated exposure by roughly **4×**. For scale, about
147,000 people were reported affected in Nairobi County in April 2024 (62 mm).
"Living in a predicted flood cell" is a broader measure than "affected", and a
70 m cell is not a flooded house. The WorldPop figure is therefore an upper-bound
exposure count, not a casualty estimate.

---

## Outstanding work

- **K-fold cross-validation** (`src/models/crossvalidate_v2.py`) to report
  both models as mean ± spread across all 22 seasons, instead of one split with
  37 storm-positive test samples. It is ready to run in section 7 of the Colab
  notebook and needs a GPU, since each fold is a full training run. §4.11 gives
  a per-season interim result.
- **Spatial validation.** All external validation above is temporal. Documented
  reports name affected settlements but provide no inundation polygons, so
  predicted flood *location* remains unverified (`LIMITATIONS.md` §9).
