# What We Tried, What Broke, and What Fixed It

A plain-language account of how this project reached its current state. Written
for presentation: every problem below was found by testing, and each one changed
what the project could honestly claim.

For the numbers and methods, see `RESULTS.md`. For what still can't be claimed,
see `LIMITATIONS.md`.

---

## The goal

Predict **where and when Nairobi floods**, so warnings can go out before water
arrives rather than after.

---

## Attempt 1 — Read floods from radar satellites

**Idea.** Sentinel-1 radar sees through cloud. Water looks smooth to radar, so
dark patches should be floods.

**What happened.** The rule flagged *more* "water" in dry weather than wet. The
correlation with rainfall was −0.74 — strongly backwards.

**Why.** Wet soil reflects radar differently than expected in a built-up city.
The rule was actually detecting smooth dry surfaces: roads, rooftops, bare
ground.

**Outcome.** Abandoned. Kept in the thesis as a finding — this is a real trap in
urban radar flood mapping, not a coding mistake.

---

## Attempt 2 — Read floods from optical satellites

**Idea.** Sentinel-2 photographs the ground. Water is easy to spot in a photo.

**What happened.** Across 156 clear images, essentially no water was detected —
including during a day with 138 mm of rain.

**Why.** Two problems. Nairobi's floods run down streets 10–30 m wide, and each
image pixel covers 10 m, so floods are too small and too brief to catch. And
clouds block the view exactly when it's raining.

**Outcome.** Abandoned. Also reported as a finding.

---

## Attempt 3 — Use rainfall as the flood signal

**Idea.** Stop trying to *see* floods. Heavy rain causes floods, and rainfall
records are reliable, so use rainfall to say *when*, and the shape of the land to
say *where*.

**Outcome.** This became the project. Everything below is about making it work.

---

## Problem 1 — The training data was broken

The model wouldn't learn. Investigation found four faults:

| what was wrong | consequence |
|---|---|
| Rainfall was matched to the wrong dates | Only **6 of 703** days were labelled as floods instead of ~280 |
| Flood maps marked the *whole city* flooded or nothing | The model could not learn *where* — only two possible answers existed |
| 77 data layers, but only 13 were different | 6 GB file that kept crashing, for no added information |
| Training and testing data came from the same storms | The model could effectively memorise the answers |

**The date fault mattered most.** Two unrelated lists were being paired
position-by-position — like matching names to phone numbers by row order in two
different address books.

**Fix.** Rebuilt from scratch. Result: **2,024 training days, 282 real flood
days, 243 different flood maps**, and the file went from **6 GB to 2 MB** — small
enough to keep in the project itself, which also ended the crashes.

---

## Problem 2 — The scoring rule rewarded giving up

Training then ran, but at epoch 45 the model suddenly predicted "no flood, ever"
and stayed there — **while its score kept improving.**

**Why.** The formula grading the model had an accidental loophole. Since most
days have no flooding, a model that predicted nothing scored 0.09, while a model
genuinely trying scored 0.98 (worse). Giving up was rewarded.

**Fix.** Rewrote the scoring so giving up is now the worst outcome (1.44) rather
than the best.

**Worth presenting**, because the failure was invisible: the headline number
improved while the model became useless. Anyone watching only the score would
have called it a success.

---

## Problem 3 — Asking the model to do the impossible

Now training properly, the model scored 0.17 out of 1.0. We tested why.

| what we compared | score |
|---|---|
| Ignore rainfall, always guess the same places | 0.14 |
| Simple statistics instead of AI | 0.16 |
| **Our AI model** | **0.17** |
| **If we simply told it how much rain would fall** | **0.94** |

**The finding.** The model was barely beating a fixed guess — but given the
rainfall, it scored 0.94. So the hard part was never mapping the flood. It was
**knowing the rain was coming.**

We confirmed this directly: predicting Nairobi's rainfall 1–3 days ahead from
past rainfall alone is only slightly better than a coin flip, no matter which
method is used.

**Fix.** Stop asking the model to forecast weather. Split into two:

- **Model A** — must predict the rain itself. Scores **0.17**. Reported as the
  honest limit.
- **Model B** — is *given* a rainfall forecast, as real warning systems receive
  from meteorological services. Scores **0.94**. This is the system.

**This turned a disappointing number into the project's main insight:** the
bottleneck is weather forecasting, not flood mapping. So the system should plug
into a weather service rather than try to replace one.

---

## Problem 4 — Right timing, wrong places

We checked the model against real floods reported in the news.

**Timing — passed.** All **6 of 6** documented Nairobi floods (2018, 2019, 2020,
2023, 2024, 2026) fell on days the model flagged. Dry seasons stayed quiet 97.6%
of the time.

**Location — failed.** Compared against 37 flood-prone neighbourhoods mapped
independently by the Nairobi Rivers Regeneration Programme, our map was **no
better than chance**. Worst of all, **Mathare** — where 7,000+ people were
displaced in April 2024 — ranked as merely *average* risk.

**Why.** The model used only the natural shape of the land: low, flat ground near
streams. But Nairobi doesn't flood only because of terrain. It floods because
**drains are blocked and homes are built inside river corridors.** That is
exactly what Mathare, Kibera and Mukuru are — and natural terrain cannot see it.

**Fix.** Rebuilt the flood-risk map around drainage instead: *built-up land,
close to a water channel, on flat ground* — i.e. places where people live
directly on top of the drainage system.

| | old (land shape) | new (drainage) |
|---|---|---|
| Matches independently mapped flood areas? | No (*p* = 0.66) | **Yes (*p* = 0.03)** |
| Consistent when measured different ways? | No — flipped sign | **Yes, all five ways** |
| **Mathare's risk ranking** | **47th percentile** | **97th percentile** |

The data for this was already in the project, unused, the whole time.

---

## Where the project stands

**What we can back up:**

- Floods are predicted on the right *days* — confirmed against 6 real events
- The flood-risk map now agrees with independent mapping of Nairobi's flood areas
- Given a rainfall forecast, the system maps flooding accurately (0.94) on
  storms it has never seen

**What we cannot yet back up:**

- Predicting rain 1–3 days ahead from rainfall history — genuinely hard, and we
  showed why
- Street-level accuracy. We can say *which parts of the city*, not which roads
- The evidence for the drainage map rests on a small sample and should be
  strengthened

**One event shows the remaining limit clearly.** Before the November 2023 flood,
only 2.2 mm of rain had fallen all week — it looked like a dry spell. Then 39.6 mm
arrived and rivers burst their banks. Nothing looking only at past rainfall could
have called it. That is precisely why the system takes a weather forecast as
input instead of guessing.

---

## What we'd do next

1. **Use the city's real drainage map.** We inferred water channels from
   terrain; Nairobi's actual storm-drain network would be better.
2. **Rainfall varies across the city; our data doesn't.** We currently use one
   rainfall figure for all of Nairobi.
3. **More reported floods.** Six events is a modest sample for checking.
4. **Street-level checking.** Needs actual flood outlines, which news reports
   don't provide.

---

## The one-slide summary

> We set out to predict Nairobi's floods. Radar and optical satellites both
> failed for documented physical reasons. Rainfall worked. Along the way we found
> and fixed a broken data pipeline, a scoring rule that rewarded giving up, and a
> flood map that ignored the actual cause of urban flooding.
>
> The key result: **flood mapping is the easy part — the hard part is knowing the
> rain is coming.** Given a rainfall forecast, the system predicts flood extent
> accurately. So a real warning system should be built on top of a weather
> service, not instead of one.
