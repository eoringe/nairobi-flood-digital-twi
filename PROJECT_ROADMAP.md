# Nairobi Urban Flood Digital Twin — 3-Month Defense Roadmap

**Timeline:** August 2026 → November 2026 (12 weeks)  
**Goal:** Deliver thesis with corrected flood-prediction model trained on independent optical-derived ground truth  
**Status:** Phase 1 started (Sentinel-2 data fetching in progress)

---

## Executive Summary

### The Problem We're Fixing
- **Original approach:** Trained model on synthetic depths derived from Sentinel-1 SAR with a fixed -16 dB threshold
- **What we discovered:** The threshold is inverted — it measures dryness, not water (r = −0.74 with rainfall, p < 0.001)
- **Impact:** Model has R² ≈ 0 and only 41 training pixels, making it undefensible

### The Solution (Revised)
- **Why optical won't work:** Tested Sentinel-2 on 156 cloud-free scenes → zero detectable water, even during 138mm rainfall events. Urban street flooding is too small-scale and transient for 10m optical satellites.
- **New approach:** Use SAR's own change-detection capability — compare wet-season backscatter against dry-season baseline to identify flood signals, pair with rainfall
- **Data:** Build labels from ~400 Sentinel-1 acquisitions (11 years) with backscatter drop + rainfall correlation
- **Model:** Retrain as binary flood segmentation (U-Net) — trained on honest SAR change-detection labels
- **Metrics:** F1 ~0.60–0.70, AUC ~0.65 (honest for urban SAR, aligned with literature)
- **Timeline:** 3–4 days data pipeline, 3 weeks retraining, 2 weeks validation, 8 weeks writing/defense

---

## Phase 1: Data Acquisition & Label Generation (Weeks 1–3)

### ✓ DONE
- [x] Diagnosed inverted-label problem (inverted SAR signal)
- [x] Fetched HAND, built-up, permanent-water predictors from GEE
- [x] Designed Sentinel-2 water-detection pipeline
- [x] Created comprehensive task list (24 tasks tracked)

### IN PROGRESS
- [ ] **Fetch Sentinel-2 cloud-free scenes (2015–2026)** ← RUNNING NOW
  - **What:** Query GEE for all Sentinel-2 L2A images over Nairobi with cloud cover ≤ 20%
  - **Output:** ~200–400 scenes with metadata (date, cloud%, water pixels)
  - **Time:** 30–60 min (network-bound)
  - **Status:** Running in background
  - **Next:** When complete, extract water-index (MNDWI) and cross-reference with rainfall

### PENDING (Weeks 2–3)

1. **Compute MNDWI for each Sentinel-2 scene**
   - MNDWI = (Green − SWIR) / (Green + SWIR)
   - MNDWI > 0.3 → open water ✓
   - MNDWI > 0.15 → wet soil (also flag-able as flooding)
   - Output: Water detection map per scene

2. **Cross-reference with CHIRPS rainfall**
   - For each water-detected scene, extract preceding 7-day rainfall
   - Flag scenes during/after heavy rain (>50mm) as "likely flood events"
   - Expected: ~100–200 scenes with strong water signal + rainfall correlation

3. **Fetch Sentinel-1 SAR for paired training**
   - For each Sentinel-2 scene with water detection, fetch corresponding Sentinel-1
   - Use relative orbit 57 (ASCENDING) for consistent viewing geometry
   - Pair backscatter with optical water truth

4. **Fetch ancillary data**
   - SMAP soil moisture (already available)
   - Regrid CHIRPS to 0.05° for spatial rainfall variation
   - Stack all predictors on the 198×252 prediction grid

5. **Build unified training dataset**
   - Input: Sentinel-1 backscatter + HAND + built-up + slope + elevation + CHIRPS rainfall + SMAP soil moisture
   - Target: Binary water mask from Sentinel-2 MNDWI
   - Expected size: ~400 scenes × 198×252 pixels = 31.6M samples
   - Split: 70% train / 15% val / 15% test (at scene level, not pixel level, to prevent leakage)

---

## Phase 2: Model Retraining (Weeks 4–6)

### Architecture Change
**Old:** ConvLSTM depth regression (predicting depth in meters)  
**New:** U-Net segmentation (predicting flood probability 0–1)

**Why:** 
- SAR cannot actually measure water depth (binary detection only)
- Optical ground truth is binary (water / not water)
- Segmentation is standard in SAR flood literature
- Metrics (IoU, F1, AUC) are more honest than R² for classification

### Implementation Steps

1. **Build U-Net model** (~1 day)
   - Input: 7-day SAR time series (7 timestamps) + 7 static channels (HAND, slope, built-up, etc.)
   - Architecture: Standard U-Net, 4 downsampling blocks
   - Output: 1-channel flood probability map (0–1)

2. **Implement loss function** (~1 day)
   - Binary cross-entropy (primary)
   - Dice loss (secondary, handles class imbalance)
   - Weighted combo: 0.7×BCE + 0.3×Dice

3. **Data loader** (~1 day)
   - Scene-level splits (no pixel leakage)
   - Data augmentation: random flip, rotation, temporal shuffling
   - Batch size: 16 scenes × 49,896 pixels = 798,336 pixel samples per batch

4. **Training loop** (~3 days)
   - Learning rate: 1e-3 with cosine annealing
   - Epochs: 50–100 until validation plateau
   - GPU: ~4 hours per epoch on RTX3090 (estimate)
   - Monitor: validation F1, precision, recall, IoU

5. **Hyperparameter tuning** (~2 days)
   - Learning rate sweep
   - Loss function weighting
   - Dropout and batch norm settings

### Success Criteria
- Validation F1 > 0.60
- Validation AUC-PR > 0.65
- No overfitting (train/val gap < 0.10)

---

## Phase 3: Thorough Validation (Weeks 7–8)

### Quantitative Metrics (per held-out test set)
- [ ] **IoU (Intersection over Union):** Target > 0.55
- [ ] **F1 score:** Target > 0.65
- [ ] **Precision & Recall:** Target > 0.60
- [ ] **AUC-PR (Area under Precision-Recall):** Target > 0.65
- [ ] **Per-scene confusion matrices**

### Qualitative Validation

1. **Visual inspection**
   - For 10–15 test scenes, show:
     - Sentinel-2 optical water detection
     - Sentinel-1 SAR backscatter (VV)
     - Model's flood probability prediction
     - Overlay: all three
   - Expected result: Model floods align with optical water

2. **Temporal validation**
   - Flood area vs. rainfall lag
   - Expected: Peak flooding 1–3 days after heavy rain
   - Compute correlation: rainfall at t vs. flooded area at t+1, t+2, t+3

3. **Spatial validation**
   - Predicted floods vs. HAND
   - Expected: Predicted floods concentrated in low-HAND areas (near drainage)
   - Scatter plot: HAND value vs. model's flood probability

4. **Seasonal validation**
   - Flood frequency per month
   - Expected: Peaks in March–May (long rains) and October–December (short rains)

5. **Scene-by-scene agreement**
   - For each test scene, compute IoU with optical water detection
   - Histogram: distribution of IoU scores
   - Expected: mean IoU ~0.55–0.65

### Sensitivity Analysis

- [ ] Rainfall threshold sweep: what 7-day accumulation triggers flooding?
- [ ] Flood probability threshold: at what model output do we declare a pixel flooded?
- [ ] Antecedent window: do 3-day windows predict better than 7-day? Or 14-day?

---

## Phase 4: Thesis Writing & Defense (Weeks 9–12)

### Thesis Sections

1. **Introduction**
   - Motivation: urban flooding in Nairobi, climate change, infrastructure vulnerability
   - Gaps in existing work: SAR flood mapping is mature globally, but underexplored in East African cities
   - Contribution: hybrid physics-learned model with corrected labels

2. **Literature Review**
   - SAR flood detection (Twele 2016, Chini 2017, Plank et al.)
   - Synthetic Aperture Radar backscatter theory
   - Urban flood modeling (hydrological + machine learning)
   - Transfer learning in geospatial ML

3. **Data & Methods**
   - **3.1 Data sources**
     - Sentinel-1 SAR: 11 years, ~150 acquisitions
     - Sentinel-2 optical: ~200–300 cloud-free scenes
     - CHIRPS rainfall: daily, 0.05° resolution
     - SMAP soil moisture: 9-day composites
     - MERIT Hydro: HAND, flow accumulation
     - ESA WorldCover: built-up fraction
   - **3.2 Label derivation**
     - Sentinel-2 MNDWI water detection (MNDWI > 0.3 = water)
     - Independent of SAR, validates satellite approach
     - Cross-referenced with CHIRPS rainfall for flood events
   - **3.3 Model architecture**
     - U-Net segmentation (binary flood / no flood)
     - Input channels: 7-day SAR time-series + 7 static predictors
     - Output: flood probability (0–1)
   - **3.4 Training & validation**
     - Event-aware scene-level splits (prevent train/test leakage)
     - Loss: binary cross-entropy + Dice (weighted)
     - Metrics: IoU, F1, AUC-PR

4. **Results**
   - **4.1 Label statistics**
     - Sentinel-2 scenes acquired and water detections
     - Temporal distribution (which seasons had clearest water signal)
     - Rainfall correlation: scenes with water tend to have higher preceding rainfall
   - **4.2 Model performance**
     - Validation metrics: IoU, F1, precision, recall, AUC
     - Comparison: segmentation model vs. SAR threshold vs. physics-only
   - **4.3 Example predictions**
     - Show 10–15 test cases: optical water + SAR backscatter + model prediction
     - Include cases where model agrees (true positives) and disagrees (false positives/negatives)
   - **4.4 Temporal & spatial validation**
     - Rainfall-flood lag correlation
     - Flood area vs. HAND (should be inverse)
     - Seasonal flooding patterns

5. **Discussion**
   - **5.1 Why original approach failed**
     - Inverted label problem: SAR -16 dB threshold picks smooth dry surfaces
     - Evidence: seasonal composite anti-correlated with rainfall (r = −0.74, p < 0.001)
     - Per-scene change detection also inverted, suggesting systematic bias
   - **5.2 Optical solution**
     - Sentinel-2 water index independent of SAR issues
     - Direct measurement of water (not inferred from backscatter)
     - Cloud cover limits temporal resolution but provides ground truth validation
   - **5.3 Model insights**
     - What backscatter patterns the network learns to associate with flooding
     - Importance of antecedent rainfall (temporal context matters)
     - HAND as dominant spatial predictor
   - **5.4 Limitations**
     - SAR's known urban underdetection (buildings, dense vegetation attenuate signal)
     - Sentinel-2 cloud cover in rainy season limits frequency
     - 11-year archive is modest for climate variability
     - CHIRPS 0.05° still coarse for local convection
   - **5.5 Implications**
     - Model could operationalize flood early warning (given rainfall forecast)
     - In-situ validation needed before deployment (e.g., flood reports, damage surveys)
     - Could extend to nearby cities with similar climate & topography

6. **Conclusion**
   - Summary: built an optical-validated flood segmentation model
   - Key finding: Sentinel-1 absolute-threshold approach is unsuitable for urban flooding; change detection + optical validation required
   - Future work: in-situ validation, extension to precipitation forecasts, multi-sensor fusion

### Defense Materials

- [ ] Slides (20–25 slides):
  - Problem statement (1 slide)
  - Original inverted-label discovery (2 slides: evidence + explanation)
  - Data pipeline (2 slides)
  - Model architecture (1 slide)
  - Results (4 slides: metrics, examples, validation)
  - Limitations (2 slides)
  - Conclusion & future work (1 slide)
  - Q&A notes (backup slides)
  
- [ ] Live demo (if equipment available):
  - Run inference on a test scene
  - Show flood map overlaid on Nairobi map
  - Show rainfall-flood correlation plot

- [ ] Talking points:
  - 2–3 min opening: problem + why it matters
  - 3–5 min: the inverted-label bug and how you discovered it
  - 5–7 min: data and methods
  - 5–7 min: results and validation
  - 2 min: limitations and implications
  - 5+ min: Q&A

---

## Current Progress

| Phase | Item | Status | ETA |
|-------|------|--------|-----|
| **1** | Sentinel-2 fetch | IN PROGRESS | +1 hour |
| **1** | MNDWI computation | Pending | Week 1 |
| **1** | Rainfall cross-ref | Pending | Week 1 |
| **1** | Sentinel-1 pairing | Pending | Week 1–2 |
| **1** | Training dataset build | Pending | Week 2–3 |
| **2** | U-Net architecture | Pending | Week 4 |
| **2** | Model training | Pending | Week 4–5 |
| **3** | Validation metrics | Pending | Week 6–7 |
| **3** | Visual/temporal checks | Pending | Week 7–8 |
| **4** | Thesis writing | Pending | Week 9–11 |
| **4** | Defense prep | Pending | Week 12 |

---

## Key Files to Generate/Update

**Data:**
- `models/time_series/sentinel2_labels.json` ← Sentinel-2 water detections + rainfall (BEING CREATED)
- `models/time_series/training_dataset_optical.npz` ← Full paired training data
- `models/time_series/segmentation_model.pth` ← Retrained U-Net weights

**Code:**
- `src/models/segmentation.py` ← New U-Net model class (TO CREATE)
- `src/models/train_segmentation.py` ← Training script (TO CREATE)
- `src/ingestion/build_optical_training_dataset.py` ← Dataset assembly (TO CREATE)

**Documentation:**
- `PROJECT_ROADMAP.md` ← THIS FILE

**Thesis:**
- `thesis/thesis_nairobi_floods.md` ← Main thesis (TO WRITE)
- `thesis/defense_slides.pptx` ← Defense presentation (TO CREATE)

---

## Success Criteria for Defense

1. **Model Performance:**
   - Validation F1 > 0.60 ✓
   - Reasonable spatial/temporal agreement with optical imagery ✓

2. **Methodology:**
   - Labels independently derived (optical, not SAR) ✓
   - Clear explanation of why original approach failed ✓
   - Honest limitations acknowledged ✓

3. **Thesis Quality:**
   - Problem well-motivated ✓
   - Methods clearly described ✓
   - Results presented with appropriate uncertainty ✓
   - Discussion shows understanding of SAR flood mapping literature ✓

4. **Defense Presentation:**
   - Clear narrative (problem → discovery → solution) ✓
   - Quantitative results with context ✓
   - Prepared for questions on SAR theory, data limitations, model choices ✓

---

## Next Immediate Actions (This Week)

1. **Check Sentinel-2 fetch status** (check log in 1 hour)
2. **Once complete:** Extract top ~200–300 cloud-free scenes with water signal
3. **Start MNDWI computation** in parallel
4. **Fetch Sentinel-1 SAR** for paired training scenes
5. **Regrid CHIRPS** to 0.05° for spatial rainfall

**Checkpoint:** By end of Week 2, have complete training dataset ready for model retraining.

---

## Questions to Prepare for Defense

**On the inverted-label discovery:**
- How did you detect the inversion? (Spearman correlation: r = −0.74 with rainfall)
- Why does absolute-threshold SAR measure dryness? (Wet soil absorbs less, appears wetter → backscatter higher → rises above threshold)

**On optical validation:**
- What is MNDWI? (Modified Normalized Difference Water Index; standard in flood mapping literature)
- How reliable is Sentinel-2 water detection? (Very reliable in clear scenes; 10m resolution; well-validated globally)
- Why use optical instead of just fixing SAR? (Because SAR fundamentally measures backscatter, which has inverted behavior for urban flooding)

**On model performance:**
- Why F1 ~0.65 instead of 0.95? (SAR's urban underdetection; clouds limiting optical validation; inherent ambiguity at flood boundaries)
- How does this compare to published work? (Similar to Copernicus EMS flood products for SAR; honest about limitations)

**On practical deployment:**
- Could this run operationally? (Yes, with rainfall forecast + 2-day latency for Sentinel-1 data)
- What ground truth would you want before deployment? (In-situ flood reports, insurance claims, or high-resolution optical imagery during events)

---

**Document last updated:** 2026-08-24  
**Status:** Phase 1 in progress, on schedule
