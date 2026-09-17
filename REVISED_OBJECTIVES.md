# Revised Research Objectives and Questions

Revised to match the system as built. Changes from the approved proposal are
listed at the end, each with the evidence behind it.

---

## 1.3 Research Objectives

### 1.3.1 General Objective

To develop a deep-learning-driven 3D Urban Digital Twin dashboard for
near-real-time flood forecasting, risk assessment and flood-aware navigation in
Nairobi County.

### 1.3.2 Specific Objectives

i. To analyse the current state of hydrologic cyberinfrastructure and
data-driven surrogate modelling for urban flood prediction.

ii. To study and evaluate existing web-based hydrologic visualisation systems and
their application in flood monitoring and risk communication.

iii. To identify key gaps and limitations in current hydrologic forecasting and
visualisation systems.

iv. To design and develop a convolutional deep-learning surrogate model that
predicts the probability of flooding across Nairobi from rainfall and terrain
data, and a hardware-accelerated 3D Urban Digital Twin dashboard that delivers
near-real-time flood forecasts, time-to-flood warnings and flood-aware route
guidance.

v. To evaluate the predictive accuracy, probability calibration and
computational performance of the system against held-out storm seasons and
documented Nairobi flood events.

---

## 1.4 Research Questions

i. What is the current state of hydrologic cyberinfrastructure and data-driven
surrogate modelling for urban flood prediction?

ii. How have existing web-based hydrologic visualisation systems been applied in
flood monitoring, and what are their strengths and limitations?

iii. What are the key gaps and challenges in current hydrologic forecasting and
visualisation systems?

iv. How can a convolutional deep-learning surrogate model and a 3D Urban Digital
Twin be developed to deliver near-real-time flood forecasts, time-to-flood
warnings and flood-aware route guidance for Nairobi?

v. How accurate, how well calibrated and how computationally efficient is the
system when evaluated on held-out storm seasons and documented Nairobi flood
events?

---

## What changed, and why

Objectives i–iii and research questions i–iii are unchanged. The changes are in
iv and v, and in the general objective.

| Proposal | Revised | Reason |
|---|---|---|
| **LSTM**-powered / LSTM-based surrogate model | **Convolutional deep-learning** surrogate (a U-Net) | Flood prediction here is a spatial task: the rainfall input has no spatial variation (a single CHIRPS value for the county), so all spatial structure comes from terrain, which a convolutional network is built for. The recurrent (ConvLSTM) model built first could not be defended: its training data contained only 6 flood-positive samples out of 703. The U-Net reaches F1 0.937 on held-out storm seasons (0.923–0.945 per season). "Deep-learning surrogate" keeps the proposal's intent without naming an architecture the evidence rejected. |
| Water **depth** prediction; RMSE, MAE, R² (proposal §3.2.4) | **Probability of flooding** (flood extent); F1, IoU, calibration error | No measured flood depths exist for Nairobi, and satellites cannot observe depth. A depth output would have been invented by a formula, not learned. Extent and its probability can be validated. |
| Model trained on **Sentinel-1 SAR** (proposal scope) | Rainfall + terrain, with SAR tested and rejected | Sentinel-1 water masks anti-correlated with rainfall (ρ = −0.74), picking up smooth dry surfaces instead of water. Sentinel-2 optical imagery detected no floods at all. Both are reported as findings (LIMITATIONS.md §4). |
| "Near-real-time flood prediction and visualization" | Adds **time-to-flood warnings** and **flood-aware route guidance** | Directly answers the proposal's problem statement: responders lack *timely, actionable* information. The system now says when flooding is expected ("moderate flooding in about 2 hours") and routes travellers around it. |
| Evaluate "under **extreme storm scenarios**" (return-period storms) | Evaluate on **held-out storm seasons and documented Nairobi floods** | Twelve years of rainfall record cannot support 25-, 50- or 100-year return-period estimates, so named return-period scenarios would be fiction. Real events are a stronger test: all 3 documented Nairobi floods since 2022 were detected in a backtest of the live system. |
| Evaluate "predictive accuracy and computational performance" | Adds **probability calibration** | The dashboard shows percentages, so they must mean what they say. Raw outputs were overconfident: 85% floods about 62% of the time. After calibration the error fell from 0.092 to 0.020. |

### Unchanged in intent, changed in tooling

These are method choices rather than objectives, but a panel may ask:

- **PyTorch** instead of TensorFlow.
- **SQLite** with PostgreSQL/PostGIS-compatible SQL instead of a PostgreSQL
  server. It runs with no installation and switches with one setting.
- Deployment on **AWS** and **real-time sensor streams** (Kafka, Redis, API
  gateway) are future work. The system runs locally, and the live data source is
  the Open-Meteo forecast API, which the proposal's scope names as "public
  meteorological APIs".
