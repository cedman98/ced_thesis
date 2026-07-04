# Checkpoint 06: Feature Engineering, Capacity Factor Normalization & Municipal Scraper

**Date**: 2026-06-22  
**Project**: Brandenburg Energy Forecast System (Bachelor's Thesis)  
**Cooperation**: RITS Project

---

## 1. Objectives Completed

In this phase, we constructed the core feature engineering pipeline, focusing on transforming raw physical and meteorological data into normalized, ML-ready structures. A critical component was the implementation of Capacity Factor (CF) normalization to decouple the system from the non-stationary upward trend of renewable asset commissioning. Furthermore, we developed a local measurement scraper to prepare for municipal-level validation.

### Part A: Bottom-Up Feature Pipeline & Capacity Factor Normalization
We implemented the primary feature pipeline orchestrator at `src/features/feature_pipeline.py`.
* **Capacity Factor Normalization**: To ensure temporal stationarity between 2022 and 2026, absolute generation data was normalized into a continuous $[0, 1]$ target bound using the dynamic `_effective_capacity` calculated strictly from active MaStR commissioning dates.
* **Dynamic Physical Priors**: Implemented dynamic meteorological disaggregation rules. Instead of static downscaling, regional physical priors (such as theoretically converted wind power and solar geometry constraints) dynamically weigh the macro 50Hertz generation actuals based on localized spatial meteorological conditions.
* **Temporal Encodings**: Built 15-minute weather resampling using forward filling and created cyclical time features (`hour_sin`, `hour_cos`, `month_sin`, `month_cos`) to capture diurnal and seasonal periodicity.

### Part B: Weather Ingestion Stability & Process Locking
Upgraded the Open-Meteo pipeline (`src/data/weather_ingestion.py`) for robustness during high-volume historical backfilling:
* Added process locking to avoid concurrent API collisions.
* Implemented automatic retry logic with dynamic rate-limiting to gracefully handle HTTP 429 warnings.

### Part C: Municipal Validation Data Collection
To validate the model's accuracy outside of the macro scale, we require local telemetry:
* **E.DIS Scraper**: Added `src/measurements/scraper_service.py` (and execution entrypoint `run_scraper.py`) to systematically extract municipal generation measurements from the E.DIS energiemonitor for Königs Wusterhausen and Nauen.
* **Configuration**: Added the corresponding spatial configuration targets to `conf/config.yaml`.

---

## 2. Updated Data Flow

1. Raw weather and structural data $\rightarrow$ `feature_pipeline.py`
2. Macro target generation $\rightarrow$ Capacity Factor target normalization.
3. Physical transformations $\rightarrow$ Downscaled Dynamic Regional Priors.
4. Independent continuous local data polling $\rightarrow$ `scraper_service.py`.

All processed output is now successfully staged as ML-ready tabular matrices for modeling.
