# Project Comprehensive Summary: Energy Forecasting Bachelor Thesis

## 1. Executive Summary & Core Value Proposition

### High-Level Purpose

This codebase implements a **granular renewable energy forecasting system** for the German federal state of **Brandenburg**. The central technical problem it solves is **spatial downscaling**: the 50Hertz Transmission System Operator (TSO) publishes renewable generation actuals only at the aggregate control-zone (macro) level, but grid operators, municipalities, and distribution-system operators need generation forecasts at the **Landkreis (district)** and **Gemeinde (municipality)** level. There is no publicly available, high-resolution ground-truth feed at that granularity across the state.

The system bridges this gap with a **hybrid bottom-up physical simulation + machine-learning approach**. It reconstructs a physics-based "prior" for what wind and solar *should* be producing at any location given the weather and the installed asset fleet, then learns a residual correction from the macro 50Hertz actuals, and finally transfers that learned mapping down to individual municipalities via a **scale-free capacity-factor formulation**. The end product is a **rolling 24-hour-ahead forecast** of wind (onshore) and solar (PV) generation, expressed as a median point forecast plus an 80% prediction interval, for the state aggregate and all 18 Brandenburg districts.

### Target Domain

- **Primary domain:** Spatio-temporal energy forecasting / renewable generation nowcasting and short-term forecasting.
- **Sub-domains touched:** Geospatial data engineering (point-in-polygon assignment, geographic clustering), atmospheric physics (wind-profile extrapolation, clear-sky irradiance modeling), time-series cross-validation methodology, and Explainable AI (XAI).
- **Scientific framing:** A Bachelor's thesis, so the codebase is explicitly optimized for **methodological defensibility and reproducibility** (honest out-of-fold evaluation, purged CV, naive baselines that ML must beat) rather than for production SLA/throughput.

### Key Capabilities

- **MaStR asset ingestion:** Streaming SAX-style parse of the German *Marktstammdatenregister* (national energy-unit registry) bulk XML export, spatially filtered to the Brandenburg boundary, yielding a per-asset fleet of wind turbines and solar installations with nameplate capacity, geometry, and technical metadata.
- **Geospatial clustering:** DBSCAN + grid aggregation to reduce thousands of assets to a manageable set of weather-query centroids while preserving micro-meteorological variation.
- **Weather ingestion (dual-branch):** Historical **reanalysis** weather (ERA5 via Open-Meteo Archive) for training, and live **NWP forecast** weather (Open-Meteo Forecast) for operational inference; resumable, rate-limit-aware downloads persisted per grid node.
- **Physics priors:** Manufacturer-matched wind power curves with logarithmic-profile height extrapolation; STC-normalized PV irradiance modeling with `pvlib` clear-sky geometry.
- **Capacity-factor ML fusion matrix:** A single scale-free feature matrix joining physical priors, spatial-mean weather, and cyclical time onto the 50Hertz capacity-factor targets.
- **Multi-model training:** LightGBM (quantile), XGBoost, EBM (glass-box), Bi-LSTM, and a Temporal Fusion Transformer — all trained on the same scale-free CF contract under a purged expanding-window CV.
- **Naive baselines:** Persistence (t−24h) and climatology (hour×month), computed under the identical CV so ML value can be *proven*, not assumed.
- **Cross-scale municipal validation:** Downscaling the macro models to real Gemeinde-level telemetry (E.ON Energiemonitor) with a fitted affine bias calibration on a held-out municipal split.
- **State-wide batch forecasting:** All 18 Brandenburg Kreise via point-in-polygon asset assignment and identity/pooled fallback calibration.
- **Operational dashboard:** Streamlit app serving pre-computed batch forecasts or running live NWP inference per district, with geospatial choropleth and interval bands.
- **XAI artifacts:** SHAP (TreeExplainer + DeepSHAP) and EBM/TFT native interpretation plots quantifying physical-prior vs. meteorological feature importance.
- **NWP-gap quantification:** An explicit study isolating the accuracy lost when driving the model with forecast weather instead of reanalysis actuals.

---

## 2. Architectural Overview & System Design

### Architectural Style

The system is a **layered, pipeline-oriented modular monolith** built around **file-based intermediate materialization**. It is not a service architecture; it is a batch/data-engineering pipeline where each stage reads well-defined artifacts (CSV / Parquet / JSON / GeoJSON) from disk, transforms them, and writes the next stage's artifacts. A single orchestration entrypoint (`main.py`) sequences the stages, and each stage module is independently runnable via `uv run -m ...`.

The layers, top to bottom:

1. **Data acquisition layer** (`src/data/`) — external ingestion and harmonization (MaStR registry, Open-Meteo weather, 50Hertz actuals, OSM boundaries, E.ON telemetry).
2. **Feature engineering layer** (`src/features/`) — clustering, solar geometry, the fusion matrix, the canonical schema contract, and the CV splitter.
3. **Physics layer** (`src/models/wind_physics_transformer.py`, `src/features/solar_geometry.py`) — deterministic physical simulation providing the priors.
4. **Modeling layer** (`src/models/train_*.py`) — the five ML model families.
5. **Evaluation & explanation layer** (`src/evaluation/`, `src/models/explain_*.py`) — metrics, baselines, OOF export, NWP gap, SHAP/XAI.
6. **Validation layer** (`src/validation/`) — cross-scale downscaling against municipal telemetry.
7. **Serving layer** (`src/dashboard/`, `dashboard.py`, notebook 11) — live/batch district forecasting and the operational UI.

### High-Level Component Topology

```
[MaStR XML Bulk]──parse/spatial-filter──▶ data/raw/mastr_{wind,solar}_brandenburg_raw.csv
                                                     │
[OSM Overpass boundary]───────────────────┐         ▼
                                          │  [Geospatial Clustering]──▶ {wind,solar}_clusters.csv
                                          │         │
[Open-Meteo Archive/Forecast]────────────┼─────────▼
                                          │  data/processed/weather/<cluster>.parquet + mapping_index.json
                                          │         │
[pvlib Solar Geometry]────────────────────┼─────────▼ solar_geometry/<cluster>.parquet
                                          │         │
[50Hertz CSV actuals]──harmonize──────────┼─────────▼ actual_generation_50hertz.parquet
                                          │         │
                                          ▼         ▼
                                    [feature_pipeline] ──▶ ml_training_matrix.parquet  (CF, scale-free)
                                                     │      cf_normalization.json
                                                     ▼
              [baselines_macro] ◀──── [train_{lightgbm,comparisons,ebm,bilstm,tft}] ──▶ models/*.{pkl,pth,ckpt}
                                                     │                                   results/macro_cv_metrics.csv
                                                     ▼
                            [explain_*] ──▶ results/*.png/csv (SHAP/EBM/TFT XAI)
                                                     │
                                                     ▼
   [E.ON scraper telemetry] ──▶ [municipal_validator / tft_municipal_validator]
                                    ──▶ results/<slug>/affine_calibration.json + validation parquet/csv
                                                     │
                                                     ▼
        [districts.py point-in-polygon] ──▶ [forecast_service] ──▶ dashboard.py / notebook 11 batch
```

Two structural "spines" hold the topology together and are the most important architectural facts about the system:

- **`src/features/schema.py`** — the single source of truth for the feature/target column contract. Every trainer and every validator imports `FEATURE_COLS`, `TARGET_CF`, `TARGET_MW`, `CAP_COLS`, and `CF_CLIP` from here, guaranteeing that training and inference operate on the *identical* scale-free column set and that no raw-MW/capacity column ever leaks into the model input `X`.
- **`src/validation/kw_features.py`** — the single feature-builder used at *every* downscale point (archive validation, TFT validation, live dashboard, state-wide batch). `assemble_municipal_features` is parameterized by a `load_node_weather` callback so the exact same physics runs whether weather comes from stored archive parquet or the live forecast API. This is what makes a persisted affine calibration remain valid at inference time.

### Core Design Principles

- **Scale invariance via capacity factor.** The single most important design decision: both targets and physical priors are expressed as dimensionless capacity factors (`prior_mw / fleet_nameplate_mw`, `generation_mw / effective_capacity_mw`), bounded roughly to `[0, 1.5]`. This makes the learned macro map transfer *unchanged* to a single municipality — the mechanism that makes downscaling work at all.
- **Reproducibility & honest evaluation.** Purged expanding-window CV with a mandatory temporal embargo; causal (leak-free) capacity normalization; out-of-fold prediction export instead of in-sample refit predictions; naive baselines computed under the identical CV.
- **Single-source-of-truth contracts.** `schema.py` for columns, `cf_normalization.json` for normalization constants, `config.yaml` for all tunable physics/URLs/registry — no duplicated magic numbers across trainers.
- **XAI compatibility as a hard constraint.** Named Pandas DataFrame columns are preserved end-to-end; no PCA or unlabeled dimensionality reduction before the model layer, so `shap.TreeExplainer` and the `interpret` EBM library remain usable.
- **Physics-first, ML-as-residual.** Deterministic physical simulation carries the bulk of the signal; ML corrects the residual. Root-cause physical fixes (e.g., correcting the `/1000` STC normalization) are preferred over adding model capacity.
- **Idempotent, resumable, cache-aware stages.** Each stage checks for existing outputs and skips network/compute where possible; weather ingestion and boundary fetching are resumable and rate-limit-aware.
- **Data mutability discipline.** `data/raw/` is strictly read-only; all pipeline outputs go to `data/processed/`; static references live in `data/external/`.

---

## 3. Technology Stack & Dependency Matrix

### Primary Runtime & Languages

- **Python** (managed exclusively via the **`uv`** package manager and runner — never `pip`/`python` directly).
- **YAML** for configuration; **JSON** for lightweight state/normalization/calibration artifacts; **GeoJSON** for boundaries.

### Core Frameworks & Engines

- **Machine learning / modeling:**
  - **LightGBM** — the production model, trained as a *quantile* regressor (q10/q50/q90) for prediction intervals.
  - **XGBoost** — GBDT comparative baseline.
  - **`interpret` (EBM / `ExplainableBoostingRegressor`)** — glass-box additive model.
  - **PyTorch** (`torch`, CUDA cu124 index) — the Bi-LSTM sequence model.
  - **`pytorch-forecasting` + `lightning.pytorch`** — the Temporal Fusion Transformer with `GroupNormalizer`, `TimeSeriesDataSet`, `EarlyStopping`, and `ModelCheckpoint`.
- **Physics engines:**
  - **`pvlib`** — solar position (zenith/azimuth) and Ineichen clear-sky GHI.
  - **`scipy.interpolate.interp1d`** — 1D spline interpolation of manufacturer wind power curves.
- **Geospatial:**
  - **`geopandas` / `shapely`** — spatial joins, point-in-polygon assignment, polygon geometry.
  - **`scikit-learn` DBSCAN (haversine metric)** — asset clustering.
- **Explainability:** **`shap`** (TreeExplainer for LightGBM, DeepExplainer/DeepLIFT for the Bi-LSTM); native EBM and TFT interpretation.
- **Serving/UI:** **Streamlit** dashboard with **Plotly** (`graph_objects`) choropleth and interval-band charts.

### Data Engineering & Storage Layer

There is **no database**. The storage layer is a **structured file system** of typed artifacts:

- **Parquet** — the primary columnar format for all time-series and matrices (weather per node/cluster, solar geometry, harmonized 50Hertz actuals, the ML training matrices, validation predictions, OOF predictions, batch forecasts).
- **CSV** — raw MaStR asset tables, cluster summaries, E.ON telemetry scrapes, metrics tables, power-curve lookup sheet.
- **JSON** — `mapping_index.json` (cluster→weather-node mapping), `cf_normalization.json` (canonical CF constants), per-municipality `affine_calibration.json`, ingestion state.
- **GeoJSON** — the Brandenburg state boundary and the 18 cached district polygons.
- **Model artifacts** — `models/*.pkl` (LightGBM/XGBoost/EBM + scaler sidecars), `*.pth` (Bi-LSTM state dicts + persisted scalers), `*.ckpt` (TFT Lightning checkpoints).

### Key Ecosystem Tools

- **`uv`** — the mandated package manager and script runner; locks the technical environment.
- **CUDA cu124** — `torch` is sourced from the PyTorch cu124 wheel index for GPU training of the deep models.
- **`tqdm`** — progress reporting for the streaming XML parse and long solar-geometry runs.
- **`loguru`** / **`omegaconf`** — used specifically inside the E.ON scraper service.
- **`requests`** — all HTTP ingestion (Open-Meteo, Nominatim, Overpass, E.ON).
- **External data providers:** Marktstammdatenregister bulk export, Open-Meteo (Archive + Forecast APIs), OpenStreetMap (Overpass for the state boundary, Nominatim for district polygons), 50Hertz TSO CSVs, E.ON Energiemonitor API.

---

## 4. Module-by-Module Domain Breakdown

### 4.1 Data Acquisition — `src/data/`

**`mastr_bulk_parser.py`**
- *Responsibility:* Memory-safe **streaming** parse of the massive MaStR bulk XML (`EinheitenWind.xml`, `EinheitenSolar*`), filtering assets spatially to the Brandenburg boundary and imputing missing coordinates from postal codes.
- *Core entities:* `ProgressFile` (a file-like wrapper reporting byte-level read progress to a `tqdm` bar), `run_parsing_pipeline`. Uses `xml.etree.ElementTree` iterative parsing + `shapely.geometry.Point` point-in-polygon against the GeoJSON mask.
- *Internal dynamics:* Row-by-row extraction avoids loading the whole XML into memory; keeps only status-35 ("in operation") units, so the raw CSVs *are* the active fleet (no downstream commissioning/decommission filter needed). Output is read-only after parsing.

**`mastr_downloader.py`** — fetches/unpacks the MaStR bulk export (the download side of the parser's input).

**`boundary_fetcher.py`**
- *Responsibility:* Fetch the official Brandenburg administrative polygon (OSM Relation ID **62504**, admin level 4) from the **Overpass API** and cache it as GeoJSON; also hosts the shared **`load_config`** used across the entire codebase.
- *Core entities:* `fetch_brandenburg_boundary`, `load_config`. Reconstructs the polygon from OSM ways via `linemerge`/`polygonize`/`unary_union`.

**`weather_ingestion.py`**
- *Responsibility:* Download hourly weather for each 0.1° grid node from Open-Meteo (Archive for reanalysis, Forecast for operational mode), writing compressed Parquet per node plus `mapping_index.json`.
- *Core entities:* `snap_coordinates` (rounds to 0.1° grid), `load_state`/state persistence, `run_weather_ingestion(config_path, mode=...)`. Rate-limit-aware and **resumable** via a JSON state file.

**`target_harmonization.py`**
- *Responsibility:* Parse/clean/standardize the raw 50Hertz onshore-wind and PV generation CSVs into a unified **UTC-indexed 15-minute Parquet**, resolving DST transitions and short gaps.
- *Core entities:* `interpolate_short_gaps` (interpolates NaN runs only up to `max_gap=3`, strictly preserving leading/trailing NaNs), `run_target_harmonization`.

**`districts.py`**
- *Responsibility:* The **state-wide scaling registry** — maps the 14 Landkreise + 4 kreisfreie Städte to boundary polygons and assigns every MaStR asset to a district by point-in-polygon (MaStR has no Kreis field).
- *Core entities:* `district_registry`, `fetch_district_boundaries` (Nominatim, cached, ≤1 req/s), `load_district_boundaries` (validates the full 18-district registry is present; uses `representative_point()` guaranteed-inside centroids), `_assign` (single `gpd.sjoin` `within` predicate), `load_district_fleets` (returns `{name: (wind, solar, wind_mw, solar_mw, nodes)}` shape-compatible with `municipal_fleet`), `resolve_calibration` (fallback affine params). Self-check asserts all 18 districts load and assets partition cleanly.

### 4.2 Feature Engineering — `src/features/`

**`schema.py`** — *the column contract.* `FEATURE_COLS` = wind/solar priors (CF) + temperature/pressure/humidity + hour/month sin/cos (9 scale-free features). `TARGET_CF`, `TARGET_MW`, `CAP_COLS` map technology→columns; `CF_CLIP = 1.5` is the shared clip bound (>1 is legal because the effective-capacity proxy can undershoot true nameplate). `tft_known_reals(tech)` returns the TFT decoder's known-future reals (priors split by technology + weather).

**`geospatial_clustering.py`**
- *Responsibility:* Reduce the asset fleet to weather-query centroids.
- *Strategies:* Wind → **DBSCAN** (haversine metric, `min_samples=1` so every turbine joins a cluster, radius from config ~3 km) with **capacity-weighted centroids**. Solar is **dual-track**: utility/ground-mounted → DBSCAN at a strict 2.0 km radius; distributed/rooftop → **static 0.1° grid aggregation**. Outputs `{wind,solar}_clusters.csv` with `cluster_id, centroid_lat/lon, total_capacity_mw, asset_count, track_type`.

**`solar_geometry.py`**
- *Responsibility:* Deterministic pre-computation of apparent solar position (zenith/azimuth) and **Ineichen clear-sky GHI** per solar cluster centroid at 15-min resolution over the multi-year window, using `pvlib`.
- *Core entities:* `process_single_cluster`, parallelized with a `ProcessPoolExecutor`. Output Parquet per cluster joined into the solar prior.

**`feature_pipeline.py`** — *the fusion assembler.*
- *Responsibility:* Build `ml_training_matrix.parquet` (reanalysis) or `..._forecast.parquet` (NWP).
- *Core functions:* `load_and_downsample_targets` (15-min→1h mean), `calculate_regional_wind_prior` (per-unit CF curve from a 3000 kW reference turbine × cluster nameplate — deliberately *not* passing cluster nameplate as `gross_power`, which historically under-scaled large clusters ~30×), `calculate_regional_solar_prior` (`shortwave / 1000` STC normalization × nameplate × 0.80 performance ratio — the `/1000` fixes a historical ~1000× inflation), `calculate_mean_weather_features` (spatial mean of temp/pressure/humidity), `_effective_capacity` (the **causal** trailing-365-day p99.9 robust peak, shifted one step and `cummax`-enforced monotone — replaces a leaky per-calendar-year percentile), `build_feature_matrix(mode)`.
- *Internal dynamics:* Joins priors + weather-mean onto targets (inner joins), adds cyclical time, then normalizes everything to capacity factors and persists `cf_normalization.json` (**only** on the reanalysis run, so a forecast run never clobbers the canonical constants).

**`validation_splitter.py`**
- *Responsibility:* `PurgedExpandingWindowSplitter` — expanding training window → **mandatory temporal purge gap (embargo, 48–72 h)** → fixed-duration test window (default 365 days, 4 splits). Eliminates autocorrelation leakage from atmospheric persistence.
- *Core entities:* `.split(X)` yields integer index arrays with half-open test windows (adjacent folds never share a boundary timestamp) and raises on too-short datasets or empty folds; `visualize_splits` renders an ASCII timeline of `T/G/V` bars.

### 4.3 Physics — `src/models/wind_physics_transformer.py`

- *Responsibility:* `WindPowerCalculator` translates 100 m wind speed into theoretical turbine power.
- *Core methods:* `extrapolate_wind_speed` (Hellmann power law `v = v_ref·(z/z_ref)^α`, α=0.14 from config), `match_turbine_curve` (a **multi-tier scored heuristic** matching MaStR metadata to a power-curve row by manufacturer name — via a 17-entry `mfr_map` of MaStR IDs — capacity within 15%, rotor diameter within 10%, and name-token substring, with a manufacturer-restricted-then-global search and a closest-capacity fallback), `get_interpolator` (`interp1d`, zero-filled outside [0, 35] m/s to model cut-in/cut-out), `calculate_power` (extrapolate → match → interpolate → clip to `[0, gross_power]`).
- *Verification:* Embedded `unittest.TestCase` checks zero-wind → 0 kW, rated-wind → nameplate, cut-out → 0 kW, and the height-extrapolation factor.

### 4.4 Modeling — `src/models/train_*.py`

All trainers share the same skeleton: load the CF matrix, select `FEATURE_COLS`, run the `PurgedExpandingWindowSplitter` (4 splits / 365 d / 72 h gap), persist held-out metrics to `results/macro_cv_metrics.csv` via `metrics.cv_rows`/`append_cv_metrics` at both CF and reconstructed-MW scale, then refit on the full dataset for serialization/XAI.

- **`train_lightgbm.py`** — the production model. Trains **quantile regressors** at q10/q50/q90 (`QUANTILES=(0.1,0.5,0.9)`). `quantile_model_path` keeps q50 as the canonical `lightgbm_{tech}.pkl` (so SHAP / nwp_gap / validators load it unchanged) with q10/q90 sidecars. Mild regularization (smaller trees, subsampling, L2) deliberately shrinks predictions toward the training mean to curb the extrapolation **MBE** that hurts downscaling. Reports PICP + normalized interval width.
- **`train_comparisons.py`** — XGBoost (`n_estimators=600, lr=0.03, max_depth=6, min_child_weight=5, subsample/colsample=0.8, reg_lambda=1.0`) + EBM, serialized for downstream XAI.
- **`train_ebm.py`** — standalone glass-box `ExplainableBoostingRegressor`; the CF reformulation keeps the municipal feature distribution *inside* the training range, avoiding the runaway linear-extrapolation collapse EBMs previously showed at local scale.
- **`train_bilstm.py`** — a 2-layer Bi-LSTM (`SEQ_LEN=24, HIDDEN=64, LAYERS=2`) over the scale-free CF features, with **persisted `StandardScaler`s** (feature + target) so the municipal validator can run inference; the CF target makes the scaler offset physically meaningful at any scale.
- **`train_tft.py`** — Temporal Fusion Transformer (`MAX_ENCODER_LENGTH=72`, `MAX_PREDICTION_LENGTH=24`) via `pytorch-forecasting`, with a `GroupNormalizer` on the scale-free CF (so applying the trained normalizer to a municipality no longer injects the multi-GW macro offset). Writes its own interpretation plots.

### 4.5 Evaluation & Explanation — `src/evaluation/` + `src/models/explain_*.py`

- **`metrics.py`** — nan-safe error metrics (`mae/rmse/mbe/nmae/nrmse/pearson_r/all_metrics`), interval metrics (`picp`, `mpiw_norm`), naive baselines (`persistence_pred`, `fit_climatology`/`climatology_pred`), the multi-window `rolling_metrics` harness, and CV-metric persistence (`cv_rows`, `append_cv_metrics` with keep-last de-dup on `(model, technology, fold, scale)`). Includes an `__main__` self-check asserting perfect-prediction ≈ 0 error, persistence tracking a daily cycle, climatology beating the global mean, and PICP ≈ 0.80 for a [10th,90th] normal interval.
- **`baselines_macro.py`** — persistence + climatology under the identical purged CV; "the bar ML must clear."
- **`export_predictions.py`** — honest **out-of-fold** macro predictions (re-runs the splitter, imports each trainer's exact params so they can't drift), avoiding in-sample flattery; outputs tidy `macro_oof_predictions.parquet`. Only the cheap tree/glass-box models are re-exported.
- **`nwp_gap.py`** — isolates the NWP accuracy penalty: reanalysis and forecast matrices share identical CF targets and timestamps; only the driving weather differs. Runs the best model on both over the common period → `results/nwp_gap_report.{csv,md}`.
- **`explain_models.py`** — SHAP `TreeExplainer` summary plots on the q50 LightGBM models (prior vs. meteorological importance), using exactly `FEATURE_COLS`.
- **`explain_bilstm.py`** — SHAP `DeepExplainer` (DeepLIFT) on the Bi-LSTM, sequences built from the last purged CV fold with the persisted production scalers; per-feature importance averaged over the 24-step window.

### 4.6 Cross-Scale Validation — `src/validation/`

- **`kw_features.py`** — the shared municipal feature builder (detailed in §2). Provides `municipal_fleet` (assets within a radius, nameplate MW, snapped weather nodes), `assemble_municipal_features` (callback-driven scale-free `X`), `build_municipal_features` (archive-backed), `load_edis_telemetry` (E.ON scrape, **UTC — no timezone conversion**, verified empirically), and the **affine calibration primitives** `calibrate_affine`/`apply_affine`.
- **`municipal_validator.py`** — runs LightGBM (interval), XGBoost, EBM, Bi-LSTM, and a persistence baseline for each configured municipality. Splits the telemetry overlap **40% calibrate / 60% validate** (no hardcoded window), fits per-`(model, tech)` affine params, reports pre-calibration (`nmae_k1`, `mbe_k1`) and post-calibration metrics plus PICP/width for the interval, and freezes calibration to `results/<slug>/affine_calibration.json`.
- **`tft_municipal_validator.py`** — the same downscaling contract for the TFT checkpoints, appending its rows into the shared KW metrics table and calibration file.
- **`extend_kw_weather.py`** — back-fills each municipality's 0.1° weather nodes from their current max timestamp up to the latest telemetry hour, so validation gets multi-week overlap instead of a few training-era days; `--all` extends every archived node (nightly cron). Macro training is unaffected because `feature_pipeline` inner-joins on earlier-ending targets.

### 4.7 Serving — `src/dashboard/` + `dashboard.py`

- **`forecast_service.py`** — live/batch inference backend. `fetch_forecast_weather` (one batched Open-Meteo Forecast call for all nodes, exponential backoff on 429/5xx), `_downscale_interval` (quantile CF → sorted, calibrated MW band), `forecast_from_fleet` (core shared by municipal-radius and district-polygon paths, filters to future hours ≥ now, horizon 24 h), `forecast_municipality` (uses the pilot's persisted `lightgbm_{tech}` calibration), `forecast_district` (uses the fallback resolver). Network-free `__main__` self-check asserts the band is ordered and MW ≥ 0.
- **`dashboard.py`** — Streamlit UI over all 18 Kreise + state aggregate; serves the pre-computed batch Parquet when present, else runs the live 24 h pipeline for the selected district; Plotly choropleth + P10–P90 interval band.

### 4.8 Measurements — `src/measurements/scraper_service.py`

- `ScraperService` polls the **E.ON Energiemonitor** API (`omegaconf` config, `loguru` logging) at 15-min resolution for municipal generation (Königs Wusterhausen, Nauen), appending a 13-column CSV (`timestamp, biomasse, photovoltaik, windkraft, ...`) and managing/migrating the CSV header. This is the ground-truth telemetry feed for cross-scale validation.

### 4.9 Notebooks — `notebooks/`

EDA/XAI/reporting only (numbered `01_`–`12_`): asset mapping, clustered mapping, ML-matrix EDA, model-performance comparison, municipal validation dashboard, macro performance, spatial downscaling, XAI & features, physics-vs-ML residuals, municipal scope comparison, the **state-wide mass-forecasting engine (11)** orchestrating batch inference over all 18 Kreise via `src/` utilities, and the calibration/capacity audit (12). Pipeline logic stays in `src/`, never in notebooks.

---

## 5. Data Architecture, Schemas & State Flow

### Core Data Models & Entities

- **MaStR asset row** — per-unit registry record. Wind: `unit_mastr_number, lat/lon, commissioning/decommission dates, gross_power (kW), net_nominal_power, manufacturer, technology, type_designation, hub_height, rotor_diameter`. Solar: adds `is_ground_mounted` (utility vs. rooftop), `azimuth`, `tilt`, `has_battery_storage`.
- **Cluster summary** — `cluster_id, centroid_lat/lon, total_capacity_mw, asset_count, track_type` (utility/distributed for solar).
- **Weather node Parquet** — UTC-indexed hourly rows of `temperature_2m, relative_humidity_2m, surface_pressure, cloud_cover, wind_speed_10m/100m, wind_direction_10m/100m, shortwave_radiation, direct_radiation, diffuse_radiation, direct_normal_irradiance`.
- **50Hertz actuals** — UTC-indexed generation: `wind_onshore_mw_50hz`, `solar_pv_mw_50hz`.
- **ML training matrix** — UTC hourly rows carrying: the 9 `FEATURE_COLS`; the CF targets `wind_cf_50hz`/`solar_cf_50hz`; the raw-MW targets and effective-capacity series (`*_eff_capacity_mw`) kept for MW reconstruction but never fed to a model.
- **`cf_normalization.json`** — `{wind,solar}.fleet_capacity_mw`, `eff_capacity_mw_by_year`, `solar_performance_ratio`, `ref_turbine_kw`.
- **`affine_calibration.json`** — per-`(model, technology)` `{scale, offset, cap_mw}` for each pilot municipality.
- **E.ON telemetry** — UTC 15-min `photovoltaik`, `windkraft` (kW) plus household/industry breakdown columns.
- **District boundary** — GeoDataFrame `name, kind (landkreis|stadt), geometry, lat, lon`.

### End-to-End Data Pipeline / Lifecycle

**Training branch (reanalysis):** boundary fetch → MaStR streaming parse (spatial filter) → geospatial clustering → per-cluster archive weather ingestion + `pvlib` solar geometry → 50Hertz harmonization → `feature_pipeline` fusion into the CF matrix (+ `cf_normalization.json`) → macro baselines → five model families (each with purged-CV metrics) → SHAP/XAI → municipal cross-scale validation (fits + freezes affine calibration).

**Downscaling transformation (the heart of the system):**
`municipal_generation = apply_affine(CF_pred × local_nameplate_mw × 1000, scale, offset)` where `CF_pred` comes from scale-free local features and `(scale, offset)` is fit by **OLS with intercept** on the held-out municipal calibration split. The intercept drives the fit-split MBE to identically zero (absorbing systematic municipal bias) while OLS keeps the fit well-posed (a pure-MBE objective is degenerate); it falls back to offset-only when the base is constant or the slope is non-positive, and the result is floored at zero (generation is non-negative).

**Operational branch (forecast):** `main.py --mode forecast` refreshes NWP weather and rebuilds the forecast matrix (identical target/timestamps, only weather differs). `forecast_service` then, per district/municipality, batch-fetches live forecast weather at the nodes, builds the identical scale-free features, runs the LightGBM quantile models → CF, and downscales with the persisted (pilot) or fallback (district) calibration into a 24 h MW median + 80% band. The dashboard/notebook 11 render or batch-persist these forecasts.

### External Integration Points

- **Marktstammdatenregister** bulk XML export (asset registry).
- **Open-Meteo Archive API** (reanalysis/ERA5 training weather) and **Forecast API** (operational NWP).
- **OpenStreetMap Overpass API** (Brandenburg state boundary, relation 62504) and **Nominatim** (18 district polygons; ≤1 req/s, cached).
- **50Hertz TSO** generation CSVs (macro targets).
- **E.ON Energiemonitor API** (municipal ground-truth telemetry).

---

## 6. Algorithmic, Methodological & Technical Specs

### Core Algorithms & Mathematical Foundations

- **Capacity-factor normalization.** `CF_t = y_t / C_t ∈ [0, 1.5]`. `C_t` is a **causal** trailing-365-day p99.9 robust peak, `.shift(1)` (strictly-before-t) and `.cummax()` (physically monotone fleet) — deliberately leak-free inside CV test folds, unlike the earlier per-calendar-year percentile.
- **Wind physics.** Hellmann power-law height extrapolation `v_hub = v_100m·(z/100)^0.14`; manufacturer power curve matched by a weighted multi-criterion score and interpolated with `interp1d` (cut-in/cut-out via zero fill outside [0,35] m/s).
- **Solar physics.** STC-normalized DC capacity factor `(shortwave/1000)·nameplate·PR(0.80)`, plus deterministic `pvlib` solar position and Ineichen clear-sky GHI.
- **Clustering.** DBSCAN with haversine metric and `eps = radius_km / EARTH_RADIUS_KM`, `min_samples=1`, capacity-weighted centroids; static 0.1° grid aggregation for distributed rooftop PV.
- **Quantile regression.** LightGBM pinball loss at q10/q50/q90; quantiles are clipped to `CF_CLIP` and sorted so they never cross; PICP + normalized MPIW report interval calibration.
- **Purged expanding-window CV** with a 48–72 h embargo — the methodological guard against atmospheric-persistence leakage.
- **Affine downscaling calibration** — degree-1 OLS `Local = scale·base + offset` fit on a held-out municipal split (§5).
- **Point-in-polygon assignment** — `geopandas.sjoin` `within` predicate assigns each asset to exactly one district (a spatial partition).

### State Management & Concurrency

- **Sequential, file-materialized state.** The pipeline is orchestrated sequentially; state lives in on-disk artifacts, and each stage is idempotent (skips if outputs exist). Weather ingestion persists a JSON state file for **resumable** downloads.
- **Concurrency is localized:** `ProcessPoolExecutor` parallelizes per-cluster solar geometry; GPU (CUDA cu124) is used only for Bi-LSTM/TFT training. There is no shared mutable in-memory state and no multi-threaded request path in the core pipeline.
- **Network politeness/batching:** Nominatim calls sleep ≥1.1 s; Open-Meteo forecast calls are batched across nodes with exponential backoff and an optional inter-district pause.

### Error Handling & Resilience Patterns

- **Graceful degradation over hard failure** in ingestion loops: missing weather nodes, unmatchable turbines, or absent BiLSTM artifacts log a warning and skip rather than abort the batch.
- **Explicit precondition guards:** forecast mode requires reanalysis outputs (raises a clear `FileNotFoundError`); the district loader validates the full 18-registry (a partial fetch can't silently drop a district); the splitter raises on too-short data or empty folds.
- **Numerical safety everywhere:** nan-safe metric cleaning, `_scale` guarding divide-by-zero, quantile sorting, `CF_CLIP` bounding, physical floors at zero, and calibration fallbacks for degenerate/constant bases.
- **Embedded self-checks:** `metrics.py`, `kw_features.py`, `forecast_service.py`, `districts.py`, and `wind_physics_transformer.py` each carry runnable `__main__`/`unittest` assertions that fail loudly if the core logic regresses.

---

## 7. Technical Debt, Patterns & Architectural Decisions (ADRs)

### Design Patterns Observed

- **Single-source-of-truth contract (`schema.py`)** — every model/validator imports the same `FEATURE_COLS`/`CF_CLIP`, preventing train/inference skew and MW/capacity leakage into `X`.
- **Strategy via callback injection** — `assemble_municipal_features(..., load_node_weather)` decouples feature construction from the weather *source* (archive parquet vs. live API), so one code path serves validation, dashboard, and batch.
- **Template-method trainers** — all `train_*.py` follow the same load → purged-CV → persist-metrics → refit-and-serialize skeleton.
- **Frozen-artifact calibration** — fit once on a held-out split, persist to JSON, re-apply verbatim at inference (`calibrate_affine`/`apply_affine`), with keep-last merge so tree and TFT validators write disjoint keys into the same file.
- **Registry + fallback** — the district registry drives a spatial partition, with an `identity`/`pooled` calibration fallback for districts lacking telemetry.
- **Idempotent skip-if-exists stages** for expensive network/compute steps.

### Implicit Constraints & Assumptions

- **Geographic scope is Brandenburg only** — boundary (relation 62504), the 18-district registry, and the 50Hertz control zone are hard scope boundaries.
- **The whole edifice rests on capacity-factor scale invariance** — if a feature or target ever became scale-dependent (MW leaked into `X`), downscaling would silently break; this is why `schema.py` and the `CF_CLIP` bound are enforced identically across trainers and validators.
- **E.ON telemetry is UTC** (empirically verified) — a timezone conversion here would silently misalign features and destroy municipal metrics.
- **Only two municipalities (Königs Wusterhausen, Nauen) have ground-truth telemetry**, so 16 of 18 districts run on an **unvalidated identity/pooled fallback** — the offset term is deliberately forced to 0 for districts because an absolute-kW offset fit to a tiny Gemeinde fleet does not transfer to a Landkreis.
- **Optimized for reproducible offline batch evaluation**, not low-latency serving — memory-safe streaming parse and file materialization over in-memory speed; single-machine execution.
- **`data/raw/` is immutable**; regenerating requires re-parsing from the MaStR export.

### Identified Critical Vectors (areas needing care)

- **The NWP gap** — models are trained on reanalysis actuals but operate on forecast weather; the `nwp_gap.py` study exists precisely because this is the dominant operational error source. Interpret live-dashboard accuracy through that report.
- **Prediction-interval under-coverage at municipal scale** — the LightGBM 80% interval is calibrated at macro scale and tends to under-cover municipally (PICP < 0.80); the affine `scale` widens/narrows it but does not re-calibrate coverage.
- **Fallback calibration divergence** — the two pilot `scale` values diverge sharply (e.g., wind ≈ 0.43 vs. 0.012), so the `pooled` mode is a sensitivity knob, *not* a validated calibration; the default `identity` mode trusts the physics prior.
- **Turbine curve matching is heuristic** — the multi-tier scored match with a closest-capacity fallback can mis-assign an obscure turbine model, biasing the wind prior for that cluster.
- **Effective-capacity proxy vs. official figures** — `C_t` is a data-driven p99.9 peak; it can undershoot true nameplate (hence `CF_CLIP=1.5`) and should be overridden with official 50Hertz installed-capacity figures via `cf_normalization.json` when available.
- **Tight coupling through `kw_features.py`** — it is intentionally the single downscaling spine, which is a strength for consistency but means any change to the feature contract there propagates to validation, dashboard, and state-wide batch simultaneously.
