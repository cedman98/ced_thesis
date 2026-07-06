# Brandenburg Energy Forecast System

A Bachelor's Thesis project (RITS cooperation) that produces 24-hour-ahead
forecasts of wind and solar generation for the German state of Brandenburg,
downscaled from 50Hertz TSO macro-level actuals down to individual
Landkreis/Gemeinde (district/municipality) level. The approach combines a
bottom-up physical simulation (wind power curves, log wind profile, pvlib
solar geometry) with ML models (LightGBM, XGBoost, EBM, BiLSTM, TFT) trained
on Capacity Factor rather than raw MW, so the learned macro-scale relationship
transfers unchanged to a single municipality via a per-site affine
calibration.

See `CLAUDE.md` for the full architecture reference and `documentation/` for
dated checkpoint write-ups tracking thesis progress.

## Setup

Requires Python >=3.12.3 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

All commands below are run with `uv run` — never invoke `pip` or `python`
directly, `uv` resolves the project's locked environment (`uv.lock`).

## Pipeline

### 1. Data ingestion + training (reanalysis / ERA5 actuals)

```bash
uv run main.py
```

Runs the full pipeline: MaStR asset parsing → geospatial clustering →
Open-Meteo historical weather → solar geometry → 50Hertz target
harmonization → feature matrix → model training.

### 2. Operational forecast (NWP branch)

```bash
uv run main.py --mode forecast
```

Refreshes Open-Meteo forecast weather and builds
`data/processed/ml_training_matrix_forecast.parquet` for live inference.

### 3. Individual pipeline stages

```bash
uv run -m src.features.feature_pipeline
uv run -m src.models.train_lightgbm
uv run -m src.models.train_tft
uv run -m src.models.train_ebm
uv run -m src.models.train_comparisons
uv run -m src.models.train_bilstm
```

### 4. Explainability (XAI)

```bash
uv run -m src.models.explain_models
uv run -m src.models.explain_bilstm
```

TFT interpretation plots are written by `train_tft.py` itself.

### 5. Evaluation

```bash
# Naive baselines (persistence + climatology) -> results/macro_cv_metrics.csv
uv run -m src.evaluation.baselines_macro

# Reanalysis-vs-forecast weather gap -> results/nwp_gap_report.{csv,md}
uv run -m src.evaluation.nwp_gap
```

### 6. Municipal (cross-scale) validation

```bash
uv run -m src.validation.municipal_validator
uv run -m src.validation.tft_municipal_validator
```

Validates trained macro models against real E.DIS/E.ON municipal telemetry
(Königs Wusterhausen, Nauen) to test the spatial downscaling contract.

### 7. Weather backfill / nightly refresh

```bash
# One-off initial backfill
uv run -m src.data.weather_ingestion

# Extend every configured municipality's weather (multi-week validation)
uv run -m src.validation.extend_kw_weather

# Extend every archived grid node through yesterday (what run_pipeline.sh cron runs nightly)
uv run -m src.validation.extend_kw_weather --all
```

### 8. Real-time municipal generation scraper

```bash
uv run run_scraper.py
```

Polls the E.ON Energiemonitor API for real-time municipal generation data at
15-minute resolution. Note: the E.ON scrape is UTC — do not timezone-convert it.

### 9. District boundaries

```bash
uv run -m src.data.districts
```

Fetches and caches the 18 Brandenburg district boundaries and assigns MaStR
assets to each.

### 10. State-wide batch forecast

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/11_brandenburg_state_forecast.ipynb
```

Runs inference for all 18 Brandenburg Kreise -> `results/forecasts/*.parquet`.

## Dashboard

```bash
uv run streamlit run dashboard.py
```

Operational forecast dashboard covering the state aggregate and all 18
districts. Serves the pre-computed batch forecast
(`results/forecasts/brandenburg_state_forecast.parquet`, from notebook 11)
when present; otherwise falls back to a live 24h NWP pipeline run for the
selected district. Shows median (P50) with an 80% (P10–P90) quantile band.

## Repository layout

```
conf/            Single config file (MaStR URLs, Open-Meteo settings, physics constants)
data/
  raw/           Read-only source data (MaStR bulk XML, 50Hertz CSVs)
  processed/     Pipeline outputs (weather parquet, clusters, ml_training_matrix.parquet)
  external/      Static reference files (Brandenburg boundary GeoJSON, postal codes)
documentation/   Dated checkpoint write-ups tracking thesis progress
models/          Trained model artifacts (.pkl / .pth / .ckpt)
notebooks/       EDA + XAI visualization only (numbered 01-12); no pipeline logic
results/         Metrics, figures, per-municipality validation output, batch forecasts
src/
  data/          Asset parsing, weather ingestion, target harmonization, district boundaries
  features/      Geospatial clustering, solar geometry, feature matrix assembly, CV splitter
  models/        Physical wind/solar priors + all ML trainers and explainers
  evaluation/    Metrics, macro baselines, NWP gap analysis
  validation/    Municipal cross-scale validation, KW weather extension
  measurements/  E.ON real-time scraper
  dashboard/     Forecast service backing dashboard.py
dashboard.py     Streamlit operational dashboard entry point
main.py          Pipeline entry point (reanalysis mode / --mode forecast)
run_pipeline.sh  Nightly cron: weather extension for all archived grid nodes
run_scraper.py   E.ON scraper entry point
```

## Key concepts

- **Capacity Factor target**: models predict `CF = generation / effective_capacity`
  (bounded [0,1]) instead of raw MW, so the macro-trained relationship
  transfers to any municipality unchanged.
- **Spatial downscaling contract**: `municipal_mw = CF_pred × local_nameplate_mw × scale + offset`,
  where `(scale, offset)` is a per-(municipality, model, technology) affine
  calibration fit on held-out municipal telemetry
  (`results/<slug>/affine_calibration.json`).
- **Purged expanding-window CV**: 48-72h temporal embargo between train/test
  to eliminate autocorrelation leakage (`src/features/validation_splitter.py`).

See `CLAUDE.md` for full details on each module and the data flow diagram.
