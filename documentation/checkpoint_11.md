# Checkpoint 11: Municipal Validation Dashboard Implementation

**Date**: 2026-07-02  
**Project**: Brandenburg Energy Forecast System (Bachelor's Thesis)  
**Cooperation**: RITS Project

---

## 1. Overview

In this phase, we implemented a dedicated municipal validation dashboard using Streamlit. This dashboard bridges the gap between the macro-scale training data (50Hertz level) and the municipal-level telemetry (Königs Wusterhausen and Nauen) we collect via our scraper. It provides an interactive interface for evaluating the spatial downscaling performance of our models across all 18 Brandenburg districts.

---

## 2. Key Dashboard Components & Additions

### A. Model Skill & Accuracy Panel (`render_skill()`)
- **What**: A "Model skill — macro cross-validation" section showing LightGBM's normalized Mean Absolute Error (nMAE) against the naive persistence baseline. The calculated skill score ($1 - \frac{\text{nMAE(model)}}{\text{nMAE(persistence)}}$) demonstrates that our LightGBM model performs roughly **69% better for wind** and **37% better for solar** than persistence.
- **How**: The `load_skill()` function reads `results/macro_cv_metrics.csv`, filters to the capacity factor (CF) scale, and averages the nMAE/PICP over the CV folds for each technology.
- **Honesty Note**: A critical thesis point is baked into the caption. The displayed CSV metrics are intentionally macro-scale. There are no per-district test metrics shown because the 18 districts are uncalibrated (using identity downscaling) and only the Königs Wusterhausen/Nauen pilots have local telemetry. Since every district runs the same macro model, this remains the most correct shared skill estimate.

### B. Prediction Interval Coverage Probability (PICP) Honesty
- **What**: Also inside `render_skill()`, a caption explicitly states the nominal 80% prediction interval target next to the *actual* realized coverage. 
- **Realized Coverage**: 
  - **Macro-scale**: Wind 63%, Solar 77%
  - **Municipal Pilots**: Wind 31%, Solar 28%
- **How**: `load_muni_picp()` averages the LightGBM PICP column from `results/municipal_validation_metrics.csv`.
- **Why**: This transparency surfaces the fact that the 80% confidence band under-covers at the local level, ensuring the model's interval certainty isn't oversold to examiners.

### C. Geospatial Map View (`map_figure()`)
- **What**: A Plotly choropleth showing the 24-hour total renewable energy (MWh) across all 18 Kreise, visible only in the state-aggregate view (ranging from roughly 300 to 33,712 MWh across districts).
- **How**: `get_boundaries()` loads polygons from `src/data/districts.py`, converting them to GeoJSON. District totals are matched by `properties.name` (successfully matching all 18). It utilizes the `carto-positron` basemap, removing the need for a Mapbox token or new dependencies. Per-district totals are calculated either directly from the pre-computed parquet file or accumulated during a live-forecast loop.
- **Note on Tech Debt**: A deliberate code comment notes that `go.Choroplethmapbox` is deprecated. While it currently renders fine without a token, the comment acts as a marker to upgrade to `go.Choroplethmap` if a future Plotly release drops support.

### D. Wind & Solar Stacked Totals (`stacked_figure()`)
- **What**: A combined stacked-area chart showing the median (P50) wind and solar predictions, accompanied by a total-renewable line. 
- **Reworked KPIs**: The top-line numbers now highlight **Peak renewable (P50)** and **Renewable energy (24 h)**, with wind and solar specific splits available in secondary tiles.
- **How**: Utilizes Plotly `stackgroup` (using only the median P50, as stacking P10/P90 prediction bands would be mathematically misleading). This applies to both aggregate state-views and single-district views.

---

## 3. Running and Testing the Dashboard

The dashboard is designed to run efficiently via Streamlit:

```bash
uv run streamlit run dashboard.py
```

### Execution Pathways
1. **Pre-computed Batch**: The dashboard prioritizes loading a pre-computed parquet file (`results/forecasts/brandenburg_state_forecast.parquet`). All four UI panels work seamlessly offline against this batch file.
2. **Live Fallback**: If the parquet file is missing, the dashboard falls back to live Open-Meteo NWP ingestion. In this mode, the map and aggregate views generate forecasts for all 18 districts on-the-fly (taking ~1–2 minutes).

### Regenerating the Batch File
To manually regenerate the fast-load batch parquet, execute the corresponding thesis notebook:
```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/11_brandenburg_state_forecast.ipynb
```

### Verification
The dashboard implementation has been thoroughly verified using `streamlit.testing.v1.AppTest`. Both the aggregate path (4 charts + map + skill panel) and the single-district path (3 charts + skill panel, no map) pass cleanly with zero exceptions.
