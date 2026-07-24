# Thesis Structure — Granular Energy Forecast System for Brandenburg

*Working title:* **Predicting Regional Renewable Power Generation in Brandenburg: An Interpretable Spatio-Temporal Machine Learning Approach for Multi-Level Governance**

Target: Bachelor's thesis. Suggested length ~50–70 pages. Section weights below are a guide, not a rule.

---

## Front Matter

- Title page (title, author, matriculation number, supervisor(s), institution, date)
- Abstract / Kurzfassung (½ page each, EN + DE): problem, method, headline result (macro nMAE ~0.20 wind / ~0.18 solar, municipal solar nMAE 0.15; beats persistence ~3×), one-line contribution.
- Declaration of authorship (Eigenständigkeitserklärung)
- Table of contents, list of figures, list of tables, list of abbreviations/symbols (CF, MW, nMAE, MBE, nRMSE, PICP, MPIW, NWP, TSO, TFT, EBM, OOF, MaStR, DBSCAN, STC)

---

## 1. Introduction (~4–5 pp)

- **Motivation:** energy transition (Energiewende), rising distributed renewables, grid congestion and curtailment at distribution level. TSO reports state totals; distribution grid operators need *local* forecasts they don't have.
- **The gap:** 50Hertz publishes macro (state) actuals; there is no public generation signal at Landkreis/Gemeinde granularity.
- **Problem statement:** can macro generation be *downscaled* to municipal level 24h ahead using only public data (asset registry + weather) plus sparse local telemetry for calibration?
- **Research questions / hypotheses:**
  - RQ1: Does a Capacity-Factor (CF) target transfer across spatial scales where a Megawatt (MW) target cannot?
  - RQ2: How much does a physics prior contribute vs. pure ML?
  - RQ3: What accuracy is achievable at municipal level, and what limits it?
- **Contributions** (bullet list, 3–5 items): scale-invariant CF formulation; physics-informed feature priors; affine downscaling contract; leak-free evaluation; state-wide 18-district scaling.
- **Thesis outline** (one short paragraph per chapter).

## 2. Background & Related Work (~6–8 pp)

- **Renewable generation forecasting taxonomy:** physical (NWP-driven) vs. statistical vs. hybrid; point vs. probabilistic; horizon classes (nowcasting → day-ahead).
- **Wind power fundamentals:** power curve, cut-in/rated/cut-out, hub-height extrapolation (log/power-law profiles).
- **Solar PV fundamentals:** irradiance→power, Standard Test Conditions (STC), performance ratio, clear-sky models (Ineichen).
- **Spatial downscaling / upscaling literature:** representative-station upscaling, model-chain approaches, why municipal-scale is under-served.
- **ML models used in the field:** gradient boosting, LSTMs, Temporal Fusion Transformer (TFT); explainability (SHAP, EBM).
- **Data sources landscape:** Marktstammdatenregister (MaStR), reanalysis (ERA5 via Open-Meteo Archive) vs. NWP forecast, TSO publications.
- **Research gap statement:** existing work forecasts at aggregate or single-site level with local ground truth; this work transfers a macro-trained model down with minimal local data.

## 3. Data (~6–8 pp)

- **MaStR asset registry:** what it contains (lat/lon, gross power, hub height, rotor diameter, mounting), streaming SAX parse, status-35 (in-operation) filter, Brandenburg spatial filter. Fleet totals (wind ~9.6 GW, solar ~9.0 GW registered).
- **50Hertz macro actuals:** resolution, harmonization to UTC parquet, coverage window.
- **Weather — Open-Meteo:** Archive (reanalysis) for training vs. Forecast (NWP) for deployment; variable list (wind_speed_100m, shortwave_radiation, temperature_2m, relative_humidity_2m, surface_pressure); grid-node snapping (0.1°).
- **E.ON Energiemonitor municipal telemetry:** Königs Wusterhausen & Nauen, 15-min → 1h, UTC (document the empirical timezone verification — cross-correlation at 0–1h lag). This is the *only* municipal ground truth.
- **Capacity normalization data:** data-driven effective capacity (causal trailing p99.9), `cf_normalization.json`, the solar commissioning trend (10.8→14.3 GW) that motivates CF.
- **Data quality, gaps, and mutability rules** (raw read-only).

## 4. Methodology (~12–16 pp — the core)

### 4.1 Target Formulation: Capacity Factor
- Definition CF = generation / effective capacity; bounded, clip at 1.5.
- **Why CF, not MW** (answers RQ1): removes non-stationary capacity-growth trend; makes target scale-free so a macro-trained model transfers to a town; MW model is anchored to "big numbers" and cannot shrink.
- Causal effective-capacity estimation (trailing p99.9, `shift(1).cummax()`) — leakage-safe.

### 4.2 Physics Priors (feature engineering)
- **Wind:** `WindPowerCalculator` — power-law height extrapolation (Hellmann α=0.14), manufacturer/generic power-curve matching, per-asset summation → cluster prior.
- **Solar:** STC-normalized (shortwave/1000 × nameplate × performance ratio 0.80); pvlib clear-sky geometry.
- Priors expressed **as capacity factors** (prior_mw / nameplate_mw) → scale-free, same as target.
- Cyclical time encodings (hour/month sin/cos).
- Feature contract single source of truth (`schema.py`); never leak MW/capacity into X.

### 4.3 Geospatial Clustering
- DBSCAN (haversine, 3 km wind / 2 km utility solar), grid aggregation for distributed rooftop; capacity-weighted centroids. Purpose: reduce weather API calls while keeping micro-meteorological variation.

### 4.4 Models
- LightGBM (quantile q10/q50/q90 → 80% interval), XGBoost, EBM, Bi-LSTM, TFT. Rationale for each; why LightGBM is the production choice (accuracy/speed/explainability balance).

### 4.5 Evaluation Protocol (leak-free — critical for defense)
- Purged Expanding-Window CV: 48–72h embargo, train-past/test-future, 4 splits.
- Out-Of-Fold (OOF) honest prediction export for all figures.
- Metrics: nMAE, MBE, nRMSE, Pearson corr; probabilistic PICP + MPIW.
- Naive baselines: persistence (t−24h), climatology (hour×month).

### 4.6 Spatial Downscaling Contract
- Local = scale × (CF × nameplate × 1000) + offset.
- Affine calibration by OLS *with intercept* (drives fit-split MBE to zero, well-posed vs. degenerate pure-MBE); fit on 40% calibrate / 60% validate split; frozen to JSON.
- State-wide fallback (identity / pooled) for the 16 districts without telemetry; point-in-polygon assignment (Nominatim polygons + geopandas sjoin).

## 5. Implementation (~4–6 pp — keep lean, no install instructions)

- Pipeline architecture / data-flow diagram (stages from XML → training matrix → models → validation).
- Module map (data / features / models / validation / dashboard) — one line each.
- Reanalysis vs. forecast branch (`--mode forecast`), NWP-gap harness.
- Reproducibility & tooling (uv, config, deterministic seeds) — brief.
- Operational artifacts: quantile interval dashboard, state-wide batch forecast.

## 6. Results & Evaluation (~10–12 pp)

- **6.1 Macro model skill:** table across folds/models; LightGBM wind nMAE ~0.20 corr ~0.95, solar ~0.18 corr ~0.97. Prediction-vs-actual and error-distribution figures (OOF).
- **6.2 Baseline comparison (RQ2 partly):** ML vs. persistence/climatology — wind ~3× better; solar persistence a tougher baseline but ML still wins. Table + bar chart.
- **6.3 Feature importance / explainability:** EBM + SHAP; physics prior dominance (wind_prior_cf importance ~0.165, others <0.013). Answers "why it works."
- **6.4 Municipal downscaling results (RQ3):** KW solar nMAE 0.15 corr 0.98; KW wind 0.39 corr 0.81; Nauen wind 0.43 corr 0.84, solar 0.31 corr 0.90. Show the *pre-calibration* `nmae_k1` catastrophe (up to ~2500) to prove CF+calibration necessity.
- **6.5 Probabilistic performance:** PICP/MPIW; macro under-coverage and worse municipal coverage (0.35–0.40 vs. 0.80 target) — honest reporting.
- **6.6 NWP gap:** reanalysis vs. forecast weather degradation (operational realism).
- **6.7 State-wide scaling:** 18-district assignment sanity (partition, totals), fallback caveat (scale divergence KW 0.44 vs Nauen 0.013).

## 7. Discussion (~5–7 pp)

- **Why it works:** weather→power translator (not autoregression) + forecastable NWP inputs + scale-free CF. Tie back to RQs.
- **Limitations (be candid):** only 2 calibrated towns; capacity-accounting mismatch (registry nameplate vs. metered footprint) is the true cause of scale divergence; intervals under-cover; reanalysis-trained numbers are optimistic; heuristic turbine-curve matching.
- **Threats to validity:** telemetry coverage, timezone assumption, effective-capacity proxy undershoot (CF>1 clipping).
- **Generalizability:** method is region-agnostic — only the boundary polygon + registry filter are Brandenburg-specific.

## 8. Conclusion & Future Work (~2–3 pp)

- Restate contributions and headline numbers.
- **Future work:** more calibrated municipalities / official capacity figures; conformal prediction for calibrated intervals; NWP ensemble uncertainty; asset/feeder-level resolution; live operational deployment across all 18 Kreise.
- Closing statement on portability of the CF downscaling method.

---

## Back Matter

- References (BibTeX; cite MaStR, Open-Meteo, 50Hertz, pvlib, LightGBM/TFT papers, DBSCAN, purged-CV/embargo source).
- Appendix A: full CV metric tables (macro + municipal).
- Appendix B: feature/schema contract, config parameters, physics constants.
- Appendix C: additional figures (per-district forecasts, calibration scatter).
- Appendix D: reproducibility notes / repository map.

---

### Writing-order tip
Draft in this order, not chapter order: **4 (Method) → 6 (Results) → 3 (Data) → 2 (Background) → 7/8 → 1 → Abstract**. Method and results are the load-bearing chapters; write them while the code is fresh, backfill context last.
