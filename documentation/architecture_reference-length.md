# Architecture — Reference

Compact companion to `architecture_defense-grade.md`. Same coverage — every module under `src/`,
the root entrypoints, and the on-disk contracts — one paragraph plus rationale bullets per file.
Use this to look something up; use the defense-grade doc to rehearse.

---

## 1. The idea in one paragraph

50Hertz publishes generation for its whole control zone; nobody publishes it for a Brandenburg
Gemeinde, and only two towns have public telemetry. So the learning problem is made
**dimensionless**: a bottom-up physical simulation of the registered fleet yields a capacity-factor
prior, and the model predicts the 50Hertz **capacity factor** from that prior plus weather and
cyclical time. The learned map has no units in it, so it applies unchanged to a single
municipality — which needs only a local nameplate and a two-parameter affine correction fitted on a
held-out slice of the little telemetry that exists.

```
MaStR XML ─► parse+mask ─► clusters ─► weather nodes (Open-Meteo) ─┐
50Hertz CSV ─► harmonise ─────────────────────────────────────────┤
                                                                   ├─► ml_training_matrix.parquet
physics (power curves, Hellmann, STC×PR) ─► priors as CF ──────────┘        (9 features, CF target)
                                    │
                                    ├─► trainers (LightGBM q10/50/90, XGB, EBM, BiLSTM, TFT)
                                    │      all under PurgedExpandingWindowSplitter (4×365 d, 72 h embargo)
                                    ├─► evaluation (baselines, OOF, significance, ablation, conformal, NWP gap)
                                    └─► validation (municipal affine contract, LOMO transfer)
                                                    │
                                            forecast_service ─► dashboard.py (18 Kreise)
```

---

## 2. The seven decisions

| # | Decision | Why | Rejected |
|---|---|---|---|
| **D1** | Target is capacity factor, clipped `[0, 1.5]` | Removes the commissioning trend; makes the map scale-free → downscaling possible (RQ1) | Raw MW (does not transfer — this was the original failure); per-municipality models (no local labels); dividing macro MW by a hardcoded zone capacity |
| **D2** | `C_t` = causal trailing 365-day p99.9, `shift(1)`, `cummax` | Per-calendar-year percentile leaked future data into test folds via the target definition itself | Official capacity figures (unavailable; override hook exists); MaStR nameplate (wrong scope); rolling max (one spike redefines capacity) |
| **D3** | Physics enters as a **feature** (bottom-up per-asset prior) | Learner gets power-curve structure and real fleet composition for free; corrects wake/curtailment/NWP bias on top | Residual learning (anchors the learner to prior error); weather-only ML (costs +189% wind nMAE); per-asset POA transposition (MaStR azimuth/tilt are coarse bins, mostly missing) |
| **D4** | Purged expanding-window CV, 4 folds, 365-day test, 72 h embargo | K-fold trains on the future; synoptic weather autocorrelates for days; expanding matches real retraining | Standard k-fold; sliding window (discards data, confounds fold with train size); no embargo |
| **D5** | Frozen per-municipality affine `scale·(CF·nameplate)+offset`, OLS **with intercept**, 40/60 split | Intercept makes fit-split MBE identically zero (algebra, not luck); frozen so serving needs no telemetry | Scale-only (can't zero a constant offset); pure-MBE objective (degenerate: scale 0 also works); per-hour calibration (overfits 855 h); hardcoded date window |
| **D6** | LightGBM quantiles (q50 = point forecast) + width-normalised split conformal | Pinball@0.5 = L1, robust to fat-tailed ramp errors; affine rescales interval width without re-deriving it, so coverage collapses and must be repaired | L2 mean regressor; additive CQR (inflates zero-width PV nights); no interval |
| **D7** | `src/features/schema.py` is the single column contract | Macro and municipal feature construction must be byte-identical or the frozen calibration is meaningless | Per-module column lists |

---

## 3. Module reference

### `src/data/` — acquisition and harmonisation

**`mastr_downloader.py`** — streams the MaStR bulk ZIP, extracts only the wind/solar XMLs, fetches
the boundary and postal-code lookup.
- Stream to disk in 1 MB chunks, selective member extraction, temp ZIP purged in `finally`.
- Every download is skip-if-exists — re-running the pipeline must not re-pull gigabytes.
- *Known duplication:* its `download_brandenburg_boundary` duplicates `boundary_fetcher`; `main.py`
  uses the latter.

**`mastr_bulk_parser.py`** — `ET.iterparse` streaming parser → `data/raw/mastr_*_raw.csv`
(4,147 wind / 174,336 solar units).
- `elem.clear()` **and** `root.clear()` per record — without clearing the root, memory still grows.
- Two-stage filter: cheap attribute pre-filter (`Bundesland`, status 35 = operating), then true
  `within(polygon)` mask. The polygon is the authority; the admin field is unreliable at borders.
- Status filtering happens once here, so no downstream consumer re-filters — `districts.py` relies
  on this invariant.
- Coordinate imputation ladder: real → PLZ centroid → state centroid, so a unit's *capacity* is
  never lost from the fleet even when its location is unknown.
- Catalogue-code maps for azimuth/tilt (bin midpoints); parsed but unused by the current solar prior.

**`boundary_fetcher.py`** — Overpass relation 62504 → cached GeoJSON. Also hosts `load_config()`,
imported across the codebase.
- Reconstructs the polygon from raw ways (`linemerge → polygonize → unary_union`, then
  `outer.difference(inner)` for enclaves) rather than trusting a prebuilt geometry.
- Cached once: a boundary that changed between runs would silently change the asset set.

**`target_harmonization.py`** — 50Hertz CSVs → `actual_generation_50hertz.parquet` (UTC, 15-min).
- `tz_localize('Europe/Berlin', ambiguous='infer') → UTC`. Everything downstream is UTC-only;
  timezone is a display concern handled once, in `dashboard.py`.
- Handles the real-world pathology: UTF-16-LE, 4 metadata rows, `;` separator, comma decimals,
  `n.v.`-family sentinels.
- Interpolates gaps of ≤ 3 steps (45 min = a telemetry hiccup); longer gaps are real outages and are
  left NaN rather than fabricated.

**`weather_ingestion.py`** — Open-Meteo → 430 grid-node parquets + `mapping_index.json`.
- Snap centroids to a 0.1° grid: 1,093 clusters → 430 unique nodes (~60% fewer calls) with no
  physical loss, since the API's own resolution is coarser.
- Resumable `.ingestion_state.json` (atomic write) + PID `.lock` validated with `os.kill(pid, 0)`,
  so a crashed run's stale lock is cleared rather than blocking forever.
- **Quota 429s sleep until the window resets and retry indefinitely**; other 429s and network errors
  get bounded exponential backoff. A 4-year × 430-node backfill will exhaust the free quota many
  times and must not abort.
- Two isolated modes: `reanalysis` (ERA5 archive) and `forecast` (**historical**-forecast archive —
  archived NWP runs, so timestamps align with the target). This is the control for the NWP-gap study.
- Batches 5 coordinates per request, 2 s spacing.

**`districts.py`** — 18 Kreise: polygons, asset assignment, fallback calibration.
- Nominatim, not Overpass (Overpass times out on named-area queries); cached GeoJSON, 1.1 s between
  calls per usage policy.
- One `gpd.sjoin` point-in-polygon for the whole fleet — MaStR has no Kreis field.
- `representative_point()` not `centroid` (a centroid may fall outside a concave polygon).
- Raises if any of the 18 is missing from the cache, so a partial fetch can't silently drop a district.
- Returns fleets in the **same tuple shape** as `municipal_fleet`, so
  `assemble_municipal_features` is reused verbatim — identical physics, no parallel code path.
- Fallback calibration defaults to **identity** (`scale=1, offset=0`). `pooled` mode exists as a
  sensitivity knob and is documented as unvalidated (the two pilot wind scales differ 30×). The
  offset is absolute kW fitted to a Gemeinde and is always forced to 0 in a fallback.

### `src/features/` — the feature contract

**`schema.py`** (41 lines) — `FEATURE_COLS` (2 priors + temp/pressure/humidity + 4 cyclical terms),
`TARGET_CF`, `TARGET_MW`, `CAP_COLS`, `CF_CLIP=1.5`, `tft_known_reals()`.
- The invariant it enforces: macro and municipal feature construction cannot drift apart, and no
  MW/capacity column may ever enter `X`.
- `sin`/`cos` time encodings so hour 23 and hour 0 are adjacent.
- Wind speed and irradiance are deliberately **not** features — the physics already consumed them.

**`geospatial_clustering.py`** — 211 wind + 882 solar clusters from 178k assets.
- DBSCAN with haversine metric, `eps = radius_km / R_earth`; `min_samples=1` so no asset (and hence
  no capacity) is discarded as noise.
- Capacity-weighted centroids — the centroid must represent where the power is.
- **Dual-track solar:** ground-mounted → DBSCAN at 2 km (455 clusters, real spatial clusters);
  rooftop → static 0.1° grid (427 cells). 174k rooftop units are a continuous density field; the
  grid also aligns exactly with weather-node snapping.
- 3 km for wind: turbines within 3 km share a weather regime at this resolution.

**`solar_geometry.py`** — pvlib solar position + Ineichen clear-sky GHI per cluster, 15-min,
`ProcessPoolExecutor`.
- Deterministic (pure function of lat/lon/time), so precompute-and-cache turns repeated work into a
  join. Process-based parallelism because it is CPU-bound NumPy.
- Worker exceptions are captured and returned, never raised into the pool.
- *Honest note:* the output is joined but the flat `GHI/1000 × PR` prior does not consume
  `ghi_clearsky` — precomputed infrastructure the final feature set does not use.

**`feature_pipeline.py`** — builds `ml_training_matrix.parquet` (37,967 h × 17 cols,
2022-01-31 → 2026-05-31).
- Targets downsampled to hourly by `mean` (weather is hourly; mean because generation is a power
  average).
- Inner joins with row counts logged after each — a timezone mismatch shows up instantly as zero rows.
- Wind prior uses a **per-unit** curve from a 3 MW reference turbine scaled by cluster nameplate.
  Passing cluster nameplate as `gross_power` matched a single ~10 MW curve and capped a 366 MW
  cluster at 10 MW (~30× under-scaled).
- Solar prior is `GHI/1000 × cap × PR(0.80)`. Omitting the `/1000` inflated the prior ~1000×.
- `_effective_capacity()` = causal trailing p99.9 + `shift(1)` + `cummax` (**D2**).
- Mode switch shares all code except the weather directory — the two matrices differ *only* in
  weather, which is the NWP-gap control.
- Only the reanalysis run writes `cf_normalization.json`, so a forecast run can't clobber the
  constants the validators depend on.

**`validation_splitter.py`** — `PurgedExpandingWindowSplitter` (**D4**).
- Half-open test window so adjacent folds never share a boundary timestamp.
- Raises rather than silently shrinking folds when data is too short.
- `visualize_splits()` prints per-fold boundaries, counts, an overlap check and an ASCII timeline →
  `results/cv_split_table.{csv,md}`. The protocol is evidenced, not asserted.

### `src/models/` — physics and learners

**`wind_physics_transformer.py`** — hub-height extrapolation + power-curve matching.
- Hellmann `v_hub = v_100 · (z/100)^α`, α = 0.14 from config. Extrapolating **from 100 m** (not
  10 m) keeps the correction small (`(150/100)^0.14 = 1.058`), so a mis-specified α barely matters.
- Scored fuzzy matcher (manufacturer +10, capacity-within-15% graded +5, rotor-within-10% graded +5,
  name containment +8), manufacturer-filtered pool first, widening if the best score is poor,
  nearest-capacity fallback with a warning. Exact matching would drop most of the fleet; one generic
  curve would erase the technology mix.
- `interp1d(bounds_error=False, fill_value=0.0)` — zero below cut-in and above cut-out, which is
  physically correct at both ends. Hard clip to `[0, nameplate]`.
- The only module with a formal `unittest` class (zero wind, rated wind, cut-out, extrapolation
  arithmetic) — it earns it because a wrong power curve produces a plausible-looking prior.

**`train_lightgbm.py`** — the production model. Three quantile fits per technology under purged CV.
- q50 **is** the point forecast (L1-optimal, robust to fat tails); one objective family produces
  both point and interval.
- `quantile_model_path()` keeps q50 at the canonical filename, so SHAP/NWP-gap/validators load it
  unchanged and never learn about quantiles.
- Deliberately regularised (`max_depth 7`, `min_child_samples 60`, subsampling, `reg_lambda 1.0`)
  **for downscaling, not macro accuracy** — shrinking toward the training mean reduces the
  systematic bias the affine step must absorb when the municipal feature distribution sits off the
  training manifold.
- Per-row sort of the three quantiles prevents crossing.
- PICP and normalised MPIW always reported together (coverage without sharpness is meaningless).

**`train_comparisons.py` / `train_ebm.py`** — XGBoost (second GBDT family, shows the result is not a
LightGBM artifact) and EBM (glass-box GA²M for RQ2, at defaults — tuning a glass-box to match a
black box makes it useless as a reference). Identical splitter, target and clipping as every other
trainer. `train_ebm.py` also exports global importances. Both write `ebm_{tech}.pkl`; last run wins.

**`train_bilstm.py`** — 2-layer BiLSTM, 24-step window, early stopping.
- `StandardScaler` on features and target, **persisted** — municipal inference is impossible without
  them. Under the CF target the scaler offset is physically meaningful at any scale; under the old
  MW target it injected a multi-GW offset that made local predictions anti-correlate.
- Early-stopping validation from the last 10% of the **train** window (it previously used the test
  fold, leaking test info into epoch selection).
- Seeded per `(technology, fold)`. Unseeded, BiLSTM solar moved 0.2751 → 0.3055 — larger than
  several between-model gaps. `seed` is exposed so the spread can be measured.
- Target alignment: the dataset emits the target at `idx + SEQ_LEN`, so eval slices `[SEQ_LEN:]`.
- `feature_cols` / `model_name` / `out_csv` / `save_production` exist as an ablation safety
  interlock, not as generality.

**`train_tft.py`** — Temporal Fusion Transformer, 72 h encoder / 24 h decoder, `GroupNormalizer`.
- Explicit `ModelCheckpoint(monitor='val_loss')` — Lightning's default saves the *last* epoch, which
  early stopping leaves 3 past the best; without this the evaluated model isn't the selected one.
- Fold windows extended backwards by `MAX_ENCODER_LENGTH` so every prediction has legal context;
  the 72 h purge gap equals the encoder length, so context never reaches into training data.
- OOF export takes **only the h=24 step** — averaging the 24 overlapping decoder windows would be
  free ensembling the single-shot models don't get. This is why its per-point nMAE differs slightly
  from the CV table (which averages horizons 1–24).
- Seeded per `(technology, fold)`; unseeded it moved 5–12% between runs.
- Writes its own attention/variable-importance figures to `documentation/`.

**`explain_models.py` / `explain_bilstm.py`** — TreeSHAP on the production q50 models; DeepSHAP
(DeepLIFT) on the BiLSTM.
- Named DataFrames throughout the model layer — this is why PCA/unlabeled reduction is banned
  upstream; it would make RQ2 unanswerable by attribution.
- BiLSTM sequences built from the last purged fold with the **persisted production scalers**.
- SHAP averaged over the 24-step sequence so the plot is comparable to the tree beeswarms.
- Limitation, stated: SHAP is correlational. That gap is why `ablation.py` exists.

### `src/evaluation/` — the evidence layer

**`metrics.py`** — every metric, both naive baselines, and the two persistence functions everything
writes through.
- All metrics nan-safe via a shared mask; municipal telemetry has gaps and inconsistent dropping
  would invalidate comparisons.
- Normalised by `mean(|y|)` so macro MW and municipal kW nMAE are directly comparable. (Normalising
  by capacity would embed the very accounting question the thesis leaves open.)
- MBE is first-class — for downscaling, systematic bias is *the* failure mode.
- `append_cv_metrics` de-dupes on `(model, tech, fold, scale)` keep-last → re-running one trainer
  refreshes only its rows. `append_oof` de-dupes on `(tech, model, timestamp)` → trees exported in
  one pass, BiLSTM/TFT append from inside their trainers (refitting a TFT just to export would be
  hours of GPU for no information).
- `__main__` self-check with real assertions (perfect prediction ≈ 0; persistence tracks a daily
  cycle; climatology beats a global mean; a `[q10,q90]` interval gives PICP ≈ 0.80).

**`baselines_macro.py`** — persistence (t−24 h) and hour×month climatology through the identical
splitter, into the same CSV as the models.
- Climatology's per-point predictions are exported (it must be fitted per fold — pooling would let
  later folds leak into earlier ones' means); persistence's are derived on the fly in `significance`.

**`export_predictions.py`** — honest out-of-fold per-point predictions for the tree models.
- Exists because `models/*.pkl` are refit on the full dataset — predicting with them is in-sample
  and would flatter every Chapter 6 scatter and overlay.
- Builders **import** `LGB_PARAMS`/`XGB_PARAMS` from the trainers so they cannot drift. (They once
  did: omitting the quantile objective silently fell back to L2 under the same model name.)
- Runtime assertion that the exported LightGBM point forecast reproduces the trainer's q50 exactly.
- Also exports LightGBM's held-out q10/q90 — what macro conformal calibrates on.
- Self-check: OOF timestamps unique per model; ML beats the raw prior.

**`significance.py`** — Diebold-Mariano (Newey-West HAC) + moving-block bootstrap + fold-level view.
- States its own motivation: with 4 folds a perfect sweep gives sign-test p = 0.125, so no
  fold-level test can reach significance. Hence ~35k paired per-point residuals.
- Bartlett-weighted HAC at lag `⌊n^(1/3)⌋` — iid variance would overstate significance by ~√(autocorr time).
- 7-day moving blocks; both models resampled on the **same** draw (that's what makes it paired).
- The bootstrap is the arbiter if the two disagree (DM needs an asymptotic normal; the bootstrap
  doesn't).
- Relative effect size printed next to every p-value — on 35k hours a significant gap can be trivial.
- Self-check: a degraded copy must lose significantly with the CI excluding zero *in the right
  direction* (guards a sign flip); a model against itself gives an identically zero differential.

**`conformal.py`** — width-normalised split CQR primitives.
- Multiplicative (width-normalised) so heteroscedasticity survives: PV night intervals stay near
  zero instead of being inflated to the mean width by an additive offset.
- `w_floor` guards the `0/0` and `+∞` degenerate cases (PV at night) and is **selected on the
  calibration split** from a grid spanning pure-multiplicative to pure-additive — all candidates hit
  nominal coverage by construction, so the sharpest wins. Persisted **with** Q (a different floor
  silently changes the correction).
- Caveats in the docstring: contiguous blocks aren't exchangeable draws (approximate, not exact,
  guarantee) and tuning the floor forfeits the strict a-priori-score condition.
- Five self-check fixtures including the forced-additive case and the all-zero no-op.

**`conformal_macro.py`** — fit Q on fold *f*, evaluate on fold *f+1*.
- Strictly sequential: same-fold is in-sample; a later fold would reintroduce look-ahead. Leaves 3
  honest evaluations per technology.
- Its real purpose: proves under-coverage is **not** purely a downscaling artefact (macro wind PICP
  0.62 vs 0.80 nominal).

**`ablation.py`** — the causal answer to RQ2. Variants `full` / `no_prior` / `prior_only` +
`physics_prior_raw` (prior as forecast, no learner), same purged CV.
- Ablation over attribution: SHAP reports how a model internally accounts for inputs — a feature can
  dominate a ranking and still be replaceable by its correlates.
- `prior_only` pre-empts "the weather features are just redundant with the prior".
- `physics_prior_raw` separates "the prior carries the signal" from "the learner corrects the prior".
- TFT has its own variant map (its `full` must be `tft_known_reals`, not `FEATURE_COLS`).
- BiLSTM has no `prior_only` row — it can't express a technology-specific one, and redefining the
  variant for one model would make the column mean two things. An omission with a stated reason.
- Self-check **asserts** the RQ2 ordering: if dropping the prior ever helps, the run fails loudly.

**`nwp_gap.py`** — reanalysis vs NWP-driven features.
- Per-fold fresh models predicting **both** feature sets on the held-out window. Reusing the
  production model made the reanalysis side in-sample and overstated the gap.
- The two matrices share target and timestamps exactly; only weather differs — the control was built
  in the data layer, not argued for afterwards.
- `demo()`: a noisier forecast input must not beat the clean one.

### `src/validation/` — cross-scale evidence

**`kw_features.py`** — the municipal feature builder; the most load-bearing file in the downscaling
story.
- `assemble_municipal_features(..., load_node_weather)` takes a weather-loader callable, so the
  **same** function serves archive-based validation and live-NWP serving. Identical physics on both
  paths is what keeps the frozen calibration valid at inference.
- Priors normalised against **local** nameplate — same column names, same range, different fleet.
- 8 km radius: a compromise between "what the meter plausibly sees" and "enough assets for a stable
  prior". Also the main source of the wind-scale divergence.
- Per-asset wind physics locally (real hub heights/rotors) rather than the macro representative
  turbine — at municipal scale a handful of specific machines determine output.
- **The E.ON scrape is UTC and is never converted** — verified empirically (PV/wind cross-correlate
  with UTC weather at 0–1 h lag, not +2 h). A silent 2 h shift would degrade every municipal metric
  while looking plausible.
- `calibrate_affine`: OLS with intercept → fit-split MBE identically zero; falls back to offset-only
  when the base is degenerate or the slope is non-positive. Self-check recovers a known
  (scale 2.0, offset 300) relation.

**`municipal_validator.py`** — the cross-scale experiment for all tree/BiLSTM models plus a municipal
persistence baseline.
- 40% calibration / 60% validation of the telemetry overlap, chronological, no hardcoded window.
- Records `nmae_k1` / `mbe_k1` (uncalibrated) beside the calibrated numbers, making the
  calibration's contribution measurable rather than assumed.
- The affine fitted on the median is applied unchanged to q10/q90 (positive slope preserves
  ordering) — and that is exactly why coverage collapses, which conformal then treats, fitted on the
  **same** calibration split so no extra data is consumed.
- Calibration params merged (not overwritten) into `results/<slug>/affine_calibration.json`, so the
  TFT validator can add its own keys.

**`tft_municipal_validator.py`** — separate file because `pytorch-forecasting` inference needs its own
dataset construction and decoder index arithmetic.
- `preds[:, 23]` with `time_idx + 23`: `x_to_index` returns the **first** decoder step, so the
  horizon covers `[t, t+23]`. Off-by-one here silently shifts every TFT municipal prediction.
- Merges into the existing CSV/parquet, replacing only `model == 'tft'` rows.

**`lomo_transfer.py`** — leave-one-municipality-out: fit at one town, apply verbatim at the other.
Variants `oracle` / `transferred` / `uncalibrated`, all scored on the target's held-out split.
- Exists because the 16 unvalidated districts have no telemetry, so "does the contract travel?" must
  be asked. The answer for wind is largely no.
- **Asserts** that Pearson `r` is invariant under a positive-slope affine map (checked where the
  zero-floor never fired). Therefore transfer degradation is *magnitude* (capacity accounting), never
  *shape* (forecast skill) — the empirical backing for §7.4.
- Reports `clipped_frac` to mark rows where the invariance argument doesn't apply.

**`extend_kw_weather.py`** — extends archived weather nodes forward.
- Exists because the batch ingester's `completed_nodes` state is binary: once fetched, a node is
  never revisited, so a nightly re-run is a permanent no-op and the archive silently stops growing.
  This extends from each node's own max timestamp.
- Drops all-NaN trailing rows — ERA5(T) lags real time and the API pads with nulls; keeping them
  would advance the max past real data and skip those hours forever.
- `--all` catches `RequestException` per node so one flaky call can't abort the other 429. Cron
  entrypoint.

### `src/measurements/` and `src/dashboard/`

**`scraper_service.py`** — polls the E.ON Energiemonitor per region, appends to CSV.
- Append-only: this is the *only* ground truth in the thesis and cannot be obtained retroactively.
- Uses the payload's interval-**end** timestamp (falling back to wall-clock) — scrape time ≠
  measurement time.
- Degrades gracefully if the weather endpoint fails but the meter endpoint succeeds.
- Header self-repair for the legacy pre-timestamp format.

**`dashboard/forecast_service.py`** — live NWP → features → CF quantiles → calibrated MW.
- Reuses `municipal_fleet` and `assemble_municipal_features` unchanged: the serving path *is* the
  validation path.
- `forecast_from_fleet(fleet, calib_provider, …)` — one core, two callers (frozen municipal
  calibration vs district identity fallback). Only the calibration source differs.
- Exponential backoff on 429/5xx (18 sequential batched calls to a free API).
- Applies the frozen conformal correction where one exists, with `lower = min(lower, median)` /
  `upper = max(upper, median)` so a negative Q can shrink the band without inverting it.
  Uncalibrated districts get an honest raw interval, not a fake corrected one.
- Requests only the five weather variables the physics actually consumes.
- Network-free self-check asserts band ordering and non-negativity.

---

## 4. Root entrypoints

| Entrypoint | Command | Notes |
|---|---|---|
| `main.py` | `uv run main.py` | Full reanalysis pipeline. Skip-if-exists on expensive immutable stages. **Baselines run before the models** — the bar is set before it's cleared. |
| `main.py --mode forecast` | `uv run main.py --mode forecast` | NWP weather + forecast matrix only. Deliberately does not retrain: models trained on reanalysis are applied unchanged, which is what makes the gap measurable. Raises if reanalysis outputs are missing. |
| `dashboard.py` | `uv run streamlit run dashboard.py` | 18 Kreise + aggregate; serves the batch parquet when present, else live NWP. Data stays UTC, `Europe/Berlin` at render only. The aggregate band is the comonotonic sum of district P10/P90 — disclosed in the UI as a conservative bound. |
| `run_scraper.py` | `uv run run_scraper.py [--loop]` | The one Hydra entrypoint (historical). `--loop` stripped from `sys.argv` before Hydra parses. Loop compensates for scrape duration so cadence doesn't drift. |
| `run_pipeline.sh` | cron, nightly | `extend_kw_weather --all` with logging. |
| `conf/config.yaml` | — | Single flat config: MaStR URLs, Open-Meteo endpoints + variables, physics constants (α = 0.14, PR = 0.80, temp. coefficient), scraper regions, 18-district registry. Physics constants live here so they're auditable and tunable rather than buried. |

---

## 5. On-disk contracts — source of truth

| Artifact | Written by | Contract |
|---|---|---|
| `data/raw/mastr_*_raw.csv` | `mastr_bulk_parser` | **Read-only.** Status-35 units inside the boundary only — no downstream re-filtering. |
| `data/external/brandenburg_{boundary,districts}.geojson` | `boundary_fetcher`, `districts` | Fetched once, cached; districts validated for completeness. |
| `data/processed/{wind,solar}_clusters.csv` | `geospatial_clustering` | Capacity-weighted centroids; **sum of `total_capacity_mw` is the fleet nameplate used for CF normalisation.** |
| `data/processed/weather{,_forecast}/grid_nodes/*.parquet` | `weather_ingestion` (+ `extend_kw_weather`) | 430 nodes, 0.1° grid, hourly, **UTC**. Isolated directories so reanalysis and NWP can never be confused. |
| `data/processed/actual_generation_50hertz.parquet` | `target_harmonization` | UTC, 15-min, two MW columns. |
| **`ml_training_matrix.parquet`** | `feature_pipeline` | 37,967 h × 17 cols. Read by every trainer, ablation, SHAP and the NWP gap. |
| `ml_training_matrix_forecast.parquet` | `feature_pipeline --mode forecast` | Identical target/timestamps; only weather-derived columns differ. |
| `cf_normalization.json` | `feature_pipeline` (**reanalysis only**) | Fleet nameplates, per-year effective capacity, PR, reference turbine. Override hook for official 50Hertz figures. |
| `models/lightgbm_{tech}.pkl` (+ `_q10/_q90`) | `train_lightgbm` | The canonical file is **q50**, so consumers never learn about quantiles. |
| `models/bilstm_{tech}.pth` + `bilstm_scalers.pkl` | `train_bilstm` | Scalers are **not optional** — municipal inference is impossible without them. |
| `models/tft/tft_{tech}_production.ckpt` | `train_tft` | Lightning checkpoint carrying `dataset_parameters`. |
| `results/macro_cv_metrics.csv` | all trainers + baselines | Keyed `(model, tech, fold, scale)`; CF and MW rows; incrementally refreshable. |
| `results/macro_oof_predictions.parquet` | trees via `export_predictions`, BiLSTM/TFT from their trainers, climatology from baselines | Keyed `(tech, model, timestamp)`. **All five models must have run** for significance to cover the full ranking. |
| `results/{model_significance,ablation_metrics,conformal_coverage,lomo_transfer,municipal_validation_metrics}.csv` | respective modules | Ablation is a separate CSV so ablated names can't collide with production rows. |
| `results/nwp_gap_report.{csv,md}` | `nwp_gap` | |
| `results/<slug>/affine_calibration.json` | `municipal_validator` + `tft_municipal_validator` (merged) | **The downscaling contract as a versioned artifact:** per `(model, tech)` → `{scale, offset, cap_mw, conformal_q, conformal_w_floor}`. |
| `results/forecasts/brandenburg_state_forecast.parquet` | notebook 11 | Batch forecast for 18 Kreise; dashboard falls back to live NWP when absent. |

---

## 6. Headline results (for cross-reference)

**Macro CF nMAE, mean over 4 purged folds:**

| Tech | LightGBM | TFT | XGBoost | EBM | BiLSTM | persistence | climatology |
|---|---|---|---|---|---|---|---|
| wind | **0.199** | 0.199 | 0.211 | 0.218 | 0.261 | 0.649 | 0.685 |
| solar | **0.179** | 0.209 | 0.198 | 0.214 | 0.312 | 0.286 | 0.345 |

**Significance** — wind LightGBM vs TFT: ΔnMAE +0.0007, DM p = 0.658 → **indistinguishable**.
Solar LightGBM vs TFT: −14.3%, p < 1e-4 → real. Both beat persistence by 69% (wind) / 37% (solar).

**Ablation (RQ2)** — removing the physics prior: wind **+189%** nMAE, solar **+39%**. Raw prior with
no learner: wind 0.281, solar 0.419; the learner improves on it by 29% / 57%. Wind is
prior-dominated (`prior_only` 0.1915 ≈ `full` 0.1995); solar is learner-dominated. The asymmetry is
itself the finding.

**Municipal (held-out 60%, LightGBM):** KW wind 0.322 / solar 0.161; Nauen wind 0.385 / solar 0.337.
Uncalibrated equivalents: 1.03 / 17.6 / 66.98 / 8.48. Municipal persistence: 0.687 / 0.294 / 0.700 /
0.319 — three of four beat it, Nauen solar does not.

**LOMO transfer** — oracle vs transferred vs uncalibrated nMAE: wind KW 0.334 / 0.965 / 1.011;
wind Nauen 0.392 / **29.98** / 66.45; solar KW 0.171 / 1.707 / 17.27. Fitted wind scales differ 30×
(0.4595 vs 0.0153). `corr` is invariant → the failure is magnitude, not shape.

**Conformal PICP (nominal 0.80):** macro wind 0.62/0.69/0.68 → 0.85/0.86/0.77. Municipal
KW wind 0.38 → 0.79, KW solar 0.33 → 0.80, Nauen wind 0.41 → 0.79, Nauen solar 0.17 → 0.82.

**NWP gap** (35,040 OOF hours): wind MAE **+90.9%**, solar **+76.8%** vs reanalysis.

---

## 7. Known limitations

1. **Priors are Brandenburg-only; the target is the whole 50Hertz zone** (effective wind capacity
   ~16 GW vs 9.57 GW Brandenburg fleet). The Brandenburg fleet is a spatially representative sample;
   CF makes the level difference absorbable and the learner absorbs it.
2. **The 30× wind `scale` divergence is a capacity-accounting open question** — the 8 km radius is a
   proxy for an unpublished metering boundary. Proven to be magnitude, not shape.
3. **Two municipalities** is the entire public ground truth for Brandenburg; LOMO is an n = 2
   experiment.
4. **Ch. 2.5.3 says residual learning; the implementation is input augmentation** (prior as feature,
   direct CF prediction). Both are valid hybrid couplings — name it correctly: feature-level hybrid.
5. **The conformal guarantee is empirical, not theoretical** (contiguous blocks aren't exchangeable;
   `w_floor` is tuned).
6. **Precomputed solar geometry is unused** by the final feature set.
7. **`mastr_downloader.download_brandenburg_boundary` duplicates `boundary_fetcher`** (dead code).
8. **Municipal validation spans 855 hours** (~36 days) — enough for a stable affine fit, not enough
   to characterise seasonality.
