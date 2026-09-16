# Architecture — Defense Grade

**Scope:** every module under `src/`, the root entrypoints that invoke them, and the on-disk
artifact contracts that bind them together. For each: what it does, which engineering decision
was made, which alternatives were rejected, and the answer to give when the decision is challenged.

**How to read this:** §1 is the one-page mental model. §2 is the seven decisions the whole system
rests on — if you only rehearse one section, rehearse that one. §3 is the per-file reference.
§4–§5 are entrypoints and data contracts. §6 is the defense dossier: the four attack surfaces,
with the numbers that answer them. §7 states the weaknesses before the committee does.

---

## 1. The system in one page

The problem is an **observation asymmetry**. 50Hertz publishes generation for its entire control
zone at 15-minute resolution. Nobody publishes generation for a Brandenburg Gemeinde. A model
trained on the zonal signal learns gigawatts; a municipality produces kilowatts; the model cannot
be shrunk to fit. Two towns (Königs Wusterhausen, Nauen) have public E.ON telemetry — enough to
*calibrate and validate* a downscaling, nowhere near enough to *train* a local model.

The architecture is the resolution of that asymmetry: **make the learning problem dimensionless,
so the map learned at zonal scale is the same map that applies at municipal scale, and let a
two-parameter local calibration absorb everything that is genuinely scale-dependent.**

```
                          ┌─────────────────────── data layer (src/data) ────────────────────────┐
  MaStR bulk XML  ──────► mastr_downloader ──► mastr_bulk_parser ──► data/raw/mastr_{wind,solar}_*.csv
  OSM Overpass    ──────► boundary_fetcher  ──► data/external/brandenburg_boundary.geojson  (spatial mask)
  50Hertz CSVs    ──────► target_harmonization ──► actual_generation_50hertz.parquet   (UTC, 15-min)
  Nominatim       ──────► districts.py      ──► brandenburg_districts.geojson  (18 Kreise, state-wide scaling)
                          └──────────────────────────────────────────────────────────────────────┘
                                                     │
                          ┌──────────────────── feature layer (src/features) ────────────────────┐
   raw assets ──► geospatial_clustering ──► 211 wind + 882 solar clusters ──► 430 weather grid nodes
                                                     │                              │
                                              solar_geometry (pvlib)          weather_ingestion (Open-Meteo)
                                                     │                              │
                                                     └──────────► feature_pipeline ◄┘
                                                                       │
                        wind_physics_transformer (power curves + Hellmann)  ──► physical priors
                                                                       │
                                                     ml_training_matrix.parquet  (37,967 h × 17 cols)
                                                     schema.py = THE column contract (9 features, CF target)
                          └──────────────────────────────────────────────────────────────────────┘
                                                     │
       ┌─────────────── models (src/models) ─────────┴──────── evaluation (src/evaluation) ───────┐
       LightGBM q10/q50/q90    ─┐                              baselines_macro (persistence, climatology)
       XGBoost, EBM            ─┤  all under                   export_predictions ──► OOF parquet
       BiLSTM                  ─┤  PurgedExpandingWindowSplitter  significance (Diebold-Mariano + block bootstrap)
       TFT                     ─┘  (4 folds, 365 d test, 72 h embargo)  ablation (RQ2, causal)
       explain_models / explain_bilstm (SHAP)                   conformal / conformal_macro (interval repair)
                                                                nwp_gap (reanalysis vs NWP)
       └────────────────────────────────────────────────────────────────────────────────────────┘
                                                     │
                          ┌─────────────── validation (src/validation) ──────────────────────────┐
        kw_features (same scale-free contract, local fleet) ──► municipal_validator ──► affine (scale, offset)
                                                             ──► tft_municipal_validator
                                                             ──► lomo_transfer (does the contract travel?)
                          └──────────────────────────────────────────────────────────────────────┘
                                                     │
        src/dashboard/forecast_service ──► live Open-Meteo NWP ──► CF ──► ×nameplate ──► ×scale + offset ──► MW
                                                     │
                                    dashboard.py (Streamlit, 18 Kreise + state aggregate)
```

**The one-sentence version:** a bottom-up physical simulation of the registered fleet produces a
dimensionless prior; a learner trained on the 50Hertz capacity factor corrects that prior; the
corrected capacity factor is multiplied by a *local* nameplate and a *locally fitted* affine
constant to produce municipal megawatts.

---

## 2. The seven decisions the system rests on

### D1 — The target is a capacity factor, not megawatts

**Decision.** Every model predicts `CF_t = y_t / C_t ∈ [0, 1.5]`. Absolute MW is reconstructed only
at reporting/inference time. Implemented in `src/features/feature_pipeline.py` (target
construction) and enforced by `src/features/schema.py` (`TARGET_CF`, `CF_CLIP = 1.5`).

**Why.** Two independent reasons, and it is worth separating them because they answer different
objections:

1. *Non-stationarity.* Solar nameplate in the zone grew from 10.8 GW (2022) to 14.3 GW (2026).
   An MW target carries that commissioning trend as a deterministic upward drift the model would
   have to learn and then extrapolate — a guaranteed failure mode on the last CV fold.
2. *Scale invariance (this is RQ1).* A capacity factor is fractional utilisation. It is the same
   quantity at 16 GW and at 4 MW. The features are constructed to match: the physical priors are
   divided by fleet nameplate, so they are capacity factors too; weather intensities and cyclical
   time are already dimensionless. **The learned map has no units in it, so nothing about it is
   zone-specific.** That is what makes downscaling possible at all.

**Alternatives rejected.**
- *Raw MW target.* This was the original implementation and it failed: a macro model predicting
  gigawatts cannot be shrunk to a Gemeinde. Dividing the macro MW prediction by a hardcoded zone
  capacity was tried and is explicitly forbidden in the codebase — it hardcodes an accounting
  assumption that is wrong the moment the fleet changes.
- *Per-municipality models.* Impossible: there are no local training labels. This is the whole
  premise of the thesis.
- *Log-MW or detrended MW.* Removes the trend but not the units; still does not transfer.

**Evidence.** §6 of the results: the uncalibrated CF downscaling at Königs Wusterhausen gives
wind nMAE 1.03 and solar nMAE 17.6 — bad, but *finite and shape-correct* (`corr` 0.82 / 0.97).
The MW formulation did not produce a bad number; it produced an anti-correlated one.

**If challenged — "isn't `CF_CLIP = 1.5` an admission the normalisation is wrong?"**
No. `C_t` is a *data-driven effective-capacity proxy* (99.9th percentile), not a registry
nameplate. A proxy can undershoot true nameplate, so CF > 1 is legal and clipping at 1.5 is a
guard against a division blow-up, not a fudge of the physics. The bound is defined once in
`schema.py` and used identically by every trainer and validator — a mismatched bound between
training and municipal inference would silently diverge the results, which is precisely why it
lives in the schema module.

---

### D2 — Effective capacity is a *causal* trailing statistic

**Decision.** `_effective_capacity()` in `src/features/feature_pipeline.py`:

```python
cap = y.rolling('365D', min_periods=24*30).quantile(0.999).shift(1).cummax()
```

A trailing-365-day robust peak, **shifted one step** so `cap_t` uses only data strictly before `t`,
then `cummax` so the implied fleet is physically monotone.

**Why.** The previous implementation used a per-calendar-year 99.9th percentile. That normalised
every timestamp in a year — including every timestamp inside a CV *test* fold — with a capacity
computed from that same year's future data. That is look-ahead leakage *inside the target
definition*, which is the most insidious kind: it survives any amount of care in the splitter.

`cummax` is not cosmetic: without it, a becalmed rolling window would shrink the implied capacity
and inflate the CF, creating spurious target variance that has nothing to do with weather.

**Alternatives rejected.**
- *Official 50Hertz installed-capacity figures.* Preferred in principle, and the code has an
  explicit override hook (`data/processed/cf_normalization.json`). Not used because a published,
  timestamp-aligned installed-capacity series for the zone was not obtainable for the full period.
- *Registry (MaStR) nameplate as `C_t`.* Wrong quantity: MaStR is a Brandenburg-only registry and
  the target is the whole 50Hertz zone (see §7, T1).
- *Rolling max instead of p99.9.* One bad meter spike permanently redefines the capacity.
  The 99.9th percentile is the robust version of the same idea.

**Cost, stated honestly.** The first ~30 days have no capacity estimate and are dropped — the
matrix starts 2022-01-31, not 2022-01-01.

---

### D3 — Physics enters as a *feature*, computed bottom-up from the real fleet

**Decision.** `wind_prior_cf` and `solar_prior_cf` are the first two columns of `FEATURE_COLS`.
They are produced by simulating every registered asset:

- **Wind** (`src/models/wind_physics_transformer.py` + `calculate_regional_wind_prior`):
  Open-Meteo 100 m wind speed → Hellmann power-law extrapolation to hub height
  (`v_hub = v_100 · (z/100)^α`, α = 0.14) → matched manufacturer power curve → 1-D interpolation
  → clipped to `[0, nameplate]`.
- **Solar** (`calculate_regional_solar_prior`): `GHI / 1000 W/m² · nameplate · PR(0.80)` —
  shortwave radiation normalised to Standard Test Conditions, times a generic performance ratio.
- Both are then divided by fleet nameplate → dimensionless.

**Why.** This is the hybrid coupling that answers RQ2. The physical model contributes the
structure that data alone would have to rediscover: the cubic-then-saturating power curve, the
cut-out cliff, the hub-height profile, the *actual spatial composition of the fleet*. The learner
contributes what physics cannot know: wake losses, curtailment, icing, availability, systematic
NWP bias, and the difference between the Brandenburg fleet shape and the zonal aggregate.

**Two implementation choices worth defending explicitly, because both were bugs first:**

1. *Per-unit power curve.* The wind prior uses a representative 3 MW turbine to build a
   **dimensionless** curve, which is then scaled by the whole cluster's nameplate
   (`per_unit = ref_kW / 3000`, then `× capacity_mw`). Passing the *cluster* nameplate as
   `gross_power` matched a single ~10 MW curve and capped a 366 MW / 153-turbine cluster at
   10 MW — a ~30× under-scaled prior. The fix is in `feature_pipeline.py:61-82` with the
   rationale in-line.
2. *STC normalisation.* `shortwave_radiation / 1000.0`. Omitting the `/1000` inflated the solar
   prior by ~1000× (peak 1.19e6 MW). Both the training pipeline and the municipal feature builder
   (`kw_features.py:119`) carry the same corrected form — they must, or the municipal features
   would be on a different scale than the ones the model was trained on.

**Alternatives rejected.**
- *Residual learning* (fit the learner on `y − prior`). This is what Ch. 2.5.3 of the thesis
  describes as the general hybrid pattern, and it is **not** what the implementation does —
  see §7, T4. Input augmentation was chosen because it lets the learner *ignore* the prior where
  the prior is unreliable (solar, where the raw prior nMAE is 0.419 against a learned 0.180),
  rather than being anchored to it. A residual formulation forces a 1:1 sensitivity to prior error.
- *Weather-only ML.* Directly measured, and it is the RQ2 answer: removing the priors costs wind
  +188% nMAE and solar +39% (§6, A2).
- *pvlib full POA transposition per asset.* MaStR azimuth/tilt are coarse catalogue codes
  (8 azimuth bins, 5 tilt bins) and are missing for most of the 174k distributed units. A
  per-asset transposition built on interpolated catalogue bins would be false precision; the flat
  performance ratio is the honest simplification, and it is a single tunable constant.

**If challenged — "the priors are Brandenburg-only but the target is the whole 50Hertz zone."**
Correct, and stated as a threat to validity in §7, T1. The defense is that both sides are
capacity factors: the Brandenburg fleet is a large, spatially representative *sample* of the zonal
fleet, so its CF is a proxy for the zonal CF's **shape**, and the learner absorbs the level
difference. The empirical backing is the fit itself (wind CF corr 0.955) — if the Brandenburg
sample carried no zonal information, that number would not exist.

---

### D4 — Validation is purged, expanding-window, with a 72-hour embargo

**Decision.** `src/features/validation_splitter.py::PurgedExpandingWindowSplitter`, used
identically by *every* trainer, every ablation, the NWP-gap study and the OOF exporter:
4 folds, 365-day test windows, 72-hour purge gap, expanding train window.

**Why.** Three separate failure modes are being closed:

1. *Temporal ordering.* K-fold on a time series trains on the future to predict the past. Any
   number produced that way is uninterpretable.
2. *Atmospheric persistence.* Synoptic weather systems have an autocorrelation time of days.
   A train window that ends at `t` and a test window that starts at `t+1h` share the same weather
   system — the model has effectively seen the test conditions. The 72-hour embargo is sized to
   exceed the synoptic decorrelation time, not chosen for convenience.
3. *Boundary sharing.* The test window is half-open at the start (`dts > test_start`), so fold
   *i*'s last timestamp is never fold *i+1*'s first. Without this, adjacent folds share a point
   and the OOF parquet would contain duplicates.

**Expanding, not sliding.** A sliding window would discard early data and give each fold a
different training-set size, confounding "later fold" with "more data". Expanding matches how the
system would actually be retrained in operation.

**A second, subtler leak that was closed.** Early-stopping validation for the sequence models
originally used the *test fold* as `val_loss`. That leaks test information into epoch selection
even though no test gradient is ever taken. Both `train_bilstm.py:146` and `train_tft.py:97` now
carve the validation split from the **last 10% of the training window**; the TFT variant also
prepends `MAX_ENCODER_LENGTH` hours of history so the validation dataset has a legal encoder
context.

**If challenged — "four folds is not many."** Agreed, and the codebase says so out loud:
`src/evaluation/significance.py` opens by noting that with 4 folds even a perfect 4–0 sweep gives
a sign-test p of 0.125, so *no* fold-level test can reach conventional significance. That is
exactly why the per-point Diebold-Mariano and block-bootstrap machinery exists (§6, A1). The fold
count is constrained by data: 4 years of 50Hertz actuals, 365-day test windows.

---

### D5 — The downscaling contract is a frozen, per-municipality affine map fitted with an intercept

**Decision.** `Local_kW = clip(scale · (CF_pred · nameplate_mw · 1000) + offset, 0, ∞)`.
`(scale, offset)` is fitted per `(municipality, model, technology)` by OLS with an intercept
(`kw_features.py::calibrate_affine`) on the **first 40%** of the telemetry overlap, evaluated on
the remaining 60%, and frozen to `results/<slug>/affine_calibration.json` for inference re-use
(`apply_affine`, `forecast_service.py`).

**Why an intercept.** The requirement was "drive the municipal bias to zero". Including an
intercept makes the mean residual on the fit split *identically zero* by construction — MBE = 0
is an algebraic property of OLS with an intercept, not an empirical hope. Minimising MBE directly
is degenerate: `scale = 0, offset = mean(y)` also achieves MBE = 0 while destroying all skill.
OLS picks the unique best-fit line *among* the MBE = 0 solutions. That argument is written into
the docstring and asserted in the module's self-check.

**Why frozen, not refitted at inference.** The dashboard must not need telemetry at request time —
that is the entire operational point. Freezing also makes the served forecast reproducible and
auditable: the calibration JSON is a versioned artifact.

**Why a 40/60 split rather than a fixed date window.** The previous implementation hardcoded a
validation window. That is brittle (it breaks when telemetry grows) and it invites the accusation
of window-shopping. A fixed *fraction* of whatever overlap exists is defensible and stable.

**Alternatives rejected.**
- *Scale-only (no intercept).* Cannot zero the bias when the local meter has a constant offset
  (behind-the-meter self-consumption, unmetered assets).
- *Per-hour or per-season calibration.* More parameters fitted on 855 validation hours; overfits.
- *Fitting on all telemetry.* Then the reported municipal nMAE is in-sample and worthless.

**The uncomfortable result, owned up front.** The fitted wind `scale` is 0.46 at Königs
Wusterhausen and 0.0153 at Nauen — a 30× divergence. See §6, A2 and §7, T2: this is a
**capacity-accounting** discrepancy (what the E.ON meter observes vs. what MaStR registers within
an 8 km radius), and the proof that it is accounting rather than skill is that Pearson `r` is
invariant under a positive-slope affine map — asserted, not claimed, in
`lomo_transfer.py:148-153`.

---

### D6 — Uncertainty is quantile regression, repaired by width-normalised split conformal

**Decision.** LightGBM is fitted three times per technology at α ∈ {0.1, 0.5, 0.9}
(`train_lightgbm.py`). The **median is the point forecast** (not a separate L2 model); q10/q90 form
an 80% interval. Independently-fitted quantiles are sorted per-row so they cannot cross. Coverage
is then repaired by split conformal (`src/evaluation/conformal.py`) with a conformity score
normalised by the interval half-width.

**Why the median is the point forecast.** The pinball loss at α = 0.5 is L1. Renewable forecast
errors are fat-tailed (ramp events, curtailment); an L2 mean regressor is pulled by those tails.
Using q50 also means one objective family produces the whole forecast — there is no risk of the
point forecast and the interval describing different models. `export_predictions.py:91-93` asserts
this equality at runtime, precisely because it once drifted (the OOF exporter silently fell back
to LightGBM's default L2 objective and reported it under the same name).

**Why conformal was necessary.** The affine downscaling rescales interval *width* by `scale`
without re-deriving what the width should be locally. Realised municipal coverage collapsed to
0.38 (KW wind) and 0.17 (Nauen solar) against an 0.80 nominal. And — importantly for the defense —
it is *not purely a downscaling artefact*: the macro wind interval also under-covers (PICP 0.62).

**Why width-normalised (multiplicative) rather than textbook additive CQR.** An additive offset
inflates every hour by the same absolute amount, which is visibly wrong for PV: night hours have
near-zero width and near-zero error, and would be inflated to the mean width. The multiplicative
form preserves heteroscedasticity. But the multiplicative form is not always right either — where
the affine step has destroyed the relationship between width and local error, it must inflate
enormously to reach coverage. So the floor `w_floor` is **selected on the calibration split** from
a grid spanning both regimes (a tiny floor = pure multiplicative; a huge floor = pure additive),
choosing the sharpest interval among candidates that all hit nominal coverage by construction.
The self-check in `conformal.py` includes a fixture (case 4) that *forces* the selector into the
additive regime and asserts it gets there.

**Honesty about the guarantee.** The module docstring states both caveats explicitly: (1)
calibration and evaluation blocks are contiguous in time, not exchangeable draws, so this restores
*approximately* nominal coverage under stationarity, not the exact finite-sample conformal
guarantee; (2) tuning `w_floor` forfeits the strict a-priori-score condition. Reported coverage is
therefore "measured on held-out data", not "guaranteed by theory". **Say this before you are
asked.**

---

### D7 — One schema module is the contract binding training, validation and serving

**Decision.** `src/features/schema.py` (41 lines) defines `FEATURE_COLS`, `TARGET_CF`, `TARGET_MW`,
`CAP_COLS`, `CF_CLIP` and `tft_known_reals()`. Every trainer, every validator, the SHAP explainer
and the live forecast service import from it. Nothing hardcodes a column list.

**Why this is architectural and not housekeeping.** The system's central claim is that *the same
feature vector* is built at macro scale (from 430 weather grid nodes and the whole Brandenburg
fleet) and at municipal scale (from a handful of nodes and an 8 km fleet) — and that this is what
makes the frozen calibration valid at inference time. If those two construction paths could drift
apart by one column, the claim is false and nothing downstream is meaningful. The schema module is
where that invariant is enforced. It also enforces the negative constraint: **no MW column and no
capacity column may ever enter `X`** — those are kept in the matrix for reconstruction and
reporting only.

---

## 3. Module reference

### 3.1 `src/data/` — acquisition and harmonisation

#### `mastr_downloader.py` (223 lines)
Streaming download of the ~multi-GB Marktstammdatenregister bulk ZIP, selective extraction of only
`EinheitenWind.xml` / `EinheitenSolar_*.xml`, plus one-off fetches of the Brandenburg boundary and
the WZB postal-code coordinate lookup.

- **Choice: stream + selective extract, never hold the ZIP in memory.** 1 MB chunks to disk, then
  `shutil.copyfileobj` per member, then the temp ZIP is purged in a `finally`. The full export
  contains dozens of unit types the thesis does not use.
- **Choice: idempotent skip-if-exists on every download.** Re-running the pipeline must not re-pull
  gigabytes. Applies to MaStR, the boundary GeoJSON and the postal codes.
- **Known duplication:** `download_brandenburg_boundary` here and `fetch_brandenburg_boundary` in
  `boundary_fetcher.py` are the same routine. `main.py` calls the `boundary_fetcher` one; this copy
  is dead weight retained from the original download-orchestration script. Call it out yourself if
  a reviewer spots it — it is duplication, not a correctness issue.

#### `mastr_bulk_parser.py` (435 lines)
SAX-style streaming parser (`ET.iterparse`) producing `data/raw/mastr_{wind,solar}_brandenburg_raw.csv`.

- **Choice: `iterparse` with `elem.clear()` + `root.clear()` per record.** A DOM parse of the full
  export does not fit in memory. Clearing the *root's* children as well as the element is the part
  people forget — without it `ElementTree` retains every processed sibling and memory still grows
  linearly.
- **Choice: two-stage filter — attribute pre-filter, then true spatial mask.** Records are first
  cheaply filtered on `Bundesland ∈ {1400, 14}` and `EinheitBetriebsstatus == 35` ("in operation"),
  then the survivors are converted to a GeoDataFrame and masked with
  `within(brandenburg_polygon)`. The administrative field is unreliable near borders and for
  mis-registered units; the polygon is the authority. This ordering matters for runtime: the
  expensive geometric test only sees a small fraction of rows.
- **Choice: status-35 filter at parse time.** This is why `districts.py` can state that the raw
  CSVs *already are* the active fleet and skip any commissioning/decommission logic — the
  invariant is established once, at the boundary.
- **Choice: postal-code coordinate imputation with a centroid fallback.** MaStR coordinates are
  frequently missing for small rooftop PV. Order: real coordinates → PLZ centroid → Brandenburg
  centroid. The last tier is deliberately crude; it exists so a unit's *capacity* is not lost from
  the fleet total even when its location is unknown. Because capacity aggregates to clusters and
  the prior is capacity-weighted, a mislocated small unit perturbs the spatial weighting slightly,
  it does not corrupt the total.
- **Choice: catalogue-code → physical-value maps for azimuth/tilt** (`695→0°`… `810→12.5°`).
  MaStR stores orientation as coarse bins; the maps take bin midpoints. These fields are parsed but
  currently unused by the solar prior (see D3, rejected alternatives) — they are retained because
  they are the natural upgrade path to a POA transposition.
- **`ProgressFile`:** a file-like wrapper feeding a `tqdm` bar. Cosmetic, but it is the reason a
  40-minute parse is not indistinguishable from a hang.

#### `boundary_fetcher.py` (149 lines)
Fetches OSM relation 62504 (Brandenburg) from Overpass and reconstructs a polygon. Also the home of
`load_config()`, which most of the codebase imports.

- **Choice: reconstruct the polygon from raw ways rather than trusting a pre-built geometry.**
  Overpass returns a relation as a bag of `way` members with `inner`/`outer` roles.
  `linemerge → polygonize → unary_union`, then `outer.difference(inner)` for enclaves. Getting the
  inner-role handling right matters for any exclave/hole in the state boundary.
- **Choice: cache to `data/external/`, fetch once.** Overpass is rate-limited and occasionally
  down; a boundary that changes between runs would silently change the asset set.
- **Note:** `load_config` lives here for historical reasons and is imported by ~10 modules. It is a
  20-line YAML reader; there is no config framework in the `src/` path (Hydra is used only by the
  scraper entrypoint).

#### `target_harmonization.py` (229 lines)
50Hertz "Hochrechnung" CSVs → `actual_generation_50hertz.parquet`, UTC, 15-minute, gap-treated.

- **Choice: explicit DST handling via `tz_localize('Europe/Berlin', ambiguous='infer')` → UTC.**
  The source files are in local time. The autumn transition produces a duplicated hour; `infer`
  resolves it from monotonicity of the surrounding series. Everything downstream is UTC-only —
  timezone is a *display* concern, handled once, in `dashboard.py`.
- **Choice: parse defensively for real-world CSV pathology.** UTF-16-LE encoding, 4 metadata rows
  to skip, `;` separator, comma decimals, and a family of missing-value sentinels
  (`-`, `n.v.`, `N.V.`, `nv`, …) all appear in the actual files.
- **Choice: interpolate gaps of ≤ 3 steps only (`interpolate_short_gaps`), never leading/trailing.**
  Three 15-minute steps is 45 minutes — the scale of a telemetry hiccup. Longer gaps are real
  outages and are left as NaN to be dropped, because interpolating across them fabricates target
  values the model would then be scored against. The implementation groups contiguous NaN runs via
  `(~is_nan).cumsum()` and reverts interpolation wherever the run exceeded the threshold — the
  standard idiom, and worth being able to explain on the whiteboard.
- **Choice: reindex onto a complete 15-minute grid before joining.** Guarantees a regular index so
  the downstream hourly `resample('1h').mean()` is well-defined.

#### `weather_ingestion.py` (456 lines)
Open-Meteo → `data/processed/weather{,_forecast}/grid_nodes/node_<lat>_<lon>.parquet` +
`mapping_index.json`.

- **Choice: snap cluster centroids to a 0.1° grid and fetch per *node*, not per cluster.**
  1,093 clusters collapse to 430 unique grid nodes. Open-Meteo's own resolution is coarser than
  0.1°, so this loses nothing physical and cuts API volume by ~60%. The `mapping_index.json`
  preserves the cluster → node relation so the physics can still be evaluated per cluster.
- **Choice: resumable state file + PID lock.** `.ingestion_state.json` tracks `completed_nodes` and
  a daily call counter, written atomically (`.tmp` + `replace`). A 4-year × 430-node backfill takes
  many hours and *will* be interrupted. The `.lock` file holds a PID and is validated with
  `os.kill(pid, 0)`, so a stale lock from a crashed run is detected and removed rather than
  blocking forever.
- **Choice: quota-aware 429 handling that distinguishes quota from failure.** A 429 whose reason
  contains "Daily/Hourly API request limit exceeded" sleeps until the window resets and retries
  **indefinitely** (it does not count against the retry budget); any other 429 or network error
  gets bounded exponential backoff. This distinction is the difference between a backfill that
  completes overnight and one that aborts at 4 a.m. having consumed its five retries against a
  quota wall.
- **Choice: two isolated modes with separate output directories.** `reanalysis` hits the ERA5
  archive; `forecast` hits the **historical-forecast archive** — the archived output of the actual
  forecast model over the same dates. That is what makes the NWP-gap study a clean isolation: same
  timestamps, same target, only the weather differs. A live forecast endpoint would not align on
  historical timestamps at all.
- **Choice: batch 5 locations per request + 2 s spacing.** Open-Meteo accepts comma-separated
  coordinate lists; batching is the single largest reduction in call count. The sleep is politeness
  toward a free service the thesis depends on.

#### `districts.py` (167 lines)
The state-wide scaling layer: 14 Landkreise + 4 kreisfreie Städte, polygons from Nominatim,
point-in-polygon asset assignment, fallback calibration.

- **Choice: Nominatim, not Overpass, for district polygons.** Overpass area queries for named
  admin regions time out; Nominatim resolves named administrative areas reliably and returns
  `polygon_geojson` directly. Cached to `data/external/brandenburg_districts.geojson`, with a
  1.1 s sleep between calls per Nominatim's usage policy.
- **Choice: point-in-polygon (`gpd.sjoin`), one join for the whole fleet.** MaStR has no Kreis
  field, only lat/lon. A single spatial join is both correct and fast; per-asset lookups would be
  quadratic.
- **Choice: `representative_point()` rather than `centroid` for the display coordinate.** A
  centroid is not guaranteed to lie inside a concave or multi-part polygon.
- **Choice: registry-completeness validation.** `load_district_boundaries` raises if any of the 18
  expected names is absent from the cache, so a partially-failed fetch cannot silently drop a
  district from the state total.
- **Choice: fleets returned in the *same tuple shape* as `municipal_fleet`.** This is what lets
  `assemble_municipal_features` be reused verbatim for districts — identical physics, no parallel
  code path. Deliberate structural reuse, and the reason a district forecast is trustworthy to the
  same degree the municipal one is.
- **Choice: identity fallback calibration (`scale=1, offset=0`) for the 16 uncalibrated districts.**
  The alternative (`mode='pooled'`, the mean pilot scale) is exposed as a sensitivity knob and
  explicitly documented as *not validated*, because the two pilot scales diverge by 30× (0.43 vs
  0.012) and averaging them is meaningless. The `offset` term is an **absolute-kW** quantity fitted
  to a small Gemeinde fleet — it cannot transfer to a Landkreis, so any fallback forces it to zero.
  Identity means "trust the physics-informed CF × nameplate", which is the honest default.
- Ships a `__main__` self-check asserting 18 districts, a plausible state total, and identity-by-
  default.

---

### 3.2 `src/features/` — the feature contract

#### `schema.py` (41 lines)
See **D7**. Nine features: 2 physical priors + 3 weather intensities + 4 cyclical time terms.
`tft_known_reals(tech)` returns the technology's *own* prior plus weather — the TFT decoder gets one
prior, not both, which is why the ablation baseline for TFT is defined against that list and not
against `FEATURE_COLS` (`ablation.py:86-94`).

**Why cyclical (`sin`/`cos`) encodings.** Hour 23 and hour 0 are adjacent; a raw integer makes them
maximally distant. The sin/cos pair embeds the circle so that adjacency is preserved and both trees
and the sequence models can express "near midnight" as a single region.

**Why only three weather variables in `X`.** Wind speed and irradiance are *not* features — they
have already been consumed by the physics. Feeding them again would duplicate the signal in a form
the model must re-learn without the power curve. Temperature, pressure and humidity enter directly
because they modulate output through air density and cell efficiency, which the priors approximate
only crudely.

#### `geospatial_clustering.py` (317 lines)
1,093 clusters from 178k assets.

- **Choice: DBSCAN with the haversine metric, `eps = radius_km / R_earth`, on radians.** Density-
  based, so it discovers wind farms (which *are* density clusters) rather than imposing a *k*.
  `min_samples=1` guarantees every asset lands in some cluster — no asset, and therefore no
  capacity, is ever discarded as noise. That is a hard requirement: the priors are capacity sums.
- **Choice: capacity-weighted centroids.** The centroid should represent where the *power* is, not
  where the *units* are; a cluster of one 6 MW turbine and twenty 100 kW units must not be centred
  on the small ones.
- **Choice: dual-track solar.** Ground-mounted (utility) assets use DBSCAN at a tighter 2 km radius
  — they are genuine spatial clusters. Distributed rooftop assets use a static 0.1° grid
  aggregation. Rationale: 174k rooftop units are effectively a continuous density field; DBSCAN on
  them would either merge the whole state into one blob or produce tens of thousands of clusters,
  and neither is useful. The grid also aligns exactly with the weather-node snapping, so
  distributed capacity maps 1:1 onto its weather cell with no interpolation. Result: 455 utility +
  427 distributed.
- **Choice: 3 km for wind.** Balances API cost against micro-meteorological variation; turbines
  within 3 km share a weather regime at Open-Meteo's resolution.

#### `solar_geometry.py` (228 lines)
Pre-computes apparent solar position and Ineichen clear-sky GHI per solar cluster at 15-minute
resolution, via `pvlib`, in a `ProcessPoolExecutor`.

- **Choice: precompute, don't compute inline.** Solar geometry is *deterministic* — a pure function
  of (lat, lon, time). Computing it once and caching to parquet turns a repeated expensive
  computation into a join.
- **Choice: process-based parallelism.** `pvlib` position calculation is CPU-bound NumPy; threads
  would contend on the GIL. `cpu_count() - 1` leaves the machine usable.
- **Choice: worker-level exception capture returning `(id, ok, msg)`.** One malformed cluster must
  not kill the pool.
- **Honest note:** the clear-sky output is currently joined in the solar-prior path but the flat
  `GHI/1000 × PR` formulation does not consume `ghi_clearsky` or the zenith angle. It is
  precomputed infrastructure for a clear-sky-index feature that the final feature set does not use.
  If asked why: adding a clear-sky index would have duplicated information already carried by
  `solar_prior_cf` and the cyclical time terms, and the ablation gives no evidence it was needed.

#### `feature_pipeline.py` (299 lines)
Assembles `ml_training_matrix.parquet`. The single most defense-relevant file after the splitter.

Order of operations: load + downsample targets to hourly → wind prior (per cluster, per-unit curve
× nameplate, summed) → solar prior (STC-normalised × PR, summed) → spatial-mean weather → cyclical
time → **CF normalisation of both priors and both targets** → persist
`cf_normalization.json` → `dropna` → parquet.

- **Choice: `resample('1h').mean()` on the 15-minute target.** Weather is hourly; a 15-minute target
  against hourly features would model quarter-hour variation the features cannot see. Mean, not
  last, because generation is a power average over the interval.
- **Choice: `join(..., how='inner')` at each stage, with row counts logged.** An inner join is the
  correct semantics (a row is only usable if target *and* weather *and* prior exist), and logging
  the count after each join is how a silent join failure is caught — a mismatched timezone shows up
  immediately as a collapse to zero rows.
- **Choice: mode switch (`reanalysis` | `forecast`) sharing all code except the weather directory.**
  The two matrices are identical in target and timestamps and differ *only* in the weather driving
  the priors and means. That is the experimental control for the NWP-gap study, established in the
  data layer rather than argued for after the fact.
- **Choice: only the reanalysis run writes `cf_normalization.json`.** The forecast run must not
  clobber the canonical normalisation constants the municipal validators depend on. A one-line
  guard preventing a whole class of silent corruption.
- `_effective_capacity()` — see **D2**.

#### `validation_splitter.py` (201 lines)
See **D4**. Also ships `visualize_splits()`, which prints per-fold train/gap/test boundaries, sample
counts, an explicit overlap assertion and an ASCII timeline. That output is the source of the
`results/cv_split_table.{csv,md}` and `cv_split_timeline.png` artifacts — **the protocol is
evidenced, not merely asserted.** Raises rather than silently shrinking folds when the dataset is
too short for the requested configuration.

---

### 3.3 `src/models/` — physics and learners

#### `wind_physics_transformer.py` (381 lines)
`WindPowerCalculator`: hub-height extrapolation + manufacturer power-curve matching + interpolation.

- **Choice: Hellmann power law with α = 0.14, extrapolating from 100 m** (not from 10 m).
  α = 0.14 is the neutral-stability open-terrain value. Extrapolating from the 100 m field rather
  than 10 m minimises the extrapolation distance — hub heights cluster near 100–140 m, so the
  correction factor is small (`(150/100)^0.14 = 1.058`) and the error contributed by a
  mis-specified α is correspondingly small. Had we extrapolated from 10 m, α would dominate.
  The value is read from config, so it is a tunable, not a magic number.
- **Choice: a scored multi-tier fuzzy matcher for turbine → power curve.** MaStR type designations
  are free text ("V90-2MW", "E-82 E2", vendor typos included). The scorer combines manufacturer
  match (+10), capacity within 15% (graded up to +5), rotor diameter within 10% (graded up to +5)
  and name-substring containment (+8), searching the manufacturer-filtered pool first and widening
  to the full database if the best score is poor (< 15). Final fallback: nearest rated capacity,
  with a warning. **Why not exact matching:** it would drop most of the fleet. **Why not
  a single generic curve:** it would erase the technology mix, which is the point of a bottom-up
  prior. The graded scoring makes near-misses degrade smoothly instead of falling off a cliff.
- **Choice: `interp1d` with `bounds_error=False, fill_value=0.0`.** Outside the tabulated speed
  range the output is zero — which is physically right at both ends: below cut-in there is no
  power, above cut-out the turbine shuts down. The cut-out behaviour is asserted in the unit tests
  (26 m/s → 0 kW).
- **Choice: hard clip to `[0, gross_power]`.** A power curve interpolation must never exceed
  nameplate; the fallback caps at the curve's rated power when nameplate is missing.
- Ships four `unittest` cases covering zero wind, rated wind, cut-out, and the extrapolation
  arithmetic (10 m/s @100 m → 10.584 m/s @150 m). This is the only module with a formal test
  class; everything else uses `__main__` self-checks. It earns that because it is the one place
  where a silent physics error would be invisible downstream — a wrong power curve produces a
  plausible-looking prior.

#### `train_lightgbm.py` (122 lines) — the production model
Three quantile fits per technology under purged CV; production models refit on the full dataset.

- **Choice: q50 is the point forecast** — see **D6**.
- **Choice: `quantile_model_path()` keeps q50 at the canonical filename** `models/lightgbm_{tech}.pkl`
  with q10/q90 as sidecars. This means SHAP, the NWP-gap study and the tree validators load the
  point model unchanged and know nothing about quantiles — the interval is an additive capability,
  not a refactor that ripples through the codebase.
- **Choice: deliberately regularised parameters** (`max_depth=7`, `num_leaves=31`,
  `min_child_samples=60`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`,
  `learning_rate=0.03`, 600 trees). The stated reason is downscaling, not macro accuracy: a
  heavily-fitted tree ensemble makes confident, high-variance predictions when the municipal
  feature distribution sits off the training manifold. Shrinking predictions toward the training
  mean reduces the systematic bias the affine calibration then has to absorb.
- **Choice: per-row sort of the three quantiles.** Independently-fitted quantiles can cross;
  sorting is the cheapest valid repair and preserves the marginal calibration of each.
- **Choice: metrics persisted at both CF and MW scale, plus PICP and normalised MPIW.** Coverage
  without sharpness is meaningless (a `[0, ∞)` interval has PICP 1.0), so they are always reported
  as a pair.

#### `train_comparisons.py` (83 lines) / `train_ebm.py` (69 lines)
XGBoost (a second GBDT family, to show the result is not a LightGBM artifact) and the Explainable
Boosting Machine (glass-box GA²M, for RQ2 attribution). `train_ebm.py` additionally exports global
per-feature importances to CSV.

- **Choice: identical splitter, identical target, identical clipping across all trainers.** The
  model comparison is only meaningful if the protocol is byte-identical; this is why every trainer
  imports the splitter and `schema` rather than configuring itself.
- **Choice: EBM at defaults.** A glass-box model tuned to match a black box stops being an honest
  interpretability reference.
- Note: `ebm_{tech}.pkl` is written by both files; the last run wins. Harmless (same
  configuration, same seed) but worth knowing if asked which file produced the shipped artifact.

#### `train_bilstm.py` (196 lines)
2-layer bidirectional LSTM, 64 hidden units, 24-step input window, Adam, MSE, early stopping.

- **Choice: `StandardScaler` on features *and* target, persisted to `models/bilstm_scalers.pkl`.**
  Neural nets need standardised inputs. The critical point for the thesis: because the target is
  now a *capacity factor*, the scaler's offset is physically meaningful at any scale (macro CF mean
  ≈ municipal CF mean). Under the old MW target the scaler injected a multi-gigawatt offset into
  municipal inference, which is what made local predictions anti-correlate. The scalers must ship
  with the weights or municipal inference is impossible — hence they are a first-class artifact.
- **Choice: early-stopping validation from the last 10% of the *train* window.** See D4.
- **Choice: seeded per `(technology, fold)`, not once per run.** Unseeded, BiLSTM solar moved
  0.2751 → 0.3055 between runs — **larger than several of the between-model gaps the ranking
  reports.** Seeding once at entry made a fold's draw depend on everything trained before it.
  `seed` is an exposed parameter *specifically so the spread can be measured across replicates*
  (`results/seed_variance.csv`). The in-code comment is explicit that this pins one draw for
  reproducibility; it does not make the ranking stable. **Have this ready — it is the single most
  effective way to demonstrate methodological maturity.**
- **Choice: careful target alignment.** `TimeSeriesDataset` emits the target at `idx + SEQ_LEN`,
  so the evaluation slices `y`, `cap` and `y_mw` with `[SEQ_LEN:]`. An off-by-24 here would produce
  quietly wrong metrics.
- **Choice: `feature_cols` / `model_name` / `out_csv` / `save_production` parameters.** These exist
  solely so the ablation driver can retrain a deliberately crippled model without (a) colliding
  with production metric rows or (b) overwriting production weights. A safety interlock, not
  generality for its own sake.

#### `train_tft.py` (193 lines)
Temporal Fusion Transformer via `pytorch-forecasting`: 72-hour encoder, 24-hour decoder,
`GroupNormalizer` target, RMSE loss, hidden 32, 4 attention heads.

- **Choice: TFT is the state-of-the-art comparison and the attention-based interpretability
  reference.** Its `interpret_output` produces attention and per-variable importance figures
  written straight to `documentation/` from inside the trainer.
- **Choice: explicit `ModelCheckpoint(monitor="val_loss", save_top_k=1)`.** Lightning's default
  saves the *last* epoch, which early stopping leaves 3 epochs past the best. Without this the
  evaluated model is not the selected model — a subtle and very common reporting error.
- **Choice: encoder-history-aware fold construction.** `df_val` and `df_test` are extended
  backwards by `MAX_ENCODER_LENGTH`, otherwise the first prediction of each window has no legal
  context. The purge gap (72 h) is exactly the encoder length, so the encoder context never reaches
  back into the training window.
- **Choice: OOF export takes only the h=24 step.** The TFT emits 24 overlapping decoder windows per
  hour; averaging them would hand it a free ensembling advantage the single-shot models do not
  have. Exporting only the h=24 step gives one clean value per timestamp at exactly the day-ahead
  horizon the thesis targets. **This is why the TFT's per-point nMAE differs slightly from its CV
  table entry**, which averages horizons 1–24 — say so before it is noticed.
- **Choice: seeded per `(technology, fold)`,** same reasoning as BiLSTM. Unseeded, the TFT moved
  5–12% between runs.

#### `explain_models.py` (60 lines) / `explain_bilstm.py` (90 lines)
`shap.TreeExplainer` on the production LightGBM q50 models; `shap.DeepExplainer` (DeepLIFT) on the
BiLSTM.

- **Choice: named DataFrames all the way through the model layer.** SHAP and `interpret` bind
  attributions to column names. This is why the codebase forbids PCA or any unlabeled
  dimensionality reduction before the model — it would make RQ2 unanswerable by attribution.
- **Choice: BiLSTM sequences built from the *last purged CV fold* with the *persisted production
  scalers*.** Explaining a model on data it trained on, or with a different scaler, is not an
  explanation of the shipped model.
- **Choice: SHAP averaged over the 24-step sequence dimension** to yield one importance row per
  feature per sample, making the plot directly comparable to the tree beeswarms.
- **Honest limitation:** SHAP is *correlational* attribution. It cannot distinguish "this feature
  carries the signal" from "this feature correlates with what carries the signal". That gap is
  exactly why `ablation.py` exists (§3.4).

---

### 3.4 `src/evaluation/` — the evidence layer

#### `metrics.py` (222 lines)
Every metric, every baseline, and the two persistence functions the whole system writes through.

- **Choice: every metric is nan-safe via a shared `_clean` mask.** Municipal telemetry has gaps;
  a metric that silently propagates NaN, or that drops different rows for different models, makes
  comparisons invalid.
- **Choice: normalisation by `mean(|y|)`, not by capacity or by range.** nMAE is then directly
  comparable between the macro MW scale and the municipal kW scale — which is the entire point of
  a cross-scale thesis. Normalising by installed capacity would embed the very capacity-accounting
  question that §7 T2 shows is unresolved.
- **Choice: MBE is a first-class metric.** For downscaling, systematic bias is *the* failure mode;
  MAE alone would hide it.
- **Choice: the naive baselines live here and are run under the same splitter as the models.**
  24-hour persistence and hour×month climatology, with climatology fitted on the training slice
  only. If the baselines were computed anywhere else, or on a different split, "ML earns its place"
  would be an unfalsifiable claim.
- **Choice: `append_cv_metrics` de-duplicates on `(model, technology, fold, scale)` keep-last.**
  Re-running one trainer refreshes only its own rows; the results table is incrementally
  reproducible rather than all-or-nothing.
- **Choice: `append_oof` merges rather than overwrites,** de-duplicating on
  `(technology, model, timestamp)`. This is what lets the cheap tree models be exported in one pass
  while BiLSTM and TFT append from inside their own trainers — refitting a TFT purely to export
  predictions would be hours of GPU time for no information.
- `rolling_metrics()` — the sliding-window harness that replaced the original single hardcoded
  validation window.
- Ships a `__main__` self-check with real assertions: perfect prediction → zero error; persistence
  tracks a daily cycle; climatology beats a global mean on a seasonal signal; a `[q10, q90]`
  interval on a normal sample gives PICP ≈ 0.80 and is wider than `[q25, q75]`.

#### `baselines_macro.py` (64 lines)
Runs persistence and climatology through the identical splitter and writes them into the *same*
`macro_cv_metrics.csv` as the models.

- **Choice: climatology's per-point predictions are exported to the OOF parquet, persistence's are
  not.** Climatology must be fitted per fold on that fold's training slice; pooling OOF rows and
  averaging afterwards would let later folds leak into earlier folds' hour×month means.
  Persistence is a pure lag and can be derived on the fly in `significance.py`. A three-line
  distinction that prevents a real leak.

#### `export_predictions.py` (123 lines)
Re-runs the purged CV to produce genuinely out-of-fold per-point predictions for the tree models.

- **Choice: this file exists because the persisted `models/*.pkl` are refit on the *full* dataset.**
  Predicting with them is in-sample and would flatter every scatter plot and time-series overlay in
  Chapter 6. Every visualisation in the thesis that shows "predicted vs actual" is fed from this
  parquet, not from the production models.
- **Choice: model builders are *imported* from the trainers, not copied.** `_builders()` pulls
  `LGB_PARAMS` and `XGB_PARAMS` from the trainer modules so the exported OOF model cannot drift
  from production. This was a real bug: omitting the quantile objective made LightGBM fall back to
  its default L2 — a different model reported under the same name.
- **Choice: a runtime assertion that the exported LightGBM point forecast reproduces the trainer's
  q50 fit exactly.** Both are deterministic, so equality is the correct test, and it is compared
  against the raw q50 rather than the sorted median.
- **Choice: LightGBM's held-out q10/q90 are exported too.** These are what macro conformal
  calibrates on; point models leave the columns NaN.
- Self-check asserts OOF timestamps are unique per model per technology (i.e. each point is held
  out exactly once) and that ML beats the raw physics prior.

#### `significance.py` (181 lines)
Diebold-Mariano with Newey-West HAC variance, plus a moving-block bootstrap CI on the nMAE gap,
plus the fold-level paired summary.

- **Choice: the module states its own motivation — the CV table supports a *ranking* but not a
  *claim of difference*.** With 4 folds, a perfect sweep gives sign-test p = 0.125. This is written
  into the docstring. **Quoting your own code's limitation section back to a committee is the
  strongest possible position.**
- **Choice: Newey-West / Bartlett-weighted HAC variance, lag `⌊n^(1/3)⌋`.** Hourly renewable
  residuals are strongly autocorrelated; an iid variance overstates significance by roughly the
  square root of the autocorrelation time. Bartlett weights keep the estimate positive
  semi-definite.
- **Choice: 7-day (168 h) moving blocks for the bootstrap.** Long enough to carry synoptic weather
  autocorrelation. Both models are resampled on the *same* block draw — that is what makes the
  comparison paired.
- **Choice: the bootstrap is the arbiter if the two disagree.** DM leans on an asymptotic normal
  approximation the bootstrap does not need. Stated explicitly in the docstring.
- **Choice: relative effect size (`dnmae_rel_%`) is reported next to every p-value.** On ~35k
  hourly observations, a statistically significant gap can be practically irrelevant. This is the
  distinction the Chapter 6 ranking claims actually turn on.
- **Choice: persistence is derived here and included in the comparison matrix,** so "ML beats the
  naive baseline" gets the same statistical treatment as the model-vs-model claims.
- Self-check: a model against a deliberately degraded copy of itself must come out significantly
  better with a CI excluding zero *in the right direction* (guards a sign flip); a model against
  itself must give an identically zero loss differential.

#### `conformal.py` (190 lines)
The width-normalised split-CQR primitives (`fit_conformal`, `apply_conformal`, `append_coverage`).
See **D6** for the rationale. Five `__main__` fixtures, all non-negative and generation-shaped
because `apply_conformal` floors at zero: (1) too-narrow intervals widen to ≈ 0.80 on a disjoint
split; (2) too-wide intervals *shrink* (Q < 0); (3) the heteroscedastic PV case — night hours must
not be inflated; (4) the degenerate-information case must select the **additive** regime; (5) the
all-zero night path is a no-op, not a NaN.

`w_floor` exists because the score is `0/0` when `hi == lo == y == 0` (PV at night) and `+∞` when
the actual is non-zero — a single such hour would poison the quantile. It is derived from the
calibration data and persisted *alongside* Q, because applying a different floor than the one
fitted silently changes the correction.

#### `conformal_macro.py` (74 lines)
Macro-scale application: **fit Q on fold *f*, evaluate on fold *f+1*.**

- **Choice: strictly sequential folds.** Fitting and evaluating on the same fold is in-sample;
  calibrating on a chronologically *later* fold reintroduces exactly the look-ahead leakage the
  purged protocol exists to prevent. That leaves n−1 = 3 honest evaluations per technology.
- **Why this module matters for the defense:** it proves the interval under-coverage is *not*
  purely a downscaling artefact — macro wind PICP is 0.62 against an 0.80 nominal. Without this
  file, the municipal coverage collapse would look like evidence against the downscaling contract.
- Self-check asserts conformal moves mean `|PICP − 0.80|` *toward* nominal; if it did not, the
  calibration fold would not be informative about the next and the sequential scheme would be
  invalid here.

#### `ablation.py` (195 lines) — the causal answer to RQ2
Four variants under the identical purged CV: `full`, `no_prior`, `prior_only`, and
`physics_prior_raw` (the prior used directly as the forecast, no learner).

- **Choice: ablation over attribution.** The docstring makes the argument: SHAP/EBM/attention
  report how a model *internally* accounts for its inputs — correlational. A feature can dominate
  an attribution ranking and still be replaceable by its correlates. Retraining without it and
  measuring the error penalty is the causal, falsifiable version of the same claim.
- **Choice: `prior_only` earns its runs.** With `no_prior` alone, a reviewer can object that the
  weather features are simply redundant with the prior, so removing it proves nothing about the
  prior itself. The three variants together decompose the skill.
- **Choice: `physics_prior_raw` as a learner-free reference.** `prior_only` still fits a model *on*
  the prior. Without the raw row the table cannot separate "the prior carries the signal" from "the
  learner corrects the prior".
- **Choice: TFT gets its own variant map.** Its `full` baseline must be `tft_known_reals` (own prior
  only), not `FEATURE_COLS`, or the penalty would be measured against a model the thesis never
  trained.
- **Choice: BiLSTM has no `prior_only` row.** It builds one feature matrix for both technologies in
  a single call and cannot express a technology-specific `prior_only`. Rather than redefine the
  variant to mean "both priors" for one model only — making the column mean two different things
  in the same table — the row is simply absent. **An omission with a stated reason beats a filled
  cell with a hidden one.**
- **Choice: the self-check asserts the RQ2 narrative.** If dropping the prior ever *improves* nMAE,
  the run fails loudly rather than quietly printing a surprising table.

#### `nwp_gap.py` (200 lines)
Quantifies the train/deploy discrepancy: reanalysis-driven vs NWP-driven features.

- **Choice: per-fold fresh models, predicting *both* feature sets on the held-out window.** Reusing
  the production model (trained on the full matrix) made the reanalysis side in-sample and
  **overstated** the gap. The current design means every compared prediction is out-of-fold.
- **Choice: the two matrices share target and timestamps exactly.** The only varying quantity is
  the weather feeding the priors and means, so the delta is attributable purely to weather-forecast
  uncertainty — not to a different model, target or period. This control was built in the data
  layer (D3 / `weather_ingestion` modes), which is why the claim is clean.
- **Choice: the historical-forecast archive, not a live forecast endpoint.** Archived NWP model runs
  align with historical timestamps; live forecasts do not.
- Ships `demo()`: on synthetic data a noisier "forecast" input must not beat the clean input.

---

### 3.5 `src/validation/` — cross-scale evidence

#### `kw_features.py` (219 lines) — the most load-bearing file in the downscaling story
Builds the municipal feature matrix using the **identical contract** as training, plus
`calibrate_affine` / `apply_affine`.

- **Choice: `assemble_municipal_features` takes a `load_node_weather` callable.** The same function
  builds features from the stored archive parquet (validation) *and* from the live forecast API
  (dashboard). **Identical physics on both paths is what keeps the frozen affine calibration valid
  at inference time.** If validation and serving built features differently, every persisted
  calibration would be meaningless. This is dependency injection used for a correctness reason, not
  for testability aesthetics.
- **Choice: priors normalised against *local* nameplate.** `wind_prior_cf = local_prior_mw /
  local_nameplate_mw`. This is the mechanical realisation of scale invariance: the same column name,
  the same range, a different fleet.
- **Choice: 8 km radius fleet definition.** A compromise between "the assets the E.ON meter plausibly
  observes" and "enough assets for the prior to be stable". It is also the single largest source of
  the wind-scale divergence (§7, T2) — the radius is a proxy for a metering boundary that is not
  published.
- **Choice: per-asset wind physics at municipal scale** (real hub heights and rotor diameters from
  MaStR, defaulting to 100 m when missing), rather than the representative-turbine shortcut used at
  macro scale. At macro scale, 4,147 turbines average out; at municipal scale a handful of specific
  machines determine the output.
- **Choice: the E.ON scrape is treated as UTC and never converted.** This was verified empirically —
  PV and wind cross-correlate with UTC weather at 0–1 h lag, not the +2 h a Berlin-local reading
  would imply. It is documented in the module docstring, in `CLAUDE.md` and in project memory,
  because a two-hour shift would silently degrade every municipal metric while leaving everything
  looking plausible. **If asked how you know: the cross-correlation lag test.**
- `calibrate_affine` — see **D5**. Falls back to offset-only (`scale = 1`) when the base is
  degenerate/constant or OLS returns a non-positive slope, so MBE is still zeroed sensibly.
- Self-check recovers a known affine relation (scale 2.0, offset 300) and asserts fit-split MBE ≈ 0.

#### `municipal_validator.py` (234 lines)
The cross-scale experiment: LightGBM (with interval), XGBoost, EBM, BiLSTM and a municipal
persistence baseline, per municipality.

- **Choice: 40% calibration / 60% validation of the telemetry overlap, chronologically ordered.**
  No hardcoded window. All reported municipal numbers are on the held-out 60%.
- **Choice: `nmae_k1` / `mbe_k1` columns record the *uncalibrated* performance alongside the
  calibrated one.** This is what makes the calibration's contribution measurable rather than
  assumed — and it is where the headline RQ1 numbers come from (KW solar: 17.6 → 0.161).
- **Choice: the affine fitted on the median is applied *unchanged* to q10 and q90.** A positive
  slope preserves ordering, so the band cannot invert. It also makes explicit that the affine step
  rescales width without re-deriving it — which is the diagnosis conformal then treats.
- **Choice: conformal is fitted on the *same* calibration split that already fits the affine.**
  No additional data is consumed and the validation split stays untouched.
- **Choice: a municipal persistence baseline on the same split.** Without it, a municipal nMAE of
  0.32 has no reference. (Persistence: 0.69.)
- **Choice: calibration params are merged, not overwritten, into
  `results/<slug>/affine_calibration.json`.** The tree/BiLSTM validator and the TFT validator each
  write their own keys into one file.

#### `tft_municipal_validator.py` (121 lines)
Separate file because `pytorch-forecasting` inference needs its own dataset construction
(`from_parameters`, dummy unknown-reals, decoder index arithmetic) that would have made the main
validator substantially harder to read.

- **Choice: `preds[:, 23]` with `time_idx + 23`.** `x_to_index` returns the time index of the
  *first* decoder step, so the 24-step horizon covers `[t, t+23]`. An off-by-one here silently
  shifts every TFT municipal prediction by an hour.
- **Choice: merges into the existing metrics CSV and parquet, replacing only `model == 'tft'`
  rows.** Re-running the TFT validator does not invalidate the other models' results.

#### `lomo_transfer.py` (154 lines) — the honesty check on D5
Leave-one-municipality-out: fit `(scale, offset)` at one town, apply verbatim at the other, both
directions. Three variants scored on the **same** window (the target's held-out split):
`oracle` (target's own fit — the §6.5 number), `transferred` (donor's frozen params — what the 16
uncalibrated districts effectively get), `uncalibrated` (scale 1, offset 0).

- **Why this file exists.** §6.5 validates each town with a calibration fitted on that town's own
  telemetry. The 16 unvalidated districts have no telemetry at all. The operationally honest
  question — *does the contract travel?* — has to be asked, and the answer is largely "no, for
  wind". Asking it yourself is worth far more than having it asked.
- **Choice: the key diagnostic is asserted, not claimed.** An affine map with positive slope cannot
  change Pearson `r`, so `corr` is identical across all three variants by construction (checked
  only where the zero-floor never activated, since clipping is non-linear and legitimately does move
  `corr`). **Therefore any degradation under transfer is magnitude — capacity accounting — never
  shape — forecast skill.** That single assertion is the empirical backing for the entire §7.4
  argument.
- **Choice: `clipped_frac` is reported.** It marks the rows where the zero-floor activated and the
  invariance argument does not apply.

#### `extend_kw_weather.py` (118 lines)
Extends archived weather nodes forward so municipal telemetry overlap grows from days to months.

- **Why it exists at all.** `weather_ingestion`'s `completed_nodes` state is binary — once a node is
  fetched it is never revisited, so a nightly re-run of the batch ingester is a permanent no-op and
  the archive silently stops growing. This module extends from each node's **own max timestamp**
  instead. That is a genuine architectural gap in the ingester, closed by a separate tool rather
  than by making the ingester stateful in a more complicated way.
- **Choice: drop all-NaN trailing rows.** ERA5(T) lags real time by days and the API pads recent
  hours with nulls. Keeping them would advance the node's max timestamp past real data, and those
  hours would be skipped forever.
- **Choice: `--all` mode catches `RequestException` per node.** One flaky call must not abort the
  remaining 429 nodes; a skipped node is simply retried next run because it extends from its own
  max. This is the cron entrypoint (`run_pipeline.sh`).

---

### 3.6 `src/measurements/` and `src/dashboard/`

#### `measurements/scraper_service.py` (181 lines)
Polls the E.ON Energiemonitor API per configured region at 15-minute intervals, appending to CSV.

- **Choice: append-only CSV, never rewrite.** This is the *only* ground truth in the entire thesis
  and it is not obtainable retroactively — the API serves current values, not history. An
  append-only file is the most robust possible store for an irreplaceable stream.
- **Choice: parse the interval-*end* timestamp from the payload, falling back to wall-clock.**
  Scrape time ≠ measurement time; using wall-clock unconditionally would smear the series by the
  polling jitter.
- **Choice: degrade gracefully when the weather endpoint fails but the meter endpoint succeeds.**
  Losing a cloud-cover reading must not cost a generation reading.
- **Choice: header self-repair.** Detects the legacy header (no `timestamp` column) and rewrites it.
  A migration artifact, kept because the file predates the timestamp column.

#### `dashboard/forecast_service.py` (184 lines)
The operational inference path: live NWP → scale-free features → CF quantiles → calibrated MW.

- **Choice: it reuses `municipal_fleet` and `assemble_municipal_features` unchanged.** See
  `kw_features.py`. The serving path and the validation path are the same code.
- **Choice: `forecast_from_fleet(fleet, calib_provider, ...)` — one core, two callers.** The
  municipal path passes the frozen per-town calibration; the district path passes
  `resolve_calibration` (identity fallback). The physics, the feature construction and the
  quantile inference are shared; only the calibration source differs. That is the correct seam.
- **Choice: exponential backoff on 429/5xx.** The state-wide loop makes 18 sequential batched calls
  to a free API; without backoff the later districts fail.
- **Choice: the frozen conformal correction is applied at serving time where one exists,** with
  `lower = min(lower, median)` / `upper = max(upper, median)` so a negative Q can shrink the band
  without ever inverting it. Districts with no fitted calibration have no Q and fall through
  unchanged — an uncalibrated district gets an honest raw interval, not a fake corrected one.
- **Choice: only the five weather variables the physics actually consumes are requested.**
  Smaller payload, faster response, and it documents the true input dependency.
- Network-free self-check: synthetic weather → assert band ordering and non-negativity.

---

## 4. Root entrypoints

| Entrypoint | Invocation | What it does | Design note |
|---|---|---|---|
| `main.py` | `uv run main.py` | Full reanalysis pipeline: boundary → MaStR parse → clustering → solar geometry → target harmonisation → weather ingestion → feature matrix → **baselines** → 4 trainers → municipal validation (trees/BiLSTM, then TFT) | Each stage is skip-if-exists where the output is expensive and immutable (boundary, parsed MaStR). Baselines run **before** the models — the bar is set before it is cleared. |
| `main.py --mode forecast` | `uv run main.py --mode forecast` | NWP branch only: forecast weather ingestion + forecast feature matrix | Raises immediately if the reanalysis outputs are missing. Deliberately does *not* retrain: models are trained on reanalysis and applied unchanged, which is what makes the NWP gap measurable. |
| `dashboard.py` | `uv run streamlit run dashboard.py` | 18 Kreise + state aggregate, 24 h wind/solar bands, choropleth, skill panel | Serves the pre-computed batch parquet when present, else runs live NWP per district. All data stays UTC; `Europe/Berlin` is applied at render only. The aggregate band is the sum of district P10/P90 — the **comonotonic** bound, disclosed in the UI as conservative because spatial errors partly decorrelate. |
| `run_scraper.py` | `uv run run_scraper.py [--loop]` | E.ON telemetry collection | The one Hydra-configured entrypoint (historical). `--loop` is stripped from `sys.argv` before Hydra parses it. Loop mode compensates for scrape duration so cadence does not drift. |
| `run_pipeline.sh` | cron, nightly | `extend_kw_weather --all` with logging | Exists because the batch ingester cannot extend (see §3.5). |
| `conf/config.yaml` | — | The single config: MaStR URLs, Open-Meteo endpoints + variable list, physics constants (α, PR, temp. coefficient), scraper regions with lat/lon and result dirs, the 18-district registry | No Hydra composition inside `src/`; one flat file, read by a 20-line loader. Physics constants live here so α and PR are auditable and tunable, not buried. |

---

## 5. On-disk contracts — what is the source of truth for what

| Artifact | Written by | Read by | Contract |
|---|---|---|---|
| `data/raw/mastr_*_brandenburg_raw.csv` | `mastr_bulk_parser` | clustering, `kw_features`, `districts`, `extend_kw_weather` | **Read-only after parsing.** Contains only status-35 (operating) units inside the boundary polygon — so no downstream consumer filters by status again. |
| `data/external/brandenburg_boundary.geojson` | `boundary_fetcher` | parser (spatial mask, centroid fallback) | Fetched once, cached. |
| `data/external/brandenburg_districts.geojson` | `districts` | dashboard, notebook 11 | 18 polygons; a partial cache raises rather than silently dropping a district. |
| `data/processed/{wind,solar}_clusters.csv` | `geospatial_clustering` | feature pipeline, weather ingestion, solar geometry | `cluster_id`, capacity-weighted centroid, `total_capacity_mw`, `asset_count`. **Fleet nameplate for CF normalisation is the sum of `total_capacity_mw`.** |
| `data/processed/weather/grid_nodes/node_<lat>_<lon>.parquet` | `weather_ingestion`, extended by `extend_kw_weather` | feature pipeline, municipal validators | 430 nodes on a 0.1° grid, hourly, **UTC**, snappy-compressed. |
| `data/processed/weather_forecast/grid_nodes/…` | `weather_ingestion --mode forecast` | forecast feature matrix → NWP gap | Same schema, archived-NWP source. Isolated directory so the two can never be confused. |
| `…/mapping_index.json` | `weather_ingestion` | feature pipeline | cluster → grid node → parquet path. |
| `data/processed/solar_geometry/<cluster>.parquet` | `solar_geometry` | feature pipeline (joined) | zenith, azimuth, clear-sky GHI, 15-min. |
| `data/processed/actual_generation_50hertz.parquet` | `target_harmonization` | feature pipeline, splitter demo | UTC, 15-min, two MW columns. |
| **`data/processed/ml_training_matrix.parquet`** | `feature_pipeline` | **every trainer, every ablation, SHAP, NWP gap** | 37,967 hourly rows, 2022-01-31 → 2026-05-31, 17 columns. 9 model features + 2 MW targets + 2 CF targets + 2 effective-capacity columns + 2 raw-MW priors. |
| `data/processed/ml_training_matrix_forecast.parquet` | `feature_pipeline --mode forecast` | `nwp_gap` only | Identical target and timestamps; only weather-derived columns differ. |
| `data/processed/cf_normalization.json` | `feature_pipeline` (**reanalysis mode only**) | municipal validators, reproducibility | Fleet nameplates, per-year effective capacity, PR and reference-turbine constants. **The override hook for official 50Hertz capacity figures.** |
| `models/lightgbm_{tech}.pkl` | `train_lightgbm` | SHAP, validators, NWP gap, dashboard | The **q50** model. Canonical filename so consumers never learn about quantiles. |
| `models/lightgbm_{tech}_q{10,90}.pkl` | `train_lightgbm` | validators, dashboard | Interval sidecars. |
| `models/bilstm_{tech}.pth` + `bilstm_scalers.pkl` | `train_bilstm` | municipal validator, DeepSHAP | **The scalers are not optional** — without them municipal inference is impossible. |
| `models/tft/tft_{tech}_production.ckpt` | `train_tft` | TFT municipal validator | Full Lightning checkpoint (carries `dataset_parameters`). |
| `results/macro_cv_metrics.csv` | all trainers + `baselines_macro`, via `append_cv_metrics` | Ch. 6 tables, `significance`, dashboard skill panel | Keyed `(model, technology, fold, scale)`; both CF and MW rows. Incrementally refreshable. |
| `results/macro_oof_predictions.parquet` | `export_predictions` (trees) + `train_bilstm` + `train_tft` + `baselines_macro` (climatology), via `append_oof` | `significance`, `conformal_macro`, every Ch. 6 figure | Keyed `(technology, model, timestamp)`. **All five models must have been run for the significance tests to cover the full ranking.** |
| `results/model_significance.csv` | `significance` | Ch. 6 | Paired DM + bootstrap CI + fold-level view. |
| `results/ablation_metrics.csv` | `ablation` | Ch. 6 (RQ2) | Separate CSV so ablated model names can never collide with production rows. |
| `results/conformal_coverage.csv` | `conformal_macro` + `municipal_validator` | Ch. 6 | Keyed `(scope, municipality, technology, model, fold)`; before/after PICP and width. |
| `results/nwp_gap_report.{csv,md}` | `nwp_gap` | Ch. 6.7 | |
| `results/<slug>/affine_calibration.json` | `municipal_validator` + `tft_municipal_validator` (merged) | `lomo_transfer`, `forecast_service` | **The downscaling contract as a versioned artifact:** per `(model, technology)` → `{scale, offset, cap_mw, conformal_q, conformal_w_floor}`. |
| `results/municipal_validation_metrics.csv`, `results/lomo_transfer.csv` | validators | Ch. 6.5, Ch. 7.4 | |
| `results/forecasts/brandenburg_state_forecast.parquet` | notebook 11 | dashboard | Pre-computed batch for all 18 Kreise; the dashboard falls back to live NWP when absent. |

---

## 6. Defense dossier — the four attack surfaces

### A1 — Leakage and validation rigor

**The claim.** Every reported number is out-of-fold under a purged expanding-window protocol with a
72-hour embargo, and the target definition itself is causal.

**The five leaks that were found and closed** — enumerate these; each one is evidence of rigor, not
of sloppiness:

| Leak | Where | Fix |
|---|---|---|
| Per-calendar-year capacity percentile used future data inside test folds | target definition | causal trailing 365-day p99.9, `shift(1)`, `cummax` (D2) |
| Test fold served as early-stopping `val_loss` | BiLSTM + TFT | validation carved from the last 10% of the *train* window |
| Lightning saved the last epoch, not the best | TFT | explicit `ModelCheckpoint(monitor='val_loss')` |
| Adjacent folds shared a boundary timestamp | splitter | half-open test window |
| OOF exporter silently used L2 instead of the trainer's quantile objective | `export_predictions` | params imported from the trainer + runtime equality assertion |

**Statistical honesty.** 4 folds cannot reach significance (sign-test p = 0.125 for a perfect
sweep), so significance comes from ~35k paired per-point residuals via Diebold-Mariano with
Newey-West HAC variance and a 7-day moving-block bootstrap, with relative effect size reported
alongside every p-value.

**Selected results** (`results/model_significance.csv`, CF scale):

| Comparison | ΔnMAE | rel. | DM p | Reading |
|---|---|---|---|---|
| wind: LightGBM vs TFT | +0.0007 | +0.3% | **0.658** | **Statistically indistinguishable.** The Ch. 6 ranking must not claim otherwise. |
| solar: LightGBM vs TFT | −0.0298 | −14.3% | <1e-4 | Real and material. |
| wind: LightGBM vs persistence | −0.4482 | −69.3% | <1e-4 | ML earns its place. |
| solar: LightGBM vs persistence | −0.1072 | −37.5% | <1e-4 | Same. |
| solar: EBM vs TFT | +0.0036 | +1.7% | 0.094 | Significant gap, negligible size. |

**Seed variance, disclosed unprompted.** Unseeded, the TFT moved 5–12% and BiLSTM solar moved
0.2751 → 0.3055 between runs — larger than several between-model gaps. Both are now seeded per
`(technology, fold)`, and the `seed` parameter is exposed so the spread itself can be measured
(`results/seed_variance.csv`). **Seeding pins one draw for reproducibility; it does not make the
rank stable.** Say this sentence verbatim.

**If asked "why not nested CV / hyperparameter search?"** Hyperparameters were fixed a priori from
standard regularised defaults and *not* tuned on the folds — so the folds remain honest test sets.
Tuning inside the purged protocol would require an inner loop and would consume the limited fold
budget. The regularisation choices are justified by the downscaling argument (D6), not by macro
performance.

---

### A2 — The downscaling contract

**The claim (RQ1).** A capacity-factor target transfers across spatial scales where a megawatt
target cannot; a two-parameter affine map absorbs the residual local accounting.

**The evidence chain:**

1. **Uncalibrated CF downscaling is shape-correct but magnitude-wrong.** At Königs Wusterhausen,
   `CF × nameplate` with no calibration gives wind nMAE 1.03 / solar nMAE 17.6, while `corr` is
   already 0.82 / 0.97. The *forecast* is right; the *accounting* is wrong.
2. **The affine map fixes the accounting.** Calibrated, held-out 60%:

   | Town | Tech | nMAE (LightGBM) | uncalibrated `nmae_k1` | `corr` | persistence |
   |---|---|---|---|---|---|
   | Königs Wusterhausen | wind | **0.322** | 1.032 | 0.818 | 0.687 |
   | Königs Wusterhausen | solar | **0.161** | 17.62 | 0.973 | 0.294 |
   | Nauen | wind | **0.385** | 66.98 | 0.811 | 0.700 |
   | Nauen | solar | **0.337** | 8.48 | 0.889 | 0.319 |

   Three of four beat municipal persistence; Nauen solar (0.337 vs 0.319) does not, and that should
   be conceded rather than glossed.
3. **The intercept is what zeroes the bias, and it is algebra not luck.** OLS with an intercept
   makes fit-split MBE identically zero. A pure-MBE objective is degenerate (scale 0 also achieves
   it). Asserted in `kw_features.py`'s self-check.
4. **The contract does *not* transfer between towns — and the failure is diagnosed, not hidden.**
   `results/lomo_transfer.csv`, nMAE by variant:

   | Tech | Target | oracle | transferred | uncalibrated |
   |---|---|---|---|---|
   | solar | KW | 0.171 | 1.707 | 17.27 |
   | solar | Nauen | 0.342 | 0.531 | 8.34 |
   | wind | KW | 0.334 | 0.965 | 1.011 |
   | wind | Nauen | 0.392 | **29.98** | 66.45 |

   The fitted wind scales differ 30× (KW 0.4595, Nauen 0.0153).
5. **The diagnosis: it is capacity accounting, not forecast skill.** A positive-slope affine map
   cannot change Pearson `r`. `lomo_transfer.py` **asserts** that `corr` is bit-identical across
   oracle / transferred / uncalibrated (wherever the zero-floor never activated). Therefore transfer
   degradation is *purely* magnitude. The physical reading: the MaStR nameplate within an 8 km
   radius of Nauen is roughly 30× what the E.ON meter actually observes — a metering-boundary
   mismatch, not a modelling failure. Solar transfers far better (scale ratio ≈ 1.85 / 0.54)
   because rooftop PV is distributed roughly in proportion to the metered load area, whereas a
   handful of large turbines either are or are not behind the meter.
6. **What the 16 uncalibrated districts therefore get.** Identity calibration — trust the physics.
   `resolve_calibration` refuses to transfer the offset (an absolute-kW term fitted to a Gemeinde
   fleet) and documents `pooled` mode as unvalidated. The dashboard states this in the UI.

**If challenged — "then your state-wide forecast is uncalibrated and unvalidated."** Correct, and
it is labelled as such in the dashboard and in the fallback code. What *is* validated is the shape:
the macro model's CF skill (corr 0.955 / 0.974) and the invariance of `corr` under affine transfer
together mean the district forecasts have the right temporal profile. The magnitude carries the
nameplate-vs-metered-footprint uncertainty quantified at the two pilots. The correct future work is
one telemetry point per district, at which point a single OLS fit per district closes the gap —
which is exactly what `resolve_calibration`'s interface anticipates.

---

### A3 — Physics vs ML

**The claim (RQ2).** The physics prior is load-bearing, and the evidence is causal (retraining
without it), not merely attributional (SHAP).

`results/ablation_metrics.csv` (CF nMAE, mean over 4 purged folds):

| Tech | Model | full | no_prior | prior_only | penalty | raw prior (no learner) | ML gain vs raw |
|---|---|---|---|---|---|---|---|
| wind | LightGBM | 0.1995 | 0.5764 | 0.1915 | **+189%** | 0.2812 | −29.1% |
| wind | TFT | 0.1989 | 0.5623 | 0.1930 | +183% | 0.2812 | −29.3% |
| wind | BiLSTM | 0.2600 | 0.5916 | — | +128% | 0.2812 | −7.6% |
| solar | LightGBM | 0.1799 | 0.2499 | 0.3090 | **+39%** | 0.4194 | −57.1% |
| solar | TFT | 0.1962 | 0.2696 | 0.2066 | +37% | 0.4194 | −53.2% |

**The three readings to have ready:**

1. **Wind is prior-dominated.** Remove the prior and error triples. `prior_only` (0.1915) is
   *marginally better* than `full` (0.1995) — meaning the extra weather and time features
   contribute nothing beyond the prior for wind, and may even add a little noise. The honest
   statement: for wind, the physics *is* the model, and the learner's job is the ~29% correction
   from raw prior to learned output (wake, availability, curtailment, Brandenburg-vs-zone fleet
   composition).
2. **Solar is learner-dominated.** The prior alone is poor (0.309 vs 0.180 full; raw prior 0.419),
   and removing it costs only 39%. The flat `GHI/1000 × PR` model omits POA transposition,
   temperature derating and inverter clipping, so there is a great deal for the learner to fix —
   and it does, cutting error 57% below the raw prior.
3. **The asymmetry is itself the finding.** RQ2 does not have one answer; it has a
   technology-dependent one, and the ablation is what separates the two cases. A SHAP ranking alone
   could not have distinguished "the prior dominates" from "the prior correlates with what
   dominates".

**Supporting attribution** (correlational, and labelled as such): SHAP beeswarms for LightGBM and
XGBoost, EBM global importances and pairwise interaction plots, TFT encoder/decoder variable
importance and attention. All of it built on named columns — which is why PCA is banned upstream.

**If challenged — "α = 0.14 and PR = 0.80 are just constants you picked."** They are standard
literature values (neutral-stability open terrain; generic crystalline-silicon performance ratio),
they live in `conf/config.yaml` as auditable tunables, and — most importantly — **the ablation
bounds their influence**: whatever error they carry is inside the raw-prior number (wind 0.2812),
and the learner demonstrably removes 29% of it. The system is not required to have the right
constants; it is required to have a prior with the right *shape*, and the learner calibrates the
level.

---

### A4 — Operational realism

**The claim.** The system is evaluated as it would be deployed, and the deployment penalty is
measured rather than assumed.

1. **The NWP gap is measured, on an airtight control.** Two feature matrices, identical target and
   timestamps, differing only in the weather that drives the priors and means; per-fold fresh
   models predicting both on held-out windows.

   | Tech | MAE degradation (CF) | RMSE degradation |
   |---|---|---|
   | wind | **+90.9%** | +93.6% |
   | solar | **+76.8%** | +64.2% |

   Over 35,040 out-of-fold hours. **Say this number before you are asked for it.** A reanalysis-only
   evaluation would have reported roughly half the true operational error, and reporting it would
   have been the most serious credibility failure available in this thesis.
2. **The ground truth is real telemetry, not a simulation.** E.ON Energiemonitor, 15-minute, two
   towns, collected continuously by `scraper_service`. It is append-only because it cannot be
   re-obtained. Its timezone was verified empirically by cross-correlation lag, not assumed.
3. **The serving path is the validation path.** `assemble_municipal_features` is called with a
   different weather loader and nothing else changes. Without that, the frozen affine calibration
   would not be valid at inference.
4. **Interval honesty at serving time.** Realised municipal PICP against an 0.80 nominal:
   KW wind 0.38, KW solar 0.33, Nauen wind 0.41, Nauen solar 0.17. After conformal: 0.79 / 0.80 /
   0.79 / 0.82. Macro wind: 0.62 / 0.69 / 0.68 → 0.85 / 0.86 / 0.77. The dashboard states in the UI
   that raw quantile bands under-cover locally and should be read as indicative.
5. **State-wide operation is real, not a mock.** 18 Kreise, point-in-polygon asset assignment, batch
   parquet with live-NWP fallback, rate-limit backoff, choropleth. The aggregate band is disclosed
   as the comonotonic (perfect-correlation) sum of district intervals — a conservative upper bound,
   because spatial forecast errors partly decorrelate.
6. **The whole pipeline runs from one command** and the nightly weather refresh runs from cron.

---

## 7. Threats to validity — state these before the committee does

**T1 — The priors are Brandenburg-only; the target is the whole 50Hertz zone.**
The effective wind capacity is ~16 GW (zonal) while the Brandenburg fleet is 9.57 GW. The
Brandenburg fleet is therefore a *sample* of the zonal fleet, and the prior is a proxy for the
zonal capacity factor's shape, not its level. The CF formulation is what makes this survivable
(both sides are dimensionless) and the learner absorbs the level difference. The empirical evidence
that the sample is informative is the fit itself (wind CF corr 0.955). **Mitigation if pushed:**
restricting the target to a Brandenburg-only actuals series, if one were published, would remove
this entirely.

**T2 — The wind `scale` divergence (0.46 vs 0.0153) is not fully explained.**
The 8 km radius is a proxy for a metering boundary that is not published. Two competing readings:
(a) Nauen's E.ON meter sees only a small fraction of the MaStR wind fleet within 8 km; (b) the
radius is simply too large for Nauen's geography. Both are magnitude effects, and `lomo_transfer`
*proves* the shape is unaffected. It is a capacity-accounting open question, and the thesis says so.

**T3 — Two municipalities is a small validation base.**
It is the entire public ground truth available for Brandenburg. LOMO transfer is a 2-point
experiment; with `n = 2` the transfer result is directional evidence, not a distribution. This
bounds every municipal claim in the thesis and is the first item in future work.

**T4 — Chapter 2.5.3 describes residual learning; the implementation is input augmentation.**
The learner predicts CF directly with the prior as a feature; it does not regress on `y − prior`.
Both are legitimate hybrid couplings, and the choice is defensible (D3), but the terminology in the
background chapter should match the method chapter. **Be ready to name the coupling correctly:
feature-level / input-augmentation hybrid, not residual.**

**T5 — The conformal guarantee is empirical, not theoretical.**
Contiguous time blocks are not exchangeable draws, and `w_floor` is tuned on the calibration split.
Reported coverage is measured on held-out data. Written into the module docstring.

**T6 — Precomputed solar geometry is unused by the final feature set.**
`ghi_clearsky` and zenith are computed and joined but do not enter `FEATURE_COLS`. Defensible
(redundant with `solar_prior_cf` + cyclical time; no ablation evidence it was needed), but it is
dead weight in the pipeline and should be named as such rather than defended as essential.

**T7 — `mastr_downloader.download_brandenburg_boundary` duplicates `boundary_fetcher`.**
Dead code retained from the original download script. `main.py` uses the `boundary_fetcher` path.

**T8 — Municipal validation uses 855 hours.**
Roughly 36 days of held-out overlap. Sufficient for a stable affine fit and a credible nMAE; not
sufficient to characterise seasonal behaviour at municipal scale.

---

## 8. The thirty-second answer, if you get exactly one

> The core problem is that generation is published at control-zone scale and needed at municipal
> scale, with almost no local ground truth. I made the learning problem dimensionless: a bottom-up
> physical simulation of the registered fleet produces a capacity-factor prior, and the model
> predicts the 50Hertz capacity factor from that prior plus weather and cyclical time. Nothing in
> the learned map has units, so it applies unchanged to a single municipality — I only need a local
> nameplate and a two-parameter affine correction fitted on a held-out split of the little telemetry
> that exists. Everything is validated under a purged expanding-window protocol with a 72-hour
> embargo, the physics prior's contribution is established causally by ablation rather than by
> attribution, model differences are tested per-point with Diebold-Mariano and a block bootstrap
> because four folds cannot carry a significance claim, and the operational penalty of using
> forecast weather instead of reanalysis is measured at +91% wind / +77% solar rather than assumed
> away.
