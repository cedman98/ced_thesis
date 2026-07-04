# Code review — Brandenburg Energy Forecast System

**TL;DR:** The core pipeline (`ingestion` → `CF` feature matrix → purged `CV` → calibrated downscaling) is sound and unusually well-documented. But there are three real correctness bugs that affect your thesis numbers (test-fold leakage in `BiLSTM` and `TFT` training, and a confirmed one-hour off-by-one in the `TFT` municipal validator), three stale `XAI` modules that will crash if run, and a subtle target-normalization leakage a thesis reviewer could flag. The cron weather job is also silently a no-op.

---

## Bugs that affect reported results

1. **BiLSTM CV early-stops on the test fold — leakage.** `train_bilstm.py:118-120`: `vl` is built from the test fold and passed as `val_loader`, and `train_model` selects the best epoch by that loss (`train_bilstm.py:79-84`). Your reported BiLSTM CV metrics are optimistically biased because the epoch was chosen to minimize test error. **Fix:** carve a validation tail off the training window (e.g. last 10%, respecting the embargo) for early stopping, and evaluate on the untouched test fold.

2. **TFT CV has the same leakage, plus "best checkpoint" is actually the last.** `train_tft.py:66-78`: `val_dl` is the test fold and `EarlyStopping(monitor="val_loss")` uses it. Additionally, you never add a `ModelCheckpoint(monitor="val_loss")`, so `trainer.checkpoint_callback.best_model_path` (`train_tft.py:80`) is Lightning's default checkpoint — the last epoch, which by construction is 3 epochs past the early-stopping optimum. Same fix as above, plus an explicit monitored `ModelCheckpoint`.

3. **TFT municipal predictions are shifted by one hour — confirmed.** `tft_municipal_validator.py:40-42` takes `preds[:, 23]` and maps it to `idx['time_idx'] + 24`. I checked `pytorch-forecasting`'s source: `x_to_index` returns the `time_idx` of the first decoder step, so the 24-step horizon covers `time_idx` … `time_idx+23`, and `preds[:, 23]` belongs to `time_idx + 23`. Every TFT municipal prediction is timestamped one hour late — for solar's steep ramps this alone degrades `nMAE`/correlation materially. **Fix:** `+ 23` (or take `preds[:, -1]` with `+ MAX_PREDICTION_LENGTH - 1`).

4. **`time_idx = arange(len(df))` treats data gaps as contiguous.** `train_tft.py:46` and `tft_municipal_validator.py:32`: after `dropna()`, missing hours make the index non-contiguous, but a sequential `time_idx` tells the TFT that rows across a gap are adjacent — encoder windows silently span discontinuities. **Fix:** derive `time_idx` from the timestamp (`(t - t0) // 1h`) and let `pytorch-forecasting`'s `allow_missing_timesteps` handle gaps, or drop windows crossing gaps.

5. **All three standalone XAI modules are stale from the CF-refactor and will crash or mislead:**
   - `explain_models.py:42` builds `X` by dropping only the two MW targets — the sample still contains the CF targets, capacity columns, and prior-MW columns, while the models were trained on the 9 `FEATURE_COLS`. SHAP will error on the feature mismatch (and if it ever ran, plotting the target as a "feature" is leakage in the figure). The PNGs in `results/` have different filenames, so these are outputs of an older version.
   - `explain_bilstm.py:105-108` constructs `BiLSTM(input_size=X_test_raw.shape[1])` with that same over-wide column set — `load_state_dict` will fail on shape mismatch against the 9-input trained weights.
   - `explain_tft.py:59-80` rebuilds the dataset with the old MW target (`wind_onshore_mw_50hz`) and `brandenburg_wind_prior_mw` known-reals, incompatible with the CF-target production checkpoints.

   Since `train_tft.py:101-111` already saves TFT interpretation plots, the lazy fix: rewrite `explain_models.py` to use `FEATURE_COLS` (a 5-line change), do the same for `explain_bilstm.py` using the persisted `bilstm_scalers.pkl` meta, and delete `explain_tft.py`.

6. **Effective-capacity normalization uses future data (thesis-rigor issue).** `feature_pipeline.py:180-192` computes `C_t` as the `p99.9` of generation per calendar year — for any test-fold timestamp, the capacity (and hence the CF target) was computed using that same year's full data, including the test period. It's mild (a single scalar per year), but it is technically look-ahead in the target definition, and the current partial year (2026) gets a cap from five months of data. For the thesis, either switch to a causal estimate (trailing 365-day robust peak) or, better, document the override path you already built (`cf_normalization.json` with official 50Hertz figures) and use it.

7. **`nwp_gap` compares in-sample vs out-of-sample.** `nwp_gap.py:125` loads the production `LightGBM`, which was trained on the full reanalysis matrix — so the "reanalysis" error is training-set error while the forecast-weather error is genuinely out-of-distribution input. The reported gap is therefore an upper bound. You already have the honest harness in `export_predictions.py`; evaluate the gap with fold-held-out models (or at least state the caveat in the report).

---

## Smaller bugs and inconsistencies

- **Municipal fleets include decommissioned assets.** `districts.py:92-97` filters `_active()`, but `kw_features.municipal_fleet` (`kw_features.py:59-78`) does not — municipal nameplate and prior-CF denominators include dead assets. The affine calibration absorbs some of this, but the prior feature itself is diluted. Move the `_active` filter into the shared CSV load.
- **Prediction clipping is inconsistent:** training and `nwp_gap` clip CF to `[0, 1.5]`, but the municipal validators and forecast service clip to `[0, 1.0]` (`municipal_validator.py:35,41`, `forecast_service.py:85`). Pick one (the CF target itself is clipped at 1.5) and put the constant in `schema.py`.
- **CV folds share a boundary hour.** `validation_splitter.py:90-91` uses inclusive masks on both ends, so fold i's `test_end` equals fold i+1's `test_start` — `export_predictions.py:81-83` even has a dedup workaround. Fix it once in the splitter with a half-open interval (`>` on `test_start`).
- **`mastr_downloader.load_config` default path is wrong** (`config/conf.yaml` vs the real `conf/config.yaml`) — crashes if run standalone with defaults.
- **`kw_features.assemble_municipal_features` crashes with `IndexError` at `weather_dfs[0]` (`kw_features.py:122`)** if no weather node was loadable — raise a clear error instead.
- **Cluster priors silently under-sum on partial data:** `feature_pipeline.py:89` `concat(axis=1).sum(axis=1)` skips `NaN`s, so an hour where some clusters lack weather quietly produces an under-scaled regional prior instead of `NaN`. Consider `min_count` or asserting aligned indices.
- **Centroid-fallback imputation (`mastr_bulk_parser.py:195-197`):** assets with no coordinates and no PLZ hit get placed at the Brandenburg centroid, injecting phantom capacity into whichever cluster/district contains that point. Persist an imputed flag column so you can quantify (and exclude at district level).

---

## Dead compute / dead config (ponytail findings)

- **The entire solar-geometry stage is unused.** `solar_geometry.py` computes `pvlib` solar position + clear-sky GHI at 15-min resolution over 4.5 years for every solar cluster in parallel — and `feature_pipeline.py:118-125` joins it, then uses only `shortwave_radiation`. Its sole effect is the inner join silently truncating rows if the geometry range is shorter than the weather. Either add `ghi_clearsky` (e.g. as a clear-sky index `shortwave/ghi_clearsky`) to `FEATURE_COLS` — a genuinely useful solar feature — or delete the stage and Step 3b from `main.py`.
- **The config's solar physics constants (`temperature_coefficient`, `inverter_efficiency`, `config.yaml:76-79`) and the parsed azimuth/tilt MaStR fields are collected but never used;** the solar prior is a flat 0.80 performance ratio. Fine as a documented simplification, but then delete the dead config or use it (temperature derating from `temperature_2m` is a two-line improvement that directly helps solar prior quality).
- **Six unused dependencies in `pyproject.toml`:** `aiohttp`, `requests-cache`, `retry-requests`, `openmeteo-requests`, `lxml`, `pydantic`. Remove them. (`hydra-core`/`omegaconf` are used only by `run_scraper.py` — plain `yaml.safe_load` would drop two more, but that's optional.)
- **The 429-handling in `weather_ingestion.py:333-418` is the same ~40-line block copy-pasted four times** — extract one `_handle_429(reason)` helper (~90 lines → ~25).

---

## Workflow issues

- **The nightly weather cron is a permanent no-op.** State shows all 430 nodes completed (checked: `last_call_date: 2026-07-03`, 430/430 done), and `completed_nodes` is binary — a node is never re-fetched, and `end_date` is hardcoded to `2026-06-01` anyway (today is `2026-07-03`). Your cron burns a run daily and ingests nothing. The fix pattern already exists in `extend_kw_weather.py` (append from each node's max timestamp): make the cron job run that extension logic for all nodes instead of `run_weather_ingestion`, and drop the fixed `end_date` default in favor of "yesterday".
- **`run_pipeline.sh` activates `.venv` and calls `python3` directly,** violating your own `CLAUDE.md` rule ("always uv run"). One line: `cd /home/ced/bachelor/final && uv run -m src.data.weather_ingestion >> logs/cron_weather.log 2>&1`.
- **`logs/cron_weather.log` is still tracked by git** (2,571 lines of churn per run) even though you added `/logs/` to `.gitignore` — run `git rm --cached logs/cron_weather.log`.
- **The MaStR bulk URL is a dated snapshot** (`Gesamtdatenexport_20260401_25.2.zip`, `config.yaml:2`) that goes stale monthly; fine for a thesis freeze, just pin the date in the text.
- **The self-checks you've built (`metrics.py`, `kw_features.py`, `forecast_service.py`, `districts.py`, `nwp_gap.py --selfcheck`, the wind-physics unittest) are good but only run ad hoc.** A trivial `test.sh` that runs them all would catch regressions like the stale `explain_*` modules — that class of bug (refactor breaks a non-pipeline module) has already happened three times.
- **Docstring drift:** `municipal_validator.py` and `CLAUDE.md` describe a "rolling-window harness", but the validator actually uses a fixed 40/60 calibration/validation split; `metrics.rolling_metrics` exists but isn't called there. Align the text or the code before your advisor reads both.

---

## What's done well (keep it)

The purged expanding-window CV with embargo, the scale-free CF contract centralized in `schema.py`, the honest OOF export explicitly avoiding in-sample flattery, the affine-calibration design with its degenerate-case fallback and self-check, the resumable rate-limit-aware ingestion, and the persistence/climatology baselines gating ML claims are all exactly the right kind of rigor for this thesis. `Notebook 11` correctly keeps pipeline logic in `src/`.

---
Priority order if you fix nothing else: the TFT off-by-one (#3, one character), the BiLSTM/TFT early-stopping leakage (#1/#2, changes headline CV numbers), the capacity-normalization leakage caveat (#6, one paragraph or one function), and the dead cron (#14 in spirit — your archive stopped growing a month ago). Want me to apply the fixes? I'd start with those four.