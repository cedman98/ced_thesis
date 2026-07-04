# Checkpoint 08: Complete Results Analysis

---

## 1. LightGBM SHAP — Wind (`lightgbm_wind_shap_summary.png`)

The beeswarm plot tells you both ranking and direction of each feature's influence on the predicted capacity factor.

- **`brandenburg_wind_prior_mw`** is by an enormous margin the single most important feature. The SHAP range runs from roughly −4 000 to +10 000 W, completely dwarfing every other variable. The colour pattern is clean monotone: low feature values (blue) push predictions down, high feature values (red) push them strongly up, with almost no overlap. This is exactly what you want — the physics-derived wind power prior is doing the heavy lifting, and the model is trusting it linearly.
- **`surface_pressure`** is second. The spread is ±1 000 W. There is a slight positive skew (higher pressure → slightly higher wind output), consistent with the physical relationship between high-pressure anticyclones and wind suppression in the North European Plain — so higher pressure suppressing wind is the expected sign, but pressure variation here is a proxy for synoptic weather regime rather than a direct aerodynamic cause.
- **`temperature_2m`** is third, also ±~500 W spread, roughly symmetric around zero — temperature here is acting as a proxy for air density (colder air = denser = more kinetic energy per unit volume), not a diurnal signal.
- **`brandenburg_solar_prior_mw`** appearing fourth in the wind model is interesting. It likely enters as a synoptic co-variate — sunny days often correspond to particular pressure/stability regimes that correlate with low wind.
- **Cyclical time features** (`month_cos`, `hour_sin`, `month_sin`, `hour_cos`) are ranked 5–9. Their SHAP spreads are all narrow (±300–500 W), which makes sense: seasonality for wind is real but secondary to the physical prior. The dense clustering near zero for `hour_cos` and `hour_sin` suggests the model found only a weak diurnal wind signal, as expected for an inland flat state like Brandenburg.
- **`relative_humidity_2m`** is notably spread leftward (negative SHAP for high humidity values in blue), which may be capturing fog/stable boundary layer conditions that suppress wind turbine output near the surface.

---

## 2. LightGBM SHAP — Solar (`lightgbm_solar_shap_summary.png`)

- **`brandenburg_solar_prior_mw`** dominates even more cleanly than the wind prior dominates the wind model. SHAP values reach +9 000 W on the positive side, nearly double the wind equivalent, and the monotone colour gradient (blue → red = low → high → positive push) is essentially perfect. Solar output is highly predictable from a physics prior; the model is correctly anchoring on it.
- **`hour_sin`** is second with a spread of roughly −2 000 to +2 000 W. Critically, the colour pattern here is non-monotone: both high (red) and low (blue) values of `hour_sin` push predictions in both directions. This is expected — `hour_sin` encodes the sine of hour-of-day, and both early morning and late afternoon land at similar values, creating this symmetric diamond cloud. The model captures the sharp intraday rise and fall of solar generation.
- **`surface_pressure`**, **`relative_humidity_2m`**, **`temperature_2m`** follow, all with ±500–800 W range. High humidity (blue) pushes solar predictions down — cloud cover proxy. Temperature pushes slightly positive (high temperature → summer → more insolation), but the effect is weak because the physics prior already captures seasonality.
- **`brandon_wind_prior_mw`** appearing sixth in the solar model is again a synoptic weather regime signal — windy conditions correlate with cloud cover.
- **`month_cos`**, **`hour_cos`**, **`month_sin`** are bottom three with very tight ±~200 W spreads, meaning the cyclical time features are almost redundant once the physics prior is included for solar. The solar prior from `pvlib` already encodes the geometric sun angle, so hour/month features add little marginal information.

---

## 3. BiLSTM SHAP — Wind (`bilstm_wind_shap.png`)

The BiLSTM SHAP plot is in capacity-factor units (x-axis ~−0.04 to +0.12), not raw watts, because the BiLSTM operates on the normalised CF target. A few important observations:

- **`brandenburg_wind_prior_mw`** is still the top feature but the distribution is markedly different from LightGBM. The cloud is mostly positive (right of zero) with a long rightward tail, but many points cluster near zero. The maximum positive SHAP is ~+0.11 CF. The BiLSTM is less decisive than LightGBM about how to use the prior — many samples with medium prior values get near-zero SHAP, suggesting the recurrent network sometimes overrides the prior based on sequence context. This could indicate partial underfitting, or that the BiLSTM has learned temporal autocorrelation patterns that sometimes conflict with the point-in-time prior.
- **`month_cos`** is second, which is a striking difference from LightGBM where it ranked 5th. The BiLSTM learned stronger seasonal dependence, plausibly because its recurrent structure tracks multi-day patterns and seasonality modulates those.
- **`brandon_solar_prior_mw`** is third, again as a weather-regime proxy — consistent across models.
- **`hour_cos`**, **`temperature_2m`**, **`surface_pressure`**, **`hour_sin`**, **`relative_humidity_2m`**, **`month_sin`** trail off. Notably, `surface_pressure` dropped from rank 2 (LightGBM) to rank 6 here. The BiLSTM may be encoding synoptic pressure evolution through the recurrent hidden state rather than using the instantaneous pressure value directly, so the point-in-time SHAP attribution is lower.

*Note: The BiLSTM SHAP values are much tighter and more clustered near zero than LightGBM's. This reflects the BiLSTM operating on a bounded [0,1] output space — individual feature attributions look smaller in absolute terms.*

---

## 4. EBM Feature Importances — Wind

EBM reports mean absolute score, which reflects the average magnitude of each term's contribution across the training data. The EBM is an interpretable GA²M (Generalised Additive Model with pairwise interactions), so the output separates main effects from pairwise interactions.

**Main effects** (69.5% of total score = 3 420 / 4 924):

| Rank | Feature | Score | Share |
| :--- | :--- | :--- | :--- |
| 1 | `brandenburg_wind_prior_mw` | 2 590 | 52.6% |
| 2 | `month_cos` | 198 | 4.0% |
| 3 | `surface_pressure` | 151 | 3.1% |
| 5 | `temperature_2m` | 116 | 2.4% |
| 6 | `brandon_solar_prior_mw` | 108 | 2.2% |

The wind prior alone accounts for more than half of all predictive signal in the EBM. This validates the physical simulation approach — a well-engineered prior captures the majority of variance before any ML is needed.

**Pairwise interactions** (30.5% = 1 504):

- The top interaction is `wind_prior × solar_prior` (score 138, 2.8%). This is the cross-term the EBM discovered: on days when both wind and solar actuals are high or both are low, there is a non-additive synoptic signal — clear skies with high irradiance typically mean calm anticyclonic conditions, so the interaction captures weather-regime clustering beyond what each prior captures alone.
- `wind_prior × surface_pressure` (83) and `temperature_2m × surface_pressure` (86) are the next largest interactions, consistent with the physical relationship between air density, synoptic pressure, and available wind energy.
- Cyclical time interactions (`wind_prior × month_cos`, `surface_pressure × month_cos`, etc.) fill ranks 10–20 and collectively explain another ~8% of variance, capturing seasonal variation in the prior's accuracy — the model is more or less confident in the prior depending on season.

---

## 5. EBM Feature Importances — Solar

**Main effects** (64.6% = 3 550 / 5 497):

| Rank | Feature | Score | Share |
| :--- | :--- | :--- | :--- |
| 1 | `brandon_solar_prior_mw` | 2 402 | 43.7% |
| 2 | `hour_sin` | 447 | 8.1% |
| 4 | `hour_cos` | 223 | 4.1% |
| 6 | `temperature_2m` | 144 | 2.6% |
| 9 | `surface_pressure` | 92 | 1.7% |

The solar prior is dominant but somewhat less so than wind's prior (43.7% vs 52.6%). The reason: for wind, the physical power curve prior is highly predictive; for solar, cloudiness introduces uncertainty that the `pvlib` clear-sky model cannot capture. Hence `hour_sin`/`hour_cos` (the diurnal cycle) carry more residual weight for solar — they help the model know when to expect irradiance even when the prior is uncertain due to cloud cover.

**Pairwise interactions** (35.4% = 1 947):

- The key interaction is `solar_prior × hour_sin` (316, 5.8%) — the model found that the prior's reliability varies by time-of-day: at dawn and dusk the prior is noisier, so the hour encoding modulates how much weight is placed on it. This is a physically meaningful non-linearity.
- `hour_cos × month_cos` (197, 3.6%) captures seasonal variation in the diurnal pattern — summer afternoons behave very differently from winter afternoons. The EBM is the only model that makes this interaction explicitly visible.

---

## 6. TFT — Wind

The TFT separates variables into encoder (observed history) and decoder (known-future covariates), plus static context. This is the most informative XAI view of all models.

**Static variables (context for the whole sequence):**
- `wind_onshore_mw_50hz_scale` — ~60% importance. The normalisation scale of the 50Hertz target series is by far the most important static context feature. This is the TFT learning to condition all predictions on how much installed capacity the Brandenburg region has at a given time period — a proxy for overall fleet size.
- `wind_onshore_mw_50hz_center` — ~35%. The mean level of the target series. Together with scale, this means the TFT is doing internal de-normalisation: it knows the baseline generation level and its spread, allowing it to modulate predictions intelligently.
- `encoder_length` — ~5%. How long the lookback window is for this sample. A small but non-trivial signal — the model slightly adjusts based on how much history it was given.

**Encoder variables (past observations):**
- `wind_onshore_mw_50hz` — ~100%. Essentially all encoder importance. The TFT learned that the most informative thing about the past is the actual 50Hertz wind generation itself. Everything else — physics prior, weather, time features — is negligible for the encoder. This makes sense: the autoregressive signal (what was generation actually doing) is the strongest short-term predictor.

**Decoder variables (future known inputs):**
- `brandon_wind_prior_mw` — ~50%. In the forecast horizon, where you no longer have actual generation to look at, the physics prior takes over as the dominant signal. This is exactly the intended design of the bottom-up approach.
- `surface_pressure` — ~11%. Synoptic pressure forecast matters for future wind.
- `relative_humidity_2m`, `month_cos`, `temperature_2m` — 7–8% each.
- `hour_cos`, `hour_sin`, `month_sin` — 4–5% each.
- `relative_time_idx` — ~2%. The TFT is using the sequence position only minimally.

**Attention** (`tft_wind_attention.png`):
The attention curve decays monotonically from the oldest lookback step (~−72) to the most recent (~0), with the range extremely narrow: 0.01385 to 0.01445 — only a 4% variation across the full 72-hour lookback window. This is essentially uniform attention. The TFT wind model is averaging across the entire encoder history rather than attending to specific past moments. This suggests the wind signal is not strongly self-similar at specific lags (no strong 24h periodicity in wind), so all past context is roughly equally informative.

---

## 7. TFT — Solar

**Static variables:**
- `solar_pv_mw_50hz_scale` — ~83%. Much more dominant than the wind equivalent. The target scale is critical for solar because installed PV capacity has been growing rapidly; the model needs to know the current fleet size to de-normalise correctly.
- `solar_pv_mw_50hz_center` — ~10%. Much less important than for wind because solar generation has a hard floor of zero and a strong diurnal structure — the mean level matters less than the scale.
- `encoder_length` — ~5%.

**Encoder variables:**
- `brandon_solar_prior_mw` — ~27%
- `solar_pv_mw_50hz` — ~19%

This is the key difference from wind: for solar, the encoder splits importance nearly evenly between the physics prior and the actual 50Hertz series. The physical prior (`pvlib` clear-sky model) is informative even about the past, which makes sense — cloudiness is persistent, and the prior captures the geometric maximum against which actual is compared.

- `hour_cos` — ~12%, `relative_time_idx` — ~11%, `relative_humidity_2m` — ~10%. These are all high relative to the wind model. The TFT solar encoder needs time-of-day to understand the current phase of the diurnal cycle.

**Decoder variables:**
- `brandon_solar_prior_mw` — ~35%. Physics prior dominates the forecast horizon.
- `hour_sin`, `hour_cos` — ~12% each. Much more important in decoder than for wind — the model needs to track the diurnal cycle in the future forecast window.
- `surface_pressure`, `month_cos` — ~9% each.
- `temperature_2m`, `relative_humidity_2m` — ~7% each.

**Attention** (`tft_solar_attention.png`):
Completely different from wind. There is a sharp spike at t ≈ −72 (attention ~0.051) that decays rapidly within the first few steps to a baseline of ~0.012, then remains flat with a very slight upward drift toward t=0. This means the solar TFT is strongly attending to the same time step 72 hours ago — i.e., exactly three days prior. This captures the 72-hour periodic pattern: solar generation three days ago at the same hour of day is informative about today's generation (similar sun angle, and weather persistence). This is a genuinely interesting learned behaviour and validates that the TFT attention mechanism is picking up physically meaningful temporal structure.

---

## 8. Municipal Validation (Königs Wusterhausen)

**Dataset**: 107 hours from 2026-05-28 13:00 UTC to 2026-06-02 00:00 UTC (approximately 4.5 days). TFT data covers only 85 of those hours (missing from 2026-06-01 02:00 UTC onward, likely a forecast horizon boundary issue).

**Actuals**:
- **Wind**: mean 1 720 W, max 7 984 W, never zero (continuous generation over this period)
- **Solar**: mean 1 088 W, max 3 392 W, 32 zero-hours (nighttime)

### Wind performance — all models underforecast substantially:

| Model | MAE (W) | RMSE (W) | Bias (W) | Pearson r | nMAE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LightGBM | 1 421 | 2 188 | −1 336 | 0.340 | 82.6% |
| XGBoost | 1 497 | 2 205 | −1 187 | 0.115 | 87.0% |
| EBM | 1 468 | 2 247 | −1 338 | 0.097 | 85.3% |
| TFT | 1 524 | 2 401 | −1 428 | −0.089 | 88.6% |

All four models significantly underforecast wind at the municipal level. The bias of ~−1 200 to −1 430 W against a mean actual of 1 720 W means all models are predicting roughly 70–80% below the true local generation on average. LightGBM is the best-performing wind model by all metrics (lowest MAE, RMSE, highest correlation). XGBoost's negative correlation near zero, and TFT's negative correlation, indicate these models are not tracking the actual wind variability at all at the sub-TSO scale.

The root cause of the underprediction bias is the spatial downscaling gap: the models were trained on 50Hertz Brandenburg-wide actuals and are being evaluated against a single Gemeinde (Königs Wusterhausen). The local wind fleet may have different capacity factors, micro-siting effects, or curtailment patterns not captured in the Brandenburg aggregate prior.

### Solar performance — dominated by systematic over-forecasting:

| Model | MAE (W) | RMSE (W) | Bias (W) | Pearson r | nMAE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LightGBM | 757 | 1 046 | +538 | 0.797 | 69.6% |
| XGBoost | 886 | 1 189 | +345 | 0.591 | 81.4% |
| EBM | 2 445 | 3 937 | +2 283 | 0.812 | 224.7% |
| TFT | 1 263 | 1 752 | +1 261 | 0.947 | 116.0% |

Solar is the opposite problem — all models over-forecast (positive bias), meaning they predict more local PV generation than actually occurred. This is again a scale issue: models trained on the large Brandenburg total are predicting too much for a single municipality.

LightGBM wins on MAE and RMSE for solar. TFT has the highest correlation (r = 0.947), meaning it tracks the diurnal shape beautifully despite the large magnitude offset. This is consistent with what the attention plot showed — the TFT learned 24h/72h solar periodicity very well. The EBM solar model is catastrophically miscalibrated (nMAE = 225%), likely because it is the least regularised model and the global-to-local scale mismatch amplifies its systematic error.

**Day-by-day wind**:
2026-05-30 had the highest actual wind day (daily mean 4 099 W), but LightGBM only forecast 537 W — a factor of ~8 underprediction. This is a strong wind event. On moderate days (May 28–29), all models are closer in relative terms. This suggests the models' underperformance is worst during high-wind events, which is a common failure mode when the prior is trained on capacity factors that compress extreme events.

---

## Summary of Key Findings

1. **Physical priors dominate everywhere.** The `brandon_wind_prior_mw` and `brandon_solar_prior_mw` features account for 44–53% of total EBM importance, appear first in LightGBM SHAP by massive margins, and are the top decoder variable in both TFT models. The bottom-up physics simulation is doing exactly what it was designed to do.
2. **TFT wind uses autoregressive history; TFT solar uses both history and prior.** The encoder importance split (wind: 100% on actual 50Hz series; solar: 27% prior + 19% actual) reveals that wind's short-term memory is dominated by persistence (what happened recently), while solar benefits from physics prior even in hindsight.
3. **TFT solar attention found a 72-hour lag.** The sharp attention spike at t=−72 is a physically meaningful discovery — same hour three days ago is the most informative historical point for solar forecasting.
4. **TFT wind attention is nearly uniform** — wind lacks strong periodic memory at sub-weekly scales, so all history is equally weighted.
5. **LightGBM is the best all-round model at the municipal validation scale.** It achieves the best wind MAE/RMSE, best solar MAE/RMSE, and reasonable correlation. TFT has better solar correlation but a large magnitude bias. EBM is catastrophically over-forecasting solar at municipal scale.
6. **All models have a large systematic underprediction bias for wind and overprediction for solar at Königs Wusterhausen.** The normalised MAE of 82–89% for wind is poor and indicates the spatial downscaling gap is the primary bottleneck, not model architecture. This is the core thesis problem to address — whether that is a capacity factor recalibration, a local capacity adjustment, or an additional municipality-level feature.