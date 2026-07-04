# Checkpoint 10: Final Verdict and Future Work

---

## 1. What you actually built (so the results make sense)

You take 50Hertz macro actuals (whole TSO zone, ~15 GW wind / ~14 GW solar effective capacity) and learn a capacity-factor map from 9 scale-free features, then transfer that same map down to a single municipality (Königs Wusterhausen) via one calibration constant `k`. That transferability is the whole thesis — and the results either support it or they don't. They mostly do, with one honest caveat.

**Experimental setup (this is your "Methods" section):**
- **Data:** hourly, 2022-01-01 → 2026-05-31, 38,687 rows, UTC. ~4.4 years.
- **Target:** capacity factor CF = y/C, bounded [0,1] — kills the non-stationary commissioning trend. This is the correct choice and defensible.
- **Features (9, all dimensionless):** `wind_prior_cf`, `solar_prior_cf` (physics priors), `temperature_2m`, `surface_pressure`, `relative_humidity_2m`, and cyclical hour/month sin/cos.
- **Validation:** 4-fold purged expanding-window CV with a 48–72h embargo. This is the single most important credibility feature of your whole thesis — it eliminates autocorrelation leakage. Examiners will ask "how do you know you're not just memorizing yesterday's weather?" and this is your answer. Lead with it.
- **Baselines:** persistence (t-24h) and climatology. You need these — a model with no baseline is unfalsifiable.

---

## 2. Macro results (50Hertz zone) — the headline table

Mean over 4 CV folds, capacity-factor scale. nMAE lower = better; corr higher = better.

### WIND

| Model | nMAE | nRMSE | corr | MBE |
| :--- | :--- | :--- | :--- | :--- |
| persistence | 0.649 | 0.885 | 0.48 | 0.00 |
| climatology | 0.679 | 0.855 | 0.27 | +0.01 |
| LightGBM | 0.202 | 0.274 | 0.954 | −0.004 |
| TFT | 0.202 | 0.278 | 0.956 | −0.005 |
| XGBoost | 0.209 | 0.284 | 0.951 | −0.003 |
| EBM | 0.213 | 0.290 | 0.948 | −0.004 |
| BiLSTM | 0.246 | 0.335 | 0.924 | −0.013 |

### SOLAR

| Model | nMAE | nRMSE | corr | MBE |
| :--- | :--- | :--- | :--- | :--- |
| persistence | 0.286 | 0.620 | 0.919 | 0.00 |
| climatology | 0.312 | 0.619 | 0.922 | +0.005 |
| XGBoost | 0.160 | 0.316 | 0.979 | −0.001 |
| LightGBM | 0.161 | 0.314 | 0.979 | −0.001 |
| TFT | 0.166 | 0.325 | 0.979 | −0.001 |
| EBM | 0.176 | 0.323 | 0.978 | +0.000 |
| BiLSTM | 0.234 | 0.448 | 0.949 | −0.005 |

**How to read this:**

- **Wind** is where your ML earns its keep. Persistence is terrible for wind (nMAE 0.65, corr 0.48) — wind is chaotic hour-to-hour, so "same as yesterday" fails. Your models cut error by ~69% (0.65 → 0.20) and push correlation from 0.48 to 0.95. That is a large, real, defensible win.
- **Solar** is a subtler story. Persistence is already good (nMAE 0.29, corr 0.92) because solar is dominated by the deterministic day/night + seasonal cycle — yesterday looks a lot like today. Your ML still improves it ~44% (0.29 → 0.16), which is solid, but the baseline is strong. Don't oversell solar; the honest framing is "ML captures cloud/weather deviations that the diurnal baseline can't."
- **MBE ≈ 0** everywhere for the ML models. No systematic over/under-forecasting bias at macro scale. That's a clean result — report it, it shows calibration.
- **nRMSE >> nMAE** (esp. solar 0.31 vs 0.16) means errors are concentrated in a few large misses — cloud-front timing, wind ramps — not spread evenly. Standard for renewables; worth one sentence.

---

## 3. The novel contribution: municipal downscaling (Königs Wusterhausen)

This is the part nobody else has done and what makes it a thesis rather than a Kaggle notebook. The macro-trained CF model applied to KW telemetry, n=391 aligned hours, calibrated `k` per (model, tech).

| Model | Tech | nMAE | corr | MBE (MW) | k |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LightGBM | wind | 0.387 | 0.82 | −274 | 0.48 |
| XGBoost | wind | 0.392 | 0.81 | −246 | 0.48 |
| EBM | wind | 0.404 | 0.81 | −179 | 0.49 |
| TFT | wind | 0.424 | 0.81 | −337 | 0.50 |
| BiLSTM | wind | 0.419 | 0.78 | −340 | 0.46 |
| persistence| wind | 0.711 | 0.49 | — | — |
| XGBoost | solar | 0.178 | 0.98 | +109 | 0.059 |
| LightGBM | solar | 0.184 | 0.98 | +125 | 0.059 |
| BiLSTM | solar | 0.194 | 0.98 | +141 | 0.059 |
| EBM | solar | 0.201 | 0.98 | +124 | 0.058 |
| TFT | solar | 0.274 | 0.94 | +96 | 0.058 |
| persistence| solar | 0.183 | 0.95 | — | — |

**The two findings that matter for your thesis:**

1. **Wind downscaling works and is the real result.** Municipal wind nMAE 0.39 vs persistence 0.71 — the macro-learned map transfers to a single town and nearly halves the error a naive local forecaster would get. Correlation 0.82 at municipal scale from a model that never saw municipal data is a genuinely strong transfer-learning result. This is your money paragraph.
2. **Solar downscaling is a statistical tie with persistence** (0.178 vs 0.183). Be honest about this — it's actually a more interesting scientific finding than pretending it's a win. Interpretation: at a single site, solar CF is so dominated by the sun's geometry (which persistence already encodes via the 24h lag) that the macro weather map adds almost nothing on top of the diurnal cycle over a short clear-ish window. The value of your solar model is at the macro scale and in cloudy regimes, not sunny single-site nowcasting. Say that.

**Caveats you must state (examiners will find these):**
- **The municipal window is tiny and single-season:** n=391 hours ≈ ~2.5 weeks in late May 2026, one municipality, summer. This is a proof-of-concept, not a validation. It is the #1 threat to validity in the whole thesis. Frame it as "pilot" and make "extend to full year + Nauen" your headline future-work item.
- **Wind MBE is negative** (~−250 to −340 MW): the models systematically under-predict KW wind. `k` corrects the mean scale but not this residual bias — likely local fleet/hub-height differences the macro prior doesn't see. Honest, and a good XAI discussion point.
- **BiLSTM/TFT have NaNs** in the municipal set (they need sequence warm-up the 2-week window can't fully provide) — another argument for the tree models operationally.

---

## 4. What the XAI / feature importance shows

From `ebm_{wind,solar}_importance.csv` (EBM is glass-box, so these are exact contributions, not SHAP approximations):

- **Wind:** `wind_prior_cf` dominates at 0.165, ~13× the next feature (`month_cos` 0.012). Your physics prior is doing the heavy lifting — the ML is mostly correcting the power-curve simulation, not replacing it. This directly validates the "physics-first, ML-as-correction" architecture. Weather features (pressure, temp, humidity) and their pairwise interactions fill in the rest.
- **Solar:** `solar_prior_cf` dominates at 0.192, then `hour_sin` (0.037), `hour_cos` (0.019), and the `solar_prior × hour` interaction (0.027). So solar = physics prior + time-of-day, exactly as expected. Temperature shows up (panel efficiency), which is physically correct and nice to point to.

**Thesis claim you can defend:** "The dominant predictor for both technologies is the deterministic physical prior; the ML layer contributes bounded meteorological corrections. This is why the model is interpretable, transfers across scale, and degrades gracefully." That sentence ties physics + ML + XAI + downscaling into one argument. Use it.

---

## 5. Model-by-model verdict (which to actually recommend)

| Model | Verdict |
| :--- | :--- |
| **LightGBM** | Winner overall. Best or tied-best macro wind, top-2 solar, best municipal wind, fast, SHAP-compatible, no warm-up. This is your production recommendation. |
| **XGBoost** | Statistical dead heat with LightGBM. Best municipal solar. Confirms the result isn't a single-library artifact — good for robustness. |
| **EBM** | ~1–2% behind the boosters but glass-box — every prediction decomposes exactly. Your interpretability workhorse. Slightly worse municipal wind. |
| **TFT** | Competitive (best macro-wind corr 0.956) but not better than a gradient-boosted tree, while being vastly more complex, GPU-hungry, and NaN-prone at municipal scale. This is a finding, not a failure: "attention-based deep forecasting did not outperform gradient boosting on this problem." Examiners love a negative result that's honestly reported. |
| **BiLSTM** | Consistently the weakest ML model (macro solar 0.23, wobbly fold 1) and NaN at municipal scale. Keep it as the "we tried recurrent nets" data point. |

**Big-picture verdict:** the simple, interpretable models win. That's a clean, honest, defensible thesis conclusion and it aligns with the physics-first philosophy.

---

## 6. Is it good? Direct answer.

**Yes, with scoped honesty.**
- **Macro forecasting:** genuinely good. nMAE 0.16 solar / 0.20 wind with corr ~0.95–0.98, zero bias, leak-proof CV. Competitive with published renewable-forecasting literature.
- **Wind downscaling:** the standout novel result. Transfers to municipal scale and beats the naive baseline ~2×.
- **Solar downscaling:** honest tie with persistence — scientifically interesting, not a headline win.
- **Rigor:** purged embargoed CV + physics priors + XAI + baselines is a methodologically strong package. This is what will get you the grade, more than the raw nMAE.

*The one thing standing between "good project" and "airtight thesis": the municipal validation is 2.5 weeks of one town in summer. Everything else is solid; this is thin. Fix it and you're bulletproof.*

---

## 7. Future work + "could municipalities actually use this?"

**Yes — and it's a realistic product, not a fantasy.** The architecture is built for it: scale-free CF features mean any municipality needs only (a) its local nameplate MW and (b) one calibration constant `k` from a short telemetry sample. No retraining per town. That's the productization thesis and it's technically sound.

**Concrete roadmap, ordered by payoff:**

1. **Extend municipal validation** to a full year + Nauen (config already has it). Turns the pilot into a validation. Do this first — it's the difference-maker.
2. **Replace weather archive with a forecast API** (Open-Meteo forecast endpoint). Right now you use reanalysis (actuals). A live 24h-ahead tool needs forecast weather — and your error will rise, because NWP weather is imperfect. Quantify that gap; it's an honest and important experiment.
3. **Fix the municipal wind under-bias** — fold `k` into a small 2-parameter affine calibration (scale + offset) to kill the −270 MW MBE.
4. **Add prediction intervals** (LightGBM quantile regression, nearly free) — municipalities need "how confident," and your nRMSE>>nMAE shows fat-tailed errors that a point forecast hides.
5. **Then the product:** per-municipality dashboard = CF model + local nameplate + `k` + live weather → 24h wind/solar MW with intervals. Grid balancing, curtailment planning, local energy-community scheduling are all real use-cases for a Landkreis.

*Honest product caveats: trained/validated only on Brandenburg summer at one site; needs a short telemetry period per town to fit `k`; degrades with forecast (vs reanalysis) weather; single-site solar barely beats persistence so the pitch is wind-first.*