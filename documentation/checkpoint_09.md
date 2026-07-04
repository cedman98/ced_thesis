# Checkpoint 09: Honest Assessment as an Energy Forecasting Tool

---

## 1. Honest Assessment as an Energy Forecasting Tool

**The short version: not production-ready, but that was never the goal.**

### Wind at municipal scale: poor

| Metric | LightGBM | XGBoost | EBM | TFT |
| :--- | :--- | :--- | :--- | :--- |
| **nMAE** | 82.6% | 87.0% | 85.3% | 88.6% |
| **Pearson r** | 0.34 | 0.12 | 0.10 | −0.09 |
| **Bias** | −78% | −69% | −78% | −83% |

For context: commercial day-ahead wind forecasting at regional TSO level typically achieves nMAE of 5–15%. A naive persistence baseline (predict tomorrow = today) usually achieves ~20–30%. Your models are at 83–89% — substantially worse than persistence for wind. The negative correlation on TFT means it is actually inversely correlated with the local wind signal.

### Solar at municipal scale: mixed

LightGBM is the most calibrated (nMAE 69.6%, r=0.80). TFT has excellent correlation (r=0.95) but a 116% nMAE — it predicts the shape of the day extremely well but at double the actual magnitude. EBM is broken at this scale (nMAE 225%).

### The honest root cause:

The models were trained on 50Hertz Brandenburg-state-level totals (representing ~7 000 MW wind, ~5 000 MW solar across a 30 000 km² state) and validated against a single Gemeinde. That is a spatial downscaling ratio of roughly 1:100+ in terms of MW. The models have no municipality-specific calibration — they produce a Brandenburg-aggregate capacity factor and apply it locally. This guarantees systematic magnitude error regardless of model quality.

What you don't know yet: you have no Brandenburg-level held-out test metrics in the results folder. You can see the XAI artifacts and the municipal validation, but you cannot determine from these results alone whether the models forecast well at the level they were trained on. That is a gap.

**Bottom line on forecasting tool:** As a standalone operational tool, these results would not meet industry standards at municipal scale. That is not a surprise and is arguably the intended finding — the spatial downscaling gap is precisely what your thesis is investigating.

---

## 2. Honest Assessment for a Bachelor's Thesis

**The good news: you have a lot to work with.**

### Genuine strengths:

**a) Multi-model comparison with principled architecture**
Five models (LightGBM, XGBoost, EBM, BiLSTM, TFT) with a shared feature set is more rigorous than most bachelor-level work. Examiners will respect the breadth.

**b) The physical prior finding is legitimate and defensible**
Across every single XAI method — SHAP (LightGBM, BiLSTM), EBM importance, TFT variable selection — the bottom-up physics prior is unambiguously the most predictive feature (44–53% of total importance). This validates your thesis premise: the bottom-up physical simulation is load-bearing for the ML layer. You can argue this confidently.

**c) The TFT 72-hour attention spike is a genuinely interesting finding**
The solar TFT attending sharply to t=−72 (same hour, three days prior) is not an obvious result and has a clean physical interpretation. This is the kind of thing you can highlight as a novel observation — you built a complex model and it learned physically meaningful temporal structure without being told to.

**d) EBM interactions give interpretable physics**
The `wind_prior × solar_prior` pairwise interaction term (2.8% of total EBM importance) is a cross-model synoptic signal that no other method surfaces. EBM makes it explicitly visible. This is good thesis material — it shows you understand what the model learned and can connect it to meteorological reality.

**e) Cross-scale validation is the thesis contribution itself**
The large municipal errors are a finding, not a failure. The correct framing is: "We show that models trained on TSO aggregate data exhibit systematic bias of X% when applied to municipal scale, and we characterise the nature of that bias." That is a real contribution.

---

### The honest weaknesses (what an examiner might challenge)

**a) The validation window is 107 hours (4.5 days)**
This is the single most vulnerable point. Any examiner or opponent will immediately ask: "Can you draw conclusions from 4.5 days of data at one location?" The answer is technically yes for directional findings (bias sign, correlation structure) but not for quantitative accuracy claims. You cannot say "LightGBM achieves MAE of X W" as if it is a reliable estimate — it could be 3× different in a different 4-day window.

*Mitigation:* Frame the municipal validation as exploratory cross-scale analysis rather than a statistical evaluation. Be explicit about the limitation. If you have more E.ON scraper data available, extend it.

**b) Only one validation location**
Königs Wusterhausen is a single data point geographically. You cannot generalise performance to other municipalities. The examiner will point this out.

*Mitigation:* Frame as a case study. "We demonstrate the method at one municipality as a proof-of-concept; generalisation requires additional measurement sites."

**c) No Brandenburg-level test metrics in results**
You do not have a table showing model accuracy at the training target scale. An examiner asking "how accurate are your models?" will expect to see this. If these metrics only exist in training logs or printed to stdout, you need to recover them.

*Action needed:* Run `uv run -m src.models.train_lightgbm` (or wherever metrics are printed/logged) and capture the held-out CV scores. This is probably the most important missing piece for the thesis.

**d) No naive baseline comparison**
You have no persistence forecast ("forecast = last observation") or climatological mean as a benchmark. Without a baseline, you cannot claim the ML adds value over a trivial model. This is a standard requirement in forecasting papers.

*Mitigation:* Easy to add a persistence baseline in post-processing on the validation parquet. Would take an hour to compute and makes the results section much more credible.

**e) EBM solar at municipal scale is an outlier result**
nMAE of 225% is hard to explain away and will draw attention. You need a clear narrative: "The EBM's lack of output bounding combined with the scale mismatch produces runaway predictions at sub-TSO level" — or similar.

---

## What the Thesis Can and Cannot Claim

| Claim | Supportable? |
| :--- | :--- |
| "Physical priors are the dominant predictive signal" | Yes, strongly |
| "LightGBM performs best at municipal scale" | Yes, with the caveat that it's 4.5 days at 1 location |
| "TFT learns physically meaningful 72h solar periodicity" | Yes |
| "The bottom-up approach successfully downscales to municipal level" | No — the errors are too large |
| "All models exhibit systematic bias at municipal scale" | Yes — and this is the honest finding |
| "Model X achieves state-of-the-art accuracy" | No — you don't have the context data to claim this |

---

## Net Assessment

For a Bachelor's thesis, this is solid work that needs honest framing. You have:
- A well-motivated research question
- A real multi-model pipeline with physical grounding
- Rich XAI results with genuine insights
- A cross-scale validation that identifies the key bottleneck

**What you should do before submission:**
1. Recover Brandenburg-level test metrics — this is the most urgent gap.
2. Add a persistence baseline to the municipal validation — one afternoon of work.
3. Frame the municipal results as a limitation study, not a performance evaluation.
4. Extend the validation window if you have more E.ON scraper data available.

The thesis will not win on forecasting accuracy numbers. It will win on methodological rigour, XAI depth, and honest characterisation of the spatial downscaling challenge. That is a legitimate and defensible contribution at bachelor level.