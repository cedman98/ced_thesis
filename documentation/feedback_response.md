# Response to the professor's feedback — what was measured, and what it changes

Each of the four objections in `feedback.md` is now answered with a persisted
artifact and a figure. Nothing required new data or a new model architecture.

| # | Objection | New artifact | Figure | Verdict |
|---|---|---|---|---|
| 1 | RQ2 answered only by attribution → correlational | `results/ablation_metrics.csv` | `s3_feature_ablation.png` | Prior is causally load-bearing; wind and solar answer **differently** |
| 2 | No leave-one-municipality-out test | `results/lomo_transfer.csv` | `s2_lomo_transfer.png` | Transfer fails on magnitude, **never on shape** |
| 3 | Table 6.2 rankings have no uncertainty | `results/model_significance.csv` | `s1_model_significance_forest.png` | "TFT > LightGBM on wind" is **not supported** |
| 3.1 | *(found while doing 3)* LightGBM's OOF/ablation model used a different objective than the trainer | one-line fix + per-fold assert | — | Corrected; **strengthens** LightGBM on solar |
| 3.2 | *(generalises 3)* Sequence models were unseeded | `results/seed_variance.csv` | — | Run-to-run spread **6.8–14.2%**, exceeds most between-model gaps |
| 4 | Conformal left to future work | `results/conformal_coverage.csv` | `s6_conformal_before_after.png` | Coverage restored: mean \|PICP − 0.80\| **0.235 → 0.031** |

---

## 1. Feature ablation — RQ2 becomes causal (§6.3, §7.2, §8.2.2)

`uv run -m src.evaluation.ablation` — identical purged expanding-window CV, four
feature sets. LightGBM, CV nMAE (CF scale, mean over 4 folds):

| | wind | solar |
|---|---|---|
| `prior_raw` — the physics prior as the forecast, **no learner at all** | 0.281 | 0.419 |
| `no_prior` — ML on weather + cyclical time, **no physics** | **0.576** | 0.250 |
| `full` — production feature set | 0.200 | 0.180 |
| `prior_only` — ML on the technology's own prior alone | 0.192 | 0.309 |

The `prior_raw` row is a fourth reference beyond the three variants originally
scoped. Without it the table cannot separate "the prior carries the signal" from
"the learner corrects the prior", because even `prior_only` still fits a model
*on* the prior. It costs nothing to compute — no fitting, same folds.

The penalty for removing the prior, across all five architectures:

| `no_prior` penalty | wind | solar |
|---|---|---|
| LightGBM | +188.9% | +38.9% |
| XGBoost | +183.9% | +37.9% |
| EBM | +188.0% | +39.0% |
| TFT | +182.7% | +37.4% |
| BiLSTM | +127.5% | +2.5% |

Four of the five agree within **±3 pp on wind (182.7–188.9%) and ±1 pp on solar
(37.4–39.0%)**, across gradient boosting, a glass-box GAM and a transformer. That
tightness is the point: the penalty is a property of the *feature*, not of one
learner's inductive bias.

One caveat on the two sequence rows: their `full` column is its own training draw,
not the run reported in Table 6.2 (re-running them under the ablation is the GPU
cost this split was designed to avoid). Given §3.2, compare sequence *penalties*
— which are within-draw ratios and therefore unaffected — rather than reading
their absolute `full` values against Table 6.2. The three tree rows are
deterministic and match it directly.

BiLSTM is the outlier and should be reported as one rather than averaged in. Its
`full` model is already the weakest of the five (0.306 solar vs 0.180–0.214), so
it was never extracting much from the prior to begin with — a model with little
skill has little to lose.

**The headline is the asymmetry.** On wind, the bare physics prior with no machine
learning whatsoever (0.281) beats a fully trained LightGBM given every weather and
time feature but no prior (0.576) by more than a factor of two. On solar the
ordering reverses — ML without the prior (0.250) beats the raw prior (0.419). The
prior is load-bearing for both technologies, but only for wind is it
*irreplaceable*. That is exactly what the physics predicts: the manufacturer power
curve already encodes the cubic wind-speed response, whereas the clear-sky solar
prior still needs the learner to supply cloud attenuation.

This sharpens §7.2 well beyond "the prior dominates the model's internal
accounting" — it now says which part of the hybrid is doing the work, per
technology, with a falsifiable number.

**The strongest single sentence available.** The TFT's `no_prior` variant still
sees 72 hours of the target's own history through its encoder *and* the full
weather set, and it still scores 0.562 on wind — twice the error of a bare
power-curve calculation with no learning at all (0.281). A sequence model with
three days of autoregressive context cannot recover what the physics prior
supplies for free.

**Two things to state carefully:**

1. On wind `prior_only` (0.192) edges out `full` (0.200) on average, but per fold
   it wins two clearly and loses two narrowly. Report the two as
   indistinguishable — do not claim the weather features hurt.
2. The TFT's `prior_only` (0.207 solar) is not directly comparable to the trees'
   (0.309–0.316), because the TFT always sees the target's own past through its encoder
   whereas the trees given one feature see literally one number per hour. Compare
   TFT variants to each other, and tree variants to each other; the *penalties*
   remain comparable across families because each is measured against that model's
   own `full` baseline.

---

## 2. Leave-one-municipality-out transfer (§6.5, §7.4.4)

`uv run -m src.validation.lomo_transfer` — fit the affine contract at one town,
apply it verbatim at the other, both directions, scored on the target's held-out
window. nMAE, mean over models:

| tech | target | `oracle` (own fit) | `transferred` (donor's) | `uncalibrated` (k=1) |
|---|---|---|---|---|
| wind | Königs Wusterhausen | 0.33 | 0.96 | 1.06 |
| wind | Nauen | 0.40 | **28.48** | 67.70 |
| solar | Königs Wusterhausen | 0.17 | 1.78 | 17.60 |
| solar | Nauen | 0.34 | **0.52** | 8.29 |

Two things to say in the text:

1. **Wind transfer is catastrophic and solar transfer is merely poor.** Applying
   KW's wind scale at Nauen is a 29× error; applying Nauen's at KW (0.96) is
   barely better than no calibration at all (1.06). Solar degrades by a factor of
   1.5–10 instead. This is the asymmetry §7.3 already predicted, now quantified.
2. **Pearson r is bit-identical across all three variants.** An affine map with
   positive slope is a monotone linear rescaling and cannot change correlation —
   asserted in the module's self-check rather than merely claimed. Whatever
   transfer breaks is *magnitude*, never *shape*.

Point 2 is the load-bearing one for §7.4: it converts "we believe the wind
divergence is accounting, not skill" into a mathematical fact plus a measurement.
It also bounds what the 16 unvalidated districts actually deliver — a correct
relative profile on an uncalibrated absolute scale, which is precisely the
honesty caveat §7.4.4 already states.

---

## 3. Statistical weight for the Ch. 6 rankings (§6.1, Table 6.2)

`uv run -m src.evaluation.significance` — all five models now export per-point
out-of-fold predictions, so the comparisons are paired on ~35k identical hours.
Diebold–Mariano with a Newey–West variance (hourly residuals are strongly
autocorrelated) plus a 7-day moving-block bootstrap CI.

**The pivotal row.** `wind, lightgbm − tft`:

| ΔnMAE | relative | bootstrap 95% CI | DM p | fold mean gap | sign-test p | n |
|---|---|---|---|---|---|---|
| +0.0007 | +0.3% | **[−0.0034, +0.0046]** | 0.658 | **0.0000** | 1.000 | 34 948 |

Every view agrees: the CI spans zero, DM is nowhere near significance, and the
fold-mean gap is *identically zero* across the four folds. **The claim "TFT
marginally outperforms LightGBM on wind" is not supported by the evidence.**
Rewrite it as statistically indistinguishable, and keep LightGBM as the production
choice on the grounds that actually decide it: training cost, interpretability,
and native quantile support.

An earlier version of this table read ΔnMAE +0.0035 with DM p = 0.034, which
forced an arbitration between a "significant" DM and a non-significant bootstrap.
That disagreement was an artefact of the objective mismatch described in §3.1, not
a property of the data. With it corrected the tests agree and the conclusion needs
no arbitration — it no longer depends on the reader accepting that the bootstrap
outranks DM.

**Why 4 folds could never have settled this.** With 4 folds the minimum achievable
sign-test p is 0.125, so no fold-level test can reach conventional significance
regardless of the result. `solar, ebm − persistence` shows the gap between the two
views starkly: DM p ≈ 0 at a 26% effect size, but the 4-fold sign test says
p = 0.625.

Every ML-vs-ML pair involving only the deterministic tree models *is* significant
with a CI excluding zero, so that part of the ranking survives. The pairs
involving a sequence model are a different matter — see §3.2.

**One methodological note for the text:** the TFT exports only its h=24 decoder
step. Averaging its 24 overlapping windows per hour would hand it a free
ensembling advantage the single-shot models do not have. Its per-point nMAE
(0.1983 wind) therefore differs slightly from the CV table, which averages
horizons 1–24.

---

## 3.1 A reported model that was not the model being reported

Found while wiring the significance tests, and worth stating in the text because
it changes two published numbers.

`train_lightgbm.py` fits the production point forecast as a **quantile regressor
at α = 0.5** — the pinball/L1 median, chosen because it is robust to the
fat-tailed errors of renewable generation. But `export_predictions._builders`,
which supplies the out-of-fold predictions for both the ablation and the
significance tests, constructed `LGBMRegressor(**PARAMS)` with no objective
argument, silently falling back to LightGBM's **default L2 mean regressor**. Two
different models were being reported under one name: Table 6.2 showed the L1
model, the ablation `full` column and every LightGBM confidence interval showed
the L2 one.

XGBoost and EBM act as controls — identical folds, identical code path — which is
what makes the diagnosis certain rather than plausible:

| | Table 6.2 (wind/solar) | ablation `full`, before fix | after fix |
|---|---|---|---|
| XGBoost | 0.2111 / 0.1981 | 0.2110 / 0.1981 | unchanged |
| EBM | 0.2181 / 0.2137 | 0.2181 / 0.2137 | unchanged |
| **LightGBM** | **0.1989 / 0.1793** | 0.2036 / 0.1966 | **0.1995 / 0.1799** |

Only LightGBM disagreed, and only LightGBM had a divergent objective. The residual
~0.3% is the difference between pooling all held-out hours and averaging four
fold-level nMAEs, which affects all five models equally.

**Two consequences, both against the author's convenience.** The bug was
*understating* the production model:

1. `solar, lightgbm − xgboost` read ΔnMAE −0.0014, CI [−0.0031, +0.0002],
   p = 0.053 — reported as indistinguishable. It is in fact **−0.0179,
   CI [−0.0232, −0.0139], p ≈ 0**: a decisive 9.1% advantage. LightGBM's lead on
   solar is real and was being hidden.
2. The RQ2 cross-architecture agreement in §1 *tightened*. LightGBM's solar
   `no_prior` penalty moved from +37.1% to +38.9%, pulling it toward the other
   three models and narrowing the four-model band from ±2 pp to **±1 pp
   (37.4–39.0%)**. The claim that the penalty is a property of the feature rather
   than the learner is better supported after the correction than before it.

The fix is one line — the builder now carries `objective='quantile', alpha=0.5` —
and `export_predictions` asserts on every fold that its LightGBM point forecast
reproduces the trainer's own q50 fit exactly, so the two cannot drift apart again.
Table 6.2, the ablation `full` column and the confidence intervals now come from
one model.

## 3.2 Sequence-model results are conditional on a training run

The professor's observation that a re-run moved the TFT's wind fold-mean from
0.1928 to 0.1979 generalises further than the one comparison that prompted it.
Neither sequence trainer was seeded: weight initialisation, dropout and dataloader
shuffling were all drawing from an unpinned RNG. Three seeds under identical
folds, `results/seed_variance.csv`:

| | seed 42 | seed 43 | seed 44 | mean | sd | spread |
|---|---|---|---|---|---|---|
| TFT wind | 0.1988 | 0.2228 | 0.1951 | 0.2056 | 0.0150 | **14.2%** |
| TFT solar | 0.2087 | 0.2278 | 0.2085 | 0.2150 | 0.0111 | **9.3%** |
| BiLSTM wind | 0.2606 | 0.2817 | 0.2564 | 0.2662 | 0.0136 | **9.9%** |
| BiLSTM solar | 0.3122 | 0.2939 | 0.3139 | 0.3067 | 0.0111 | **6.8%** |

**The spread dwarfs the gaps the ranking rests on.** The pivotal `lightgbm − tft`
wind gap is 0.3%. The TFT's own wind result varies by 14.2% depending on nothing
but the seed — a factor of forty. Across separate runs the TFT's solar result has
been observed at 0.1962, 0.2044, 0.2087, 0.2225, 0.2278 and 0.2654, which spans
from second place in Table 6.2's solar ranking to last among the ML models. Solar
`ebm − tft` is now ΔnMAE +0.0036, CI [−0.0027, +0.0101] — statistically tied,
where an earlier draw had TFT losing to EBM decisively.

**This is a property of the method, not a defect to be re-run away.** A fourth run
produces a fourth number, not the correct one. The honest statement for §6.1 is
that **a paired significance test is conditional on a particular trained model**.
For the deterministic trees that conditioning is immaterial and the intervals mean
what they appear to mean. For the sequence models it is not, and their rank should
not be read as stable: the between-run spread exceeds most of the between-model
gaps separating them.

**Seeding fixes reproducibility, not variance.** Both trainers now seed per
(technology, fold) and expose a `seed` argument, so the reported run reproduces
exactly — verified by re-running seed 42 and recovering the production numbers
fold-for-fold. Seeding once at function entry was *not* sufficient: the TFT's
production refit runs inside the technology loop, so solar's draw depended on
whether production artifacts were being saved, and reproduced only when that
setting happened to match. Pinning a seed makes one arbitrary draw repeatable. It
does not make it representative, which is why the spread above is reported
alongside it.

Reproduce with:

```bash
uv run -m src.models.train_tft      # seed 42, the reported run
# replicates: train_and_evaluate_tft(seed=N, model_name=f"tft_seed{N}",
#             out_csv="results/seed_variance.csv", save_production=False)
```

---

## 4. Split conformal recalibration (§6.6, §8.3)

`src/evaluation/conformal.py`, wired into the municipal validator, the macro
sequential-fold evaluation, and the dashboard's served interval.

| | PICP before | PICP after | MPIW before | MPIW after |
|---|---|---|---|---|
| KW · solar | 0.333 | **0.806** | 0.266 | 0.387 |
| KW · wind | 0.388 | **0.796** | 0.402 | 0.883 |
| Nauen · solar | 0.160 | **0.816** | 0.244 | 0.855 |
| Nauen · wind | 0.407 | **0.779** | 0.497 | 1.053 |
| macro · wind (3 folds) | 0.62–0.69 | 0.77–0.86 | ~0.46 | ~0.61 |
| macro · solar (3 folds) | 0.79–0.81 | 0.81–0.89 | ~0.44 | ~0.46 |

Mean |PICP − 0.80| falls **0.235 → 0.031**. Macro solar was already
well-calibrated and the correction correctly leaves it almost untouched (Q ≈ 0),
which is a useful sanity signal: the method does not widen what does not need
widening.

§8.3's conformal bullet can be deleted and §6.6 can end on the fix rather than the
limitation.

**Implementation note worth a sentence.** The conformity score is normalised by
the interval half-width, and the *floor* on that half-width is tuned on the
calibration split. A small floor keeps the correction multiplicative (preserving
the day/night shape of solar intervals); a large one recovers textbook additive
CQR. Every candidate reaches the same calibration coverage by construction, so
they are compared on width alone and the sharpest wins. This is not cosmetic: on
Nauen solar the purely multiplicative form reaches nominal coverage only with an
interval **26× mean generation**, versus 0.86× for the selected form — a 31×
sharpening at identical coverage.

**Two caveats that must appear in the text**, since both weaken the guarantee from
theoretical to empirical:

1. The calibration and validation blocks are contiguous in time, not exchangeable
   draws. This restores *approximately* nominal coverage under an assumption of
   stationarity across the split — not the exact finite-sample guarantee.
2. The half-width floor is selected on the calibration split rather than fixed a
   priori. Selection consumes no validation data, but it does forfeit the strict
   a-priori-score condition. Report the coverage as measured on held-out data, not
   as guaranteed by theory.

---

## Reproducing all of it

```bash
uv run -m src.evaluation.export_predictions          # trees + LightGBM q10/q90 OOF
uv run -m src.models.train_bilstm                    # appends BiLSTM OOF
uv run -m src.models.train_tft                       # appends TFT OOF (h=24)
uv run -m src.evaluation.ablation                    # trees + physics_prior_raw
uv run -m src.evaluation.ablation --skip-trees --with-sequence
uv run -m src.evaluation.significance
uv run -m src.validation.municipal_validator         # fits + freezes the conformal Q
uv run -m src.evaluation.conformal_macro
uv run -m src.validation.lomo_transfer
```

Figures are regenerated by notebooks 06 (§1e), 07 (§2d), 08 (ablation) and
12 (§C). Every new module carries a runnable `__main__` self-check.

## Thesis sections to revise

- **§6.1 / Table 6.2** — add CIs; soften the TFT-vs-LightGBM wind claim; add the
  seed-variance table (§3.2) and the caveat that sequence-model ranks are
  conditional on a training run; note the objective correction (§3.1) if the
  earlier LightGBM figures have already been circulated.
- **§6.3, §7.2, §8.2.2** — replace the attribution-only RQ2 answer with the
  ablation numbers and the wind/solar asymmetry.
- **§6.5, §7.4.4** — add the LOMO table; lead with correlation invariance.
- **§6.6** — before/after coverage; end on the fix.
- **§8.3** — delete the conformal future-work bullet (item 4); the feature-ablation
  bullet (item 2) is also now done.
- **§7.5 Threats to validity** — the interval under-coverage threat is now
  mitigated rather than open; the "no spatial hold-out for calibration" threat is
  now measured rather than asserted.
