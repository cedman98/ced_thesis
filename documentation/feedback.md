## Feedback from Professor

- Right now RQ2 is answered entirely through attribution (EBM importances, SHAP, TFT attention in §6.3–6.4). That's a solid triangulation, but it's correlational. Retraining the production models with wind_prior_cf / solar_prior_cf removed and reporting the nMAE penalty — which you propose yourself in §8.3 — turns "the prior dominates the model's internal accounting" into a causal, falsifiable number. No new data or code paths needed.
    - so maybe good to run the feature-ablation study for RQ2

- also, add the leave-one-municipality-out test between your two existing sites

- you can test whether a scale fitted at Königs Wusterhausen transfers to Nauen, which is exactly the situation your 16 unvalidated districts are in
    - so fit the affine contract on one town, apply it to the other, report the result.
    - and also you might need to put some statistical weight behind the model comparisons in Ch. 6

- I mean Table 6.2 reports means and [min, max] across 4 folds, but claims like "TFT marginally outperforms LightGBM on wind" aren't backed by a significance test. With only 4 folds a formal test is limited, but even a simple paired comparison or an explicit uncertainty bound on the metric gap would let you distinguish "meaningfully different" from "within noise". right now the ranking reads more decisive than the evidence strictly shows

- and if time allows, implement the conformal recalibration you already scoped in §8.3, rather than leaving it entirely to future work. 
    - becuase you correctly diagnosed why municipal coverage collapses (the affine step rescales interval width but doesn't re-calibrate it), and you note that split conformal prediction on the existing out-of-fold residuals "would restore nominal coverage without retraining any model." so since the residuals already exist, this could be a real fix rather than a proposal. letting you end that section on "here's the problem and here's the fix that works," which is a stronger note than "here's a known limitation."
