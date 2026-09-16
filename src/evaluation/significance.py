"""Statistical weight for the macro model comparisons.

The CV table reports each model's nMAE as a mean over 4 folds with a [min, max]
range. That supports a ranking but not a claim of difference: with 4 folds even a
perfect 4-0 sweep gives a sign-test p of 0.125, so no fold-level test can ever
reach conventional significance. The rankings therefore read more decisively than
the fold evidence strictly licenses.

This module supplies the missing uncertainty from the per-point out-of-fold
predictions, which pair the models on identical timestamps:

  1. Diebold-Mariano on the absolute-error loss differential, with a Newey-West
     (HAC) long-run variance. Hourly renewable residuals are strongly
     autocorrelated; an iid variance would overstate significance by roughly the
     square root of the autocorrelation time.
  2. A moving-block bootstrap CI on the nMAE gap itself, with 7-day blocks. This
     is the distribution-free companion and it is the one to trust if the two
     disagree — DM leans on an asymptotic normal approximation the bootstrap
     does not need.
  3. The fold-level paired summary the CV table already implies, carried
     alongside so the reader can reconcile the two views.

Reporting the effect size next to the p-value is the point. On ~35k hourly
observations a statistically significant gap can still be practically
irrelevant, and that distinction is what the ranking claims actually turn on.
"""

import itertools
import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation import metrics as M

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OOF_PATH = "results/macro_oof_predictions.parquet"
CV_PATH = "results/macro_cv_metrics.csv"
OUT_CSV = "results/model_significance.csv"
BLOCK_HOURS = 168  # 7 days: long enough to carry synoptic weather autocorrelation


def diebold_mariano(e_a, e_b, lag=None):
    """DM statistic and two-sided p for H0: equal absolute-error loss.

    Negative statistic => model A has the lower loss. `lag` defaults to the usual
    n^(1/3) HAC truncation; the Bartlett weights keep the variance estimate
    positive semi-definite.
    """
    d = np.abs(np.asarray(e_a, float)) - np.abs(np.asarray(e_b, float))
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    lag = int(np.floor(n ** (1 / 3))) if lag is None else lag
    dc = d - d.mean()
    s = float(np.mean(dc ** 2))
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * float(np.mean(dc[k:] * dc[:-k]))
    if s <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(s / n)
    return float(stat), float(2 * (1 - stats.t.cdf(abs(stat), n - 1)))


def block_bootstrap_dnmae(y, e_a, e_b, block=BLOCK_HOURS, n_boot=2000, seed=0):
    """Percentile CI for nMAE(A) - nMAE(B) under a moving-block bootstrap.

    Blocks preserve the local autocorrelation an iid bootstrap would destroy.
    Both models are resampled on the *same* block draw, which is what makes the
    comparison paired.
    """
    y, e_a, e_b = (np.asarray(v, float) for v in (y, e_a, e_b))
    n = len(y)
    if n < 2 * block:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offs = np.arange(block)
    out = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + offs).ravel()[:n]
        scale = np.mean(np.abs(y[idx]))
        out[i] = (np.mean(np.abs(e_a[idx])) - np.mean(np.abs(e_b[idx]))) / scale if scale > 0 else np.nan
    return tuple(np.nanpercentile(out, [2.5, 97.5]))


def _residual_matrix(oof_path=OOF_PATH):
    """{technology: (y_series, DataFrame of per-model errors indexed by timestamp)}.

    A 24h-persistence column is derived here so the "ML earns its place over the
    naive baselines" claim gets the same treatment as the model-vs-model ones.
    """
    df = pd.read_parquet(oof_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    out = {}
    for tech, g in df.groupby("technology"):
        pred = g.pivot_table(index="timestamp", columns="model", values="pred_cf")
        y = g.drop_duplicates("timestamp").set_index("timestamp")["y_cf"].sort_index()
        pred["persistence"] = M.persistence_pred(y, pred.index, horizon_hours=24)
        err = pred.sub(y, axis=0)
        out[tech] = (y, err)
    return out


def _fold_summary(cv_path=CV_PATH):
    """Per-fold nMAE by (technology, model) from the CV table — the view Table 6.2
    reports, kept so the per-point results can be reconciled against it."""
    cv = pd.read_csv(cv_path)
    cv = cv[cv["scale"] == "cf"]
    return {tech: g.pivot_table(index="fold", columns="model", values="nmae")
            for tech, g in cv.groupby("technology")}


def compare(oof_path=OOF_PATH, cv_path=CV_PATH, n_boot=2000):
    mats, folds = _residual_matrix(oof_path), _fold_summary(cv_path)
    rows = []
    for tech, (y, err) in mats.items():
        fold_tab = folds.get(tech, pd.DataFrame())
        for a, b in itertools.combinations(sorted(err.columns), 2):
            sub = pd.concat([y.rename("y"), err[a].rename("a"), err[b].rename("b")],
                            axis=1).dropna()  # inner join: TFT loses encoder warm-up hours
            if len(sub) < 100:
                continue
            scale = float(np.mean(np.abs(sub["y"])))
            dnmae = (np.mean(np.abs(sub["a"])) - np.mean(np.abs(sub["b"]))) / scale
            stat, p = diebold_mariano(sub["a"], sub["b"])
            lo, hi = block_bootstrap_dnmae(sub["y"], sub["a"], sub["b"], n_boot=n_boot)

            row = dict(technology=tech, model_a=a, model_b=b, n_hours=len(sub),
                       nmae_a=np.mean(np.abs(sub["a"])) / scale,
                       nmae_b=np.mean(np.abs(sub["b"])) / scale,
                       dnmae=dnmae, dnmae_ci_lo=lo, dnmae_ci_hi=hi,
                       dm_stat=stat, dm_p=p)
            # Relative effect size: a significant gap can still be trivial.
            row["dnmae_rel_%"] = 100 * dnmae / row["nmae_b"] if row["nmae_b"] else np.nan

            if {a, b}.issubset(fold_tab.columns):
                gap = (fold_tab[a] - fold_tab[b]).dropna()
                wins = int((gap < 0).sum())
                decided = int((gap != 0).sum())
                row.update(fold_mean_gap=gap.mean(), fold_min=gap.min(), fold_max=gap.max(),
                           fold_wins_a=wins, fold_n=decided,
                           sign_test_p=(stats.binomtest(wins, decided).pvalue if decided else np.nan))
            rows.append(row)
            logger.info(f"{tech} {a} vs {b}: dnMAE {dnmae:+.4f} "
                        f"[{lo:+.4f}, {hi:+.4f}] DM p={p:.3g}")

    df = pd.DataFrame(rows).sort_values(["technology", "dnmae"])
    df.to_csv(OUT_CSV, index=False)
    logger.info(f"Wrote {len(df)} comparisons -> {OUT_CSV}")
    return df


if __name__ == "__main__":
    res = compare()
    cols = ["technology", "model_a", "model_b", "nmae_a", "nmae_b", "dnmae", "dnmae_rel_%",
            "dnmae_ci_lo", "dnmae_ci_hi", "dm_p", "fold_mean_gap", "sign_test_p", "n_hours"]
    print("\n=== Paired model comparisons (negative dnMAE => model_a is better) ===")
    print(res[[c for c in cols if c in res.columns]].round(4).to_string(index=False))

    # Self-check: a model against a deliberately degraded copy of itself must come
    # out significantly better, and the CI must exclude zero in the right
    # direction. Guards against a sign flip in the loss differential.
    rng = np.random.default_rng(0)
    y = rng.gamma(2.0, 0.2, 6000)
    e_good = rng.normal(0, 0.05, 6000)
    e_bad = e_good + rng.normal(0, 0.20, 6000)
    stat, p = diebold_mariano(e_good, e_bad)
    lo, hi = block_bootstrap_dnmae(y, e_good, e_bad, n_boot=500)
    assert stat < 0 and p < 0.01, (stat, p)
    assert hi < 0, f"CI should exclude zero below it, got [{lo:.4f}, {hi:.4f}]"

    # A model against itself has an identically zero loss differential.
    s0, p0 = diebold_mariano(e_good, e_good)
    assert not np.isfinite(s0) or abs(s0) < 1e-9, s0
    print("\nsignificance self-check passed.")
