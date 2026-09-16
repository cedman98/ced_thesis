"""Macro-scale conformal recalibration of the LightGBM 80% interval.

The municipal side of the fix lives in `municipal_validator`; this is the macro
counterpart, which matters because the wind interval under-covers at the 50Hertz
zone scale too (PICP ~0.53 against an 0.80 nominal) — so the collapse seen at the
municipal scale is not purely a downscaling artefact.

Calibration is strictly sequential: Q is fitted on fold f and evaluated on fold
f+1. Fitting and evaluating on the same fold would be in-sample, and calibrating
on a chronologically *later* fold would reintroduce exactly the look-ahead
leakage the purged expanding-window protocol exists to prevent. That leaves
n_folds - 1 honest evaluations per technology.

Input is the per-point OOF parquet, which carries the held-out q10/q90 columns
written by `export_predictions`. No model is retrained.
"""

import logging

import pandas as pd

from src.evaluation import metrics as M
from src.evaluation.conformal import ALPHA, append_coverage, apply_conformal, fit_conformal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OOF_PATH = "results/macro_oof_predictions.parquet"


def macro_report(oof_path=OOF_PATH, alpha=ALPHA):
    df = pd.read_parquet(oof_path)
    df = df[(df.model == "lightgbm") & df.pred_lo_cf.notna()].copy()
    if df.empty:
        raise RuntimeError(
            f"No LightGBM quantile columns in {oof_path} — re-run "
            "`uv run -m src.evaluation.export_predictions` first.")
    df = df.sort_values("timestamp")

    rows = []
    for tech, g in df.groupby("technology"):
        folds = sorted(g.fold.unique())
        for cal_fold, eval_fold in zip(folds, folds[1:]):
            cal, ev = g[g.fold == cal_fold], g[g.fold == eval_fold]
            q, w_floor = fit_conformal(cal.y_cf, cal.pred_lo_cf, cal.pred_hi_cf, alpha)
            lo_c, hi_c = apply_conformal(ev.pred_lo_cf, ev.pred_hi_cf, q, w_floor)
            rows.append(dict(
                scope="macro", municipality=pd.NA, technology=tech, model="lightgbm",
                fold=eval_fold, cal_fold=cal_fold, n=len(ev),
                picp_before=M.picp(ev.y_cf, ev.pred_lo_cf, ev.pred_hi_cf),
                picp_after=M.picp(ev.y_cf, lo_c, hi_c),
                mpiw_norm_before=M.mpiw_norm(ev.y_cf, ev.pred_lo_cf, ev.pred_hi_cf),
                mpiw_norm_after=M.mpiw_norm(ev.y_cf, lo_c, hi_c),
                conformal_q=q, conformal_w_floor=w_floor))
            logger.info(f"{tech} fold {cal_fold}->{eval_fold}: PICP "
                        f"{rows[-1]['picp_before']:.3f} -> {rows[-1]['picp_after']:.3f}")

    append_coverage(rows)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    res = macro_report()
    print("\n=== Macro conformal recalibration (sequential folds, LightGBM 80% interval) ===")
    print(res.drop(columns=["municipality"]).to_string(index=False))

    # Conformal cannot fix a coverage gap it never sees, but it must move coverage
    # toward nominal on average rather than away from it — otherwise the
    # calibration fold is not informative about the next one and the whole
    # sequential scheme is invalid here.
    before = (res.picp_before - (1 - ALPHA)).abs().mean()
    after = (res.picp_after - (1 - ALPHA)).abs().mean()
    assert after < before, f"conformal moved macro coverage away from nominal ({before:.3f} -> {after:.3f})"
    print(f"\nconformal_macro self-check passed: mean |PICP - 0.80| {before:.3f} -> {after:.3f}")
