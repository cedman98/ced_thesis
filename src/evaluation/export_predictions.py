"""Export honest out-of-fold (OOF) macro predictions for the visualization suite.

The persisted models/*.pkl are refit on the *full* dataset, so predicting with
them is in-sample and would flatter every scatter/time-series. Instead this
re-runs the same PurgedExpandingWindowSplitter used by the trainers and records
each fold's held-out predictions, reusing each trainer's exact params *and
objective* (imported, not copied, so they can't drift — see _builders). Only the
cheap tree/glass-box models are
exported; BiLSTM/TFT are heavy to refit and are already represented in the
aggregate metrics CSV.

Output: results/macro_oof_predictions.parquet — tidy, one row per (timestamp,
technology, model) with CF + reconstructed-MW columns and the physical prior.
"""

import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from interpret.glassbox import ExplainableBoostingRegressor

from src.features.validation_splitter import PurgedExpandingWindowSplitter
from src.features.schema import FEATURE_COLS, TARGET_CF, TARGET_MW, CAP_COLS
from src.evaluation import metrics as M
from src.models.train_lightgbm import PARAMS as LGB_PARAMS, _fit_quantiles, _predict_interval
from src.models.train_comparisons import XGB_PARAMS
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PRIOR_CF = {"wind": "wind_prior_cf", "solar": "solar_prior_cf"}


def _builders():
    """(name -> fresh untrained model) with the trainers' exact params.

    LightGBM must carry the trainer's quantile objective: train_lightgbm's point
    forecast IS the q50 pinball fit, and omitting it here silently fell back to
    LightGBM's default L2 — a different model reported under the same name.
    """
    return {
        "lightgbm": lambda: lgb.LGBMRegressor(objective='quantile', alpha=0.5, **LGB_PARAMS),
        "xgboost": lambda: xgb.XGBRegressor(**XGB_PARAMS),
        "ebm": lambda: ExplainableBoostingRegressor(random_state=42),
    }


def export(matrix_path="data/processed/ml_training_matrix.parquet",
           out_path="results/macro_oof_predictions.parquet"):
    df = pd.read_parquet(matrix_path)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    X = df[FEATURE_COLS]

    splitter = PurgedExpandingWindowSplitter(
        n_splits=4, test_duration=pd.Timedelta(days=365), purge_gap=pd.Timedelta(hours=72)
    )
    builders = _builders()

    out = []
    for tech in ("wind", "solar"):
        y = df[TARGET_CF[tech]]
        for fold, (tr, te) in enumerate(splitter.split(df), start=1):
            idx = df.index[te]
            base = pd.DataFrame({
                "timestamp": idx,
                "technology": tech,
                "fold": fold,
                "prior_cf": df[PRIOR_CF[tech]].iloc[te].to_numpy(),
                "y_cf": y.iloc[te].to_numpy(),
                "cap_mw": df[CAP_COLS[tech]].iloc[te].to_numpy(),
                "y_mw": df[TARGET_MW[tech]].iloc[te].to_numpy(),
            })
            # LightGBM's held-out 80% interval, refit per fold with the trainer's
            # own quantile helpers. These are what macro conformal calibrates on;
            # the point models below leave them NaN.
            qm = _fit_quantiles(X.iloc[tr], y.iloc[tr])
            lo_cf, _, hi_cf = _predict_interval(qm, X.iloc[te])

            for name, make in builders.items():
                m = make()
                m.fit(X.iloc[tr], y.iloc[tr])
                pred_cf = np.clip(m.predict(X.iloc[te]), 0.0, 1.5)
                # The lightgbm builder refits what _fit_quantiles already fit at
                # q50 — deterministic, so it must reproduce it exactly. Compared
                # against the raw q50 rather than _predict_interval's median,
                # which is sorted against quantile crossing. Guards the objective
                # against drifting apart from train_lightgbm again.
                if name == "lightgbm":
                    assert np.allclose(pred_cf, np.clip(qm[0.5].predict(X.iloc[te]), 0.0, 1.5)), \
                        "lightgbm OOF point forecast diverged from the trainer's q50 fit"
                row = base.copy()
                row["model"] = name
                row["pred_cf"] = pred_cf
                row["pred_mw"] = pred_cf * base["cap_mw"].to_numpy()
                row["pred_lo_cf"] = lo_cf if name == "lightgbm" else np.nan
                row["pred_hi_cf"] = hi_cf if name == "lightgbm" else np.nan
                out.append(row)
            logger.info(f"{tech} fold {fold}: {len(idx)} OOF points x {len(builders)} models")

    # Merge rather than overwrite: the BiLSTM/TFT trainers append their own rows
    # to this same file. append_oof also drops the duplicate boundary hour that
    # adjacent purged folds share, so each (model, tech, timestamp) is held out once.
    res = M.append_oof(pd.concat(out, ignore_index=True), path=out_path)
    logger.info(f"Wrote {len(res)} rows -> {out_path}")
    return res


if __name__ == "__main__":
    r = export()
    # Self-check: OOF timestamps are disjoint across folds (each point held out
    # once per model), and ML beats the raw prior on wind CF error.
    per_model = r[r.model == "lightgbm"]
    for tech in ("wind", "solar"):
        t = per_model[per_model.technology == tech]
        assert t["timestamp"].is_unique, f"{tech}: OOF timestamps not unique"
        ml_err = (t.pred_cf - t.y_cf).abs().mean()
        prior_err = (t.prior_cf - t.y_cf).abs().mean()
        print(f"{tech}: OOF nMAE(ML)={ml_err:.4f}  MAE(prior)={prior_err:.4f}  n={len(t)}")
        assert ml_err < prior_err, f"{tech}: ML should beat the raw prior"
    print("export_predictions self-check passed.")
