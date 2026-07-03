"""LightGBM Training Pipeline (Capacity-Factor target, quantile regression, purged CV).

Trains LightGBM *quantile* regressors for the wind and solar capacity factor
(not absolute MW) at the 10th / 50th / 90th percentiles, giving a robust 80%
prediction interval on top of the median point forecast — municipal clients need
the interval for grid-balancing and curtailment risk, and the median is L1-optimal
so it is more robust to the fat-tailed errors than a mean regressor.

Runs under a purged expanding-window CV and persists held-out macro test metrics
(median CF/MW scale, plus interval PICP) to results/macro_cv_metrics.csv. Mild
regularisation curbs the extrapolation bias (MBE) that hurts downscaling.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.features.validation_splitter import PurgedExpandingWindowSplitter
from src.features.schema import FEATURE_COLS, TARGET_CF, TARGET_MW, CAP_COLS
from src.evaluation import metrics as M

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Quantiles for the 80% prediction interval: (lower, median, upper).
QUANTILES = (0.1, 0.5, 0.9)


def quantile_model_path(tech, q):
    """Median (q=0.5) is the canonical point-forecast file so SHAP / nwp_gap /
    the other tree validators load it unchanged; q10/q90 are sidecar files."""
    return f'models/lightgbm_{tech}.pkl' if q == 0.5 else f'models/lightgbm_{tech}_q{int(q * 100)}.pkl'


# Regularised params: smaller trees + subsampling + L2 shrink predictions toward
# the training mean, which reduces the systematic bias when extrapolating to a
# municipal feature distribution.
PARAMS = dict(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=7,
    num_leaves=31,
    min_child_samples=60,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)


def _fit_quantiles(X, y):
    """Fit one LGBM quantile regressor per QUANTILE; return {q: model}."""
    models = {}
    for q in QUANTILES:
        m = lgb.LGBMRegressor(objective='quantile', alpha=q, **PARAMS)
        m.fit(X, y)
        models[q] = m
    return models


def _predict_interval(models, X):
    """(lower, median, upper) CF arrays, clipped and monotone-sorted so the
    independently-fit quantiles never cross."""
    P = np.vstack([np.clip(models[q].predict(X), 0.0, 1.5) for q in QUANTILES]).T
    P.sort(axis=1)
    return P[:, 0], P[:, 1], P[:, 2]


def _load(matrix_path):
    df = pd.read_parquet(matrix_path)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def train_and_evaluate_models(matrix_path: str = 'data/processed/ml_training_matrix.parquet') -> None:
    logger.info(f"Loading matrix from {matrix_path}")
    df = _load(matrix_path)
    X = df[FEATURE_COLS]
    logger.info(f"Feature matrix shape: {X.shape}. Target: capacity factor.")

    splitter = PurgedExpandingWindowSplitter(
        n_splits=4, test_duration=pd.Timedelta(days=365), purge_gap=pd.Timedelta(hours=72)
    )

    rows = []
    for tech in ("wind", "solar"):
        y = df[TARGET_CF[tech]]
        for fold, (tr, te) in enumerate(splitter.split(df), start=1):
            models = _fit_quantiles(X.iloc[tr], y.iloc[tr])
            lo, med, hi = _predict_interval(models, X.iloc[te])  # median is the point forecast

            cap = df[CAP_COLS[tech]].iloc[te]
            y_mw = df[TARGET_MW[tech]].iloc[te]
            fold_rows = M.cv_rows("lightgbm", tech, fold, y.iloc[te], med, cap, y_mw)
            # Interval validity on the held-out fold (CF scale) — attach to the cf row.
            fold_rows[0]["picp"] = M.picp(y.iloc[te], lo, hi)
            fold_rows[0]["mpiw_norm"] = M.mpiw_norm(y.iloc[te], lo, hi)
            rows += fold_rows
            r = M.all_metrics(y.iloc[te], med)
            logger.info(f"{tech} fold {fold} | CF nMAE {r['nmae']:.3f} MBE {r['mbe']:.4f} "
                        f"corr {r['corr']:.3f} | PICP {fold_rows[0]['picp']:.3f} "
                        f"MPIW {fold_rows[0]['mpiw_norm']:.3f}")

    M.append_cv_metrics(rows)
    logger.info("Saved LightGBM macro CV metrics to results/macro_cv_metrics.csv")

    # Production quantile models on the full dataset.
    os.makedirs('models', exist_ok=True)
    for tech in ("wind", "solar"):
        models = _fit_quantiles(X, df[TARGET_CF[tech]])
        for q, m in models.items():
            joblib.dump(m, quantile_model_path(tech, q))
    logger.info("Saved production LightGBM quantile models (q10/q50/q90).")


if __name__ == "__main__":
    train_and_evaluate_models()
