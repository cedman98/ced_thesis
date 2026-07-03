"""XGBoost + EBM comparative pipeline (Capacity-Factor target, purged CV).

Trains XGBoost (GBDT baseline) and Explainable Boosting Machines (glass-box) on
the wind/solar capacity factor, records held-out macro metrics, and serialises
full-dataset models for downstream XAI.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from interpret.glassbox import ExplainableBoostingRegressor

from src.features.validation_splitter import PurgedExpandingWindowSplitter
from src.features.schema import FEATURE_COLS, TARGET_CF, TARGET_MW, CAP_COLS
from src.evaluation import metrics as M

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

XGB_PARAMS = dict(
    n_estimators=600, learning_rate=0.03, max_depth=6,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, random_state=42, n_jobs=-1,
)


def _load(matrix_path):
    df = pd.read_parquet(matrix_path)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def train_and_evaluate_comparisons(matrix_path: str = 'data/processed/ml_training_matrix.parquet') -> None:
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Training matrix not found at: {matrix_path}")
    df = _load(matrix_path)
    X = df[FEATURE_COLS]
    logger.info(f"Loaded features {X.shape}. Target: capacity factor.")

    splitter = PurgedExpandingWindowSplitter(
        n_splits=4, test_duration=pd.Timedelta(days=365), purge_gap=pd.Timedelta(hours=72)
    )

    rows = []
    for tech in ("wind", "solar"):
        y = df[TARGET_CF[tech]]
        for fold, (tr, te) in enumerate(splitter.split(df), start=1):
            cap = df[CAP_COLS[tech]].iloc[te]
            y_mw = df[TARGET_MW[tech]].iloc[te]

            xgbm = xgb.XGBRegressor(**XGB_PARAMS)
            xgbm.fit(X.iloc[tr], y.iloc[tr])
            xpred = np.clip(xgbm.predict(X.iloc[te]), 0.0, 1.5)
            rows += M.cv_rows("xgboost", tech, fold, y.iloc[te], xpred, cap, y_mw)

            ebm = ExplainableBoostingRegressor(random_state=42)
            ebm.fit(X.iloc[tr], y.iloc[tr])
            epred = np.clip(ebm.predict(X.iloc[te]), 0.0, 1.5)
            rows += M.cv_rows("ebm", tech, fold, y.iloc[te], epred, cap, y_mw)

            logger.info(
                f"{tech} fold {fold} | XGB nMAE {M.nmae(y.iloc[te], xpred):.3f} "
                f"EBM nMAE {M.nmae(y.iloc[te], epred):.3f}"
            )

    M.append_cv_metrics(rows)
    logger.info("Saved XGBoost/EBM macro CV metrics.")

    os.makedirs('models', exist_ok=True)
    for tech in ("wind", "solar"):
        y = df[TARGET_CF[tech]]
        fx = xgb.XGBRegressor(**XGB_PARAMS); fx.fit(X, y)
        joblib.dump(fx, f'models/xgboost_{tech}.pkl')
        fe = ExplainableBoostingRegressor(random_state=42); fe.fit(X, y)
        joblib.dump(fe, f'models/ebm_{tech}.pkl')
    logger.info("Saved production XGBoost + EBM models.")


if __name__ == "__main__":
    train_and_evaluate_comparisons()
