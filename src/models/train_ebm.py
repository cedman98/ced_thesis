"""Explainable Boosting Machine (EBM) pipeline (Capacity-Factor target).

Glass-box model on the wind/solar capacity factor. Because both the priors and
the target are now scale-free CFs in ~[0,1], the municipal feature distribution
falls *inside* the training range, so the EBM no longer linearly extrapolates
into the runaway calibration collapse seen at the local scale.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from interpret.glassbox import ExplainableBoostingRegressor

from src.features.validation_splitter import PurgedExpandingWindowSplitter
from src.features.schema import FEATURE_COLS, TARGET_CF, TARGET_MW, CAP_COLS
from src.evaluation import metrics as M

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _load(matrix_path):
    df = pd.read_parquet(matrix_path)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def train_and_evaluate_ebm(matrix_path: str = 'data/processed/ml_training_matrix.parquet') -> None:
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
            model = ExplainableBoostingRegressor(random_state=42)
            model.fit(X.iloc[tr], y.iloc[tr])
            pred = np.clip(model.predict(X.iloc[te]), 0.0, 1.5)
            cap = df[CAP_COLS[tech]].iloc[te]
            y_mw = df[TARGET_MW[tech]].iloc[te]
            rows += M.cv_rows("ebm", tech, fold, y.iloc[te], pred, cap, y_mw)
            logger.info(f"{tech} EBM fold {fold} | CF nMAE {M.nmae(y.iloc[te], pred):.3f}")

    M.append_cv_metrics(rows)

    results_dir = Path('results'); results_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    for tech in ("wind", "solar"):
        final = ExplainableBoostingRegressor(random_state=42)
        final.fit(X, df[TARGET_CF[tech]])
        g = final.explain_global()
        pd.DataFrame({'Feature': g.data()['names'], 'ImportanceScore': g.data()['scores']}) \
            .sort_values('ImportanceScore', ascending=False) \
            .to_csv(results_dir / f'ebm_{tech}_importance.csv', index=False)
        joblib.dump(final, f'models/ebm_{tech}.pkl')
    logger.info("Saved EBM models + global importances.")


if __name__ == "__main__":
    train_and_evaluate_ebm()
