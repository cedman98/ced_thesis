"""Macro (50Hertz) naive baselines under the same purged CV as the models.

Persistence (t-24h) and climatology (hour x month) — the bar ML must clear.
Writes rows into results/macro_cv_metrics.csv alongside the trained models.
"""

import logging
import numpy as np
import pandas as pd

from src.features.validation_splitter import PurgedExpandingWindowSplitter
from src.features.schema import TARGET_CF, TARGET_MW, CAP_COLS
from src.evaluation import metrics as M

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run(matrix_path: str = "data/processed/ml_training_matrix.parquet") -> None:
    df = pd.read_parquet(matrix_path)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    splitter = PurgedExpandingWindowSplitter(
        n_splits=4, test_duration=pd.Timedelta(days=365), purge_gap=pd.Timedelta(hours=72)
    )

    all_rows, oof_rows = [], []
    for tech in ("wind", "solar"):
        cf_col, mw_col, cap_col = TARGET_CF[tech], TARGET_MW[tech], CAP_COLS[tech]
        for fold, (tr, te) in enumerate(splitter.split(df), start=1):
            train, test = df.iloc[tr], df.iloc[te]
            y_cf = test[cf_col]
            cap = test[cap_col]
            y_mw = test[mw_col]

            pers = M.persistence_pred(df[cf_col], test.index, horizon_hours=24)
            clim = M.climatology_pred(train[cf_col], test.index)

            all_rows += M.cv_rows("persistence", tech, fold, y_cf, pers, cap, y_mw)
            all_rows += M.cv_rows("climatology", tech, fold, y_cf, clim, cap, y_mw)

            # Per-point climatology predictions, fit leak-free on this fold's train
            # slice only (unlike persistence's pooled lag, a mean over pooled OOF
            # rows would let later folds leak into earlier ones' hour x month
            # average) -> lets significance.py pair it against the other models.
            oof_rows.append(pd.DataFrame({
                "timestamp": test.index, "technology": tech, "fold": fold, "model": "climatology",
                "y_cf": y_cf.to_numpy(), "pred_cf": clim.to_numpy(),
                "cap_mw": cap.to_numpy(), "y_mw": y_mw.to_numpy(),
            }))

    M.append_oof(pd.concat(oof_rows, ignore_index=True))
    out = M.append_cv_metrics(all_rows)
    logger.info("Saved macro baselines. Mean MW nMAE by model/tech:")
    mw = out[out["scale"] == "mw"]
    print(
        mw.groupby(["model", "technology"])[["nmae", "mbe", "corr"]]
        .mean().round(3).to_string()
    )


if __name__ == "__main__":
    run()
