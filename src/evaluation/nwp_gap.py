"""Quantify the NWP gap: how much accuracy is lost by driving the model with a
weather *forecast* (NWP) instead of reanalysis *actuals*.

Method (clean isolation): the reanalysis and forecast feature matrices share an
identical 50Hertz capacity-factor target and identical timestamps; only the
weather that feeds the physical priors + weather means differs. For each purged
CV fold we train a fresh LightGBM (same params as the production trainer) on
the reanalysis train window and predict BOTH feature sets on the held-out test
window, so every compared prediction is out-of-fold. (Reusing the production
model — trained on the full matrix — made the reanalysis side in-sample and
overstated the gap.) The delta in MAE/RMSE is therefore attributable purely to
weather-forecast uncertainty — not to a different model, target, or period.

Prereqs (see CLAUDE.md / main.py):
    uv run main.py                      # reanalysis matrix + trained models
    uv run main.py --mode forecast      # forecast (NWP) feature matrix

Output:
    results/nwp_gap_report.csv          # tidy metrics per (tech, weather, scale)
    results/nwp_gap_report.md           # human-readable degradation summary
"""

import logging
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.schema import FEATURE_COLS, TARGET_CF, TARGET_MW, CAP_COLS, CF_CLIP
from src.features.validation_splitter import PurgedExpandingWindowSplitter
from src.models.train_lightgbm import PARAMS as LGB_PARAMS
from src.evaluation import metrics as M

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REANALYSIS_MATRIX = "data/processed/ml_training_matrix.parquet"
FORECAST_MATRIX = "data/processed/ml_training_matrix_forecast.parquet"
CSV_OUT = "results/nwp_gap_report.csv"
MD_OUT = "results/nwp_gap_report.md"


def _predict_cf(model, df):
    """CF prediction, clipped to the same range as training."""
    return np.clip(model.predict(df[FEATURE_COLS]), 0.0, CF_CLIP)


def compute_gap(df_re: pd.DataFrame, df_fc: pd.DataFrame, factories: dict,
                n_splits: int = 4, test_duration="365D", purge_gap="72h") -> pd.DataFrame:
    """Out-of-fold gap computation over the common index. `factories` maps
    tech -> callable returning a fresh untrained estimator; one is trained per
    purged fold on the reanalysis train window and predicts BOTH feature sets
    on the held-out test window.

    Returns a tidy frame with one row per (technology, weather, scale) plus the
    forecast-vs-reanalysis degradation rows (weather='degradation').
    """
    common = df_re.index.intersection(df_fc.index).sort_values()
    if len(common) == 0:
        raise ValueError("Reanalysis and forecast matrices share no timestamps.")
    logger.info(f"Common evaluation period: {len(common)} hours "
                f"({common.min()} .. {common.max()})")
    re, fc = df_re.loc[common], df_fc.loc[common]
    splitter = PurgedExpandingWindowSplitter(
        n_splits=n_splits, test_duration=pd.to_timedelta(test_duration),
        purge_gap=pd.to_timedelta(purge_gap))

    rows = []
    for tech, make_model in factories.items():
        oof = {"reanalysis": [], "forecast": []}
        held = []
        for tr, te in splitter.split(re):
            model = make_model()
            model.fit(re[FEATURE_COLS].iloc[tr], re[TARGET_CF[tech]].iloc[tr])
            oof["reanalysis"].append(pd.Series(_predict_cf(model, re.iloc[te]), index=re.index[te]))
            oof["forecast"].append(pd.Series(_predict_cf(model, fc.iloc[te]), index=re.index[te]))
            held.append(re.iloc[te])
        held = pd.concat(held)
        y_cf, y_mw, cap = held[TARGET_CF[tech]], held[TARGET_MW[tech]], held[CAP_COLS[tech]]

        err = {}
        for weather in ("reanalysis", "forecast"):
            pred_cf = pd.concat(oof[weather])
            for scale, yt, yp in (
                ("cf", y_cf, pred_cf),
                ("mw", y_mw, pred_cf * cap.values),
            ):
                m = M.all_metrics(yt, yp)
                rows.append(dict(technology=tech, weather=weather, scale=scale,
                                 mae=m["mae"], rmse=m["rmse"], nmae=m["nmae"],
                                 nrmse=m["nrmse"], n=m["n"]))
                err[(weather, scale)] = m
        # Degradation = forecast error - reanalysis error (the NWP gap).
        for scale in ("cf", "mw"):
            d_mae = err[("forecast", scale)]["mae"] - err[("reanalysis", scale)]["mae"]
            d_rmse = err[("forecast", scale)]["rmse"] - err[("reanalysis", scale)]["rmse"]
            base_mae = err[("reanalysis", scale)]["mae"]
            base_rmse = err[("reanalysis", scale)]["rmse"]
            rows.append(dict(
                technology=tech, weather="degradation", scale=scale,
                mae=d_mae, rmse=d_rmse,
                nmae=d_mae / base_mae if base_mae else np.nan,      # relative MAE increase
                nrmse=d_rmse / base_rmse if base_rmse else np.nan,  # relative RMSE increase
                n=err[("forecast", scale)]["n"],
            ))
    return pd.DataFrame(rows)


def _write_markdown(report: pd.DataFrame, common_n: int) -> None:
    lines = ["# NWP Gap Report", "",
             f"Weather-forecast uncertainty isolated on {common_n} out-of-fold hours: "
             "per-fold LightGBM models (purged CV) predict both weather inputs against "
             "the same 50Hertz actuals.", ""]
    for tech in report["technology"].unique():
        deg = report[(report.technology == tech) & (report.weather == "degradation")]
        lines.append(f"## {tech.capitalize()}")
        for _, r in deg.iterrows():
            unit = "CF" if r["scale"] == "cf" else "MW"
            lines.append(
                f"- **{unit}**: MAE +{r['mae']:.4f} ({r['nmae'] * 100:+.1f}%), "
                f"RMSE +{r['rmse']:.4f} ({r['nrmse'] * 100:+.1f}%) vs reanalysis"
            )
        lines.append("")
    lines.append("See `nwp_gap_report.csv` for the full reanalysis/forecast breakdown.")
    with open(MD_OUT, "w") as f:
        f.write("\n".join(lines))


def run(n_splits: int = 4, test_duration: str = "365D") -> pd.DataFrame:
    for path in (REANALYSIS_MATRIX, FORECAST_MATRIX):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Build it first: "
                "`uv run main.py` (reanalysis) and `uv run main.py --mode forecast`."
            )
    df_re = pd.read_parquet(REANALYSIS_MATRIX)
    df_fc = pd.read_parquet(FORECAST_MATRIX)
    for df in (df_re, df_fc):
        df.index = pd.to_datetime(df.index, utc=True)

    # Same estimator family + params as the production point forecast (median
    # quantile), but freshly trained per fold so nothing is in-sample.
    factories = {t: (lambda: lgb.LGBMRegressor(objective="quantile", alpha=0.5, **LGB_PARAMS))
                 for t in ("wind", "solar")}
    report = compute_gap(df_re, df_fc, factories,
                         n_splits=n_splits, test_duration=test_duration)

    os.makedirs("results", exist_ok=True)
    report.to_csv(CSV_OUT, index=False)
    _write_markdown(report, int(report["n"].max()))
    logger.info(f"Wrote {CSV_OUT} and {MD_OUT}")
    logger.info("\n" + report.to_string(index=False))
    return report


def demo() -> None:
    """Self-check on synthetic data: a noisier 'forecast' input must not produce a
    *lower* error than the clean 'reanalysis' input -> degradation >= ~0."""
    idx = pd.date_range("2023-01-01", periods=500, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    base = pd.DataFrame({c: rng.random(len(idx)) for c in FEATURE_COLS}, index=idx)

    class Lin:  # tiny linear stand-in for a trained model
        def fit(self, X, y):
            return self

        def predict(self, X):
            return X[FEATURE_COLS].sum(axis=1).values / len(FEATURE_COLS)

    df_re = base.copy()
    true_cf = np.clip(Lin().predict(df_re) + 0.01 * rng.standard_normal(len(idx)), 0, CF_CLIP)
    df_re[TARGET_CF["wind"]] = true_cf
    df_re[TARGET_MW["wind"]] = true_cf * 1000.0
    df_re[CAP_COLS["wind"]] = 1000.0
    # Forecast = reanalysis features perturbed by NWP error.
    df_fc = df_re.copy()
    df_fc[FEATURE_COLS] = base[FEATURE_COLS] + 0.1 * rng.standard_normal(base[FEATURE_COLS].shape)

    rep = compute_gap(df_re, df_fc, {"wind": Lin}, n_splits=4,
                      test_duration="2D", purge_gap="6h")
    deg = rep[(rep.weather == "degradation") & (rep.scale == "cf")]["mae"].iloc[0]
    assert deg >= -1e-9, f"noisier forecast should not beat reanalysis, got {deg}"
    assert set(rep.weather) == {"reanalysis", "forecast", "degradation"}
    print(f"nwp_gap self-check passed: CF MAE degradation = {deg:.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Quantify the NWP gap (OOF, purged CV).")
    p.add_argument("--selfcheck", action="store_true")
    # Shrink folds when the forecast matrix covers less than 4 years, e.g. a
    # 2-year fetch -> --splits 4 --test-days 120.
    p.add_argument("--splits", type=int, default=4)
    p.add_argument("--test-days", type=int, default=365)
    args = p.parse_args()
    if args.selfcheck:
        demo()
    else:
        run(n_splits=args.splits, test_duration=f"{args.test_days}D")
