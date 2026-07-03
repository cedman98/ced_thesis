"""Quantify the NWP gap: how much accuracy is lost by driving the model with a
weather *forecast* (NWP) instead of reanalysis *actuals*.

Method (clean isolation): the reanalysis and forecast feature matrices share an
identical 50Hertz capacity-factor target and identical timestamps; only the
weather that feeds the physical priors + weather means differs. So we run the
*same* production LightGBM model (our best performer) on both feature sets over
the common period and compare each prediction to the same actuals. The delta in
MAE/RMSE is therefore attributable purely to weather-forecast uncertainty — not
to a different model, target, or period.

Prereqs (see CLAUDE.md / main.py):
    uv run main.py                      # reanalysis matrix + trained models
    uv run main.py --mode forecast      # forecast (NWP) feature matrix

Output:
    results/nwp_gap_report.csv          # tidy metrics per (tech, weather, scale)
    results/nwp_gap_report.md           # human-readable degradation summary
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd

from src.features.schema import FEATURE_COLS, TARGET_CF, TARGET_MW, CAP_COLS
from src.evaluation import metrics as M

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REANALYSIS_MATRIX = "data/processed/ml_training_matrix.parquet"
FORECAST_MATRIX = "data/processed/ml_training_matrix_forecast.parquet"
CSV_OUT = "results/nwp_gap_report.csv"
MD_OUT = "results/nwp_gap_report.md"


def _predict_cf(model, df):
    """Best-model CF prediction, clipped to the same [0, 1.5] range as training."""
    return np.clip(model.predict(df[FEATURE_COLS]), 0.0, 1.5)


def compute_gap(df_re: pd.DataFrame, df_fc: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Pure metrics computation over the common index. `models` maps tech -> estimator.

    Returns a tidy frame with one row per (technology, weather, scale) plus the
    forecast-vs-reanalysis degradation rows (weather='degradation').
    """
    common = df_re.index.intersection(df_fc.index)
    if len(common) == 0:
        raise ValueError("Reanalysis and forecast matrices share no timestamps.")
    logger.info(f"Common evaluation period: {len(common)} hours "
                f"({common.min()} .. {common.max()})")
    re, fc = df_re.loc[common], df_fc.loc[common]

    rows = []
    for tech, model in models.items():
        y_cf = re[TARGET_CF[tech]]
        y_mw = re[TARGET_MW[tech]]
        cap = re[CAP_COLS[tech]]
        preds = {
            "reanalysis": _predict_cf(model, re),
            "forecast": _predict_cf(model, fc),
        }
        err = {}
        for weather, pred_cf in preds.items():
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
             f"Weather-forecast uncertainty isolated on {common_n} common hours, "
             "same LightGBM model and same 50Hertz actuals for both inputs.", ""]
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


def run() -> pd.DataFrame:
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

    models = {t: joblib.load(f"models/lightgbm_{t}.pkl") for t in ("wind", "solar")}
    report = compute_gap(df_re, df_fc, models)

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
        def predict(self, X):
            return X[FEATURE_COLS].sum(axis=1).values / len(FEATURE_COLS)

    df_re = base.copy()
    true_cf = np.clip(Lin().predict(df_re) + 0.01 * rng.standard_normal(len(idx)), 0, 1.5)
    df_re[TARGET_CF["wind"]] = true_cf
    df_re[TARGET_MW["wind"]] = true_cf * 1000.0
    df_re[CAP_COLS["wind"]] = 1000.0
    # Forecast = reanalysis features perturbed by NWP error.
    df_fc = df_re.copy()
    df_fc[FEATURE_COLS] = base[FEATURE_COLS] + 0.1 * rng.standard_normal(base[FEATURE_COLS].shape)

    rep = compute_gap(df_re, df_fc, {"wind": Lin()})
    deg = rep[(rep.weather == "degradation") & (rep.scale == "cf")]["mae"].iloc[0]
    assert deg >= -1e-9, f"noisier forecast should not beat reanalysis, got {deg}"
    assert set(rep.weather) == {"reanalysis", "forecast", "degradation"}
    print(f"nwp_gap self-check passed: CF MAE degradation = {deg:.4f}")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        demo()
    else:
        run()
