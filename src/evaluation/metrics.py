"""Forecast error metrics and naive baselines, shared by macro + municipal eval.

Every metric is nan-safe and normalised the same way so macro (MW) and
municipal (kW) numbers are comparable. Baselines (persistence, climatology)
exist so ML value can actually be proven instead of assumed.
"""

import numpy as np
import pandas as pd


def _clean(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    m = np.isfinite(yt) & np.isfinite(yp)
    return yt[m], yp[m]


def mae(y_true, y_pred):
    yt, yp = _clean(y_true, y_pred)
    return float(np.mean(np.abs(yp - yt))) if len(yt) else np.nan


def rmse(y_true, y_pred):
    yt, yp = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((yp - yt) ** 2))) if len(yt) else np.nan


def mbe(y_true, y_pred):
    """Mean Bias Error (pred - actual). The key number for downscaling."""
    yt, yp = _clean(y_true, y_pred)
    return float(np.mean(yp - yt)) if len(yt) else np.nan


def _scale(yt):
    d = float(np.mean(np.abs(yt)))
    return d if d > 1e-9 else np.nan


def nmae(y_true, y_pred):
    yt, yp = _clean(y_true, y_pred)
    return mae(yt, yp) / _scale(yt) if len(yt) else np.nan


def nrmse(y_true, y_pred):
    yt, yp = _clean(y_true, y_pred)
    return rmse(yt, yp) / _scale(yt) if len(yt) else np.nan


def pearson_r(y_true, y_pred):
    yt, yp = _clean(y_true, y_pred)
    if len(yt) < 2 or np.std(yt) < 1e-9 or np.std(yp) < 1e-9:
        return np.nan
    return float(np.corrcoef(yt, yp)[0, 1])


def all_metrics(y_true, y_pred):
    yt, yp = _clean(y_true, y_pred)
    return {
        "n": int(len(yt)),
        "mae": mae(yt, yp),
        "rmse": rmse(yt, yp),
        "mbe": mbe(yt, yp),
        "nmae": nmae(yt, yp),
        "nrmse": nrmse(yt, yp),
        "corr": pearson_r(yt, yp),
    }


# ----------------------------------------------------------------------------
# Prediction-interval metrics (for the LightGBM quantile forecasts).
# ----------------------------------------------------------------------------
def _clean3(y_true, lower, upper):
    yt = np.asarray(y_true, float)
    lo = np.asarray(lower, float)
    hi = np.asarray(upper, float)
    m = np.isfinite(yt) & np.isfinite(lo) & np.isfinite(hi)
    return yt[m], lo[m], hi[m]


def picp(y_true, lower, upper):
    """Prediction Interval Coverage Probability: fraction of actuals inside
    [lower, upper]. A well-calibrated 80% interval should score ~0.80."""
    yt, lo, hi = _clean3(y_true, lower, upper)
    if not len(yt):
        return np.nan
    return float(np.mean((yt >= lo) & (yt <= hi)))


def mpiw_norm(y_true, lower, upper):
    """Mean prediction-interval width, normalised by mean|y| so it is comparable
    to nMAE. Read alongside PICP: coverage is only meaningful if the interval is
    not trivially wide."""
    yt, lo, hi = _clean3(y_true, lower, upper)
    s = _scale(yt)
    return float(np.mean(hi - lo)) / s if len(yt) and s == s else np.nan


# ----------------------------------------------------------------------------
# Naive baselines. All operate on a pd.Series with a DatetimeIndex.
# ----------------------------------------------------------------------------
def persistence_pred(y_full, target_index, horizon_hours=24):
    """Predict y[t] = y[t - horizon]. y_full must cover target_index minus horizon."""
    y = pd.Series(y_full).sort_index()
    shifted = y.shift(freq=pd.Timedelta(hours=horizon_hours))
    return shifted.reindex(target_index)


def fit_climatology(y_train):
    """Hour-of-day x month mean, fit on the training slice only (no leakage)."""
    y = pd.Series(y_train)
    return y.groupby([y.index.hour, y.index.month]).mean()


def climatology_pred(y_train, target_index):
    clim = fit_climatology(y_train)
    vals = [clim.get((h, m), np.nan) for h, m in zip(target_index.hour, target_index.month)]
    return pd.Series(vals, index=target_index)


# ----------------------------------------------------------------------------
# CV metric recording (shared by every trainer so macro test metrics are saved,
# not just logged, at both CF and reconstructed-MW scale).
# ----------------------------------------------------------------------------
def cv_rows(model, technology, fold, y_cf, pred_cf, cap=None, y_mw=None):
    """One row for the CF-scale metrics, plus one MW-scale row if capacity given."""
    r = all_metrics(y_cf, pred_cf)
    r.update(model=model, technology=technology, fold=fold, scale="cf")
    rows = [r]
    if cap is not None and y_mw is not None:
        pred_mw = np.asarray(pred_cf, dtype=float) * np.asarray(cap, dtype=float)
        rm = all_metrics(y_mw, pred_mw)
        rm.update(model=model, technology=technology, fold=fold, scale="mw")
        rows.append(rm)
    return rows


def rolling_metrics(y_true, y_pred, window_hours=48, step_hours=24):
    """Metrics over sliding windows — the multi-week validation harness that
    replaces the brittle single hardcoded window. Inputs are DatetimeIndexed."""
    s = pd.DataFrame({"y": pd.Series(y_true), "p": pd.Series(y_pred)}).dropna().sort_index()
    if len(s) < 3:
        return pd.DataFrame()
    w, step = pd.Timedelta(hours=window_hours), pd.Timedelta(hours=step_hours)
    out, t, end = [], s.index.min(), s.index.max()
    while t + w <= end + pd.Timedelta(hours=1):
        blk = s[(s.index >= t) & (s.index < t + w)]  # half-open: exactly window_hours
        if len(blk) >= max(3, window_hours // 4):
            m = all_metrics(blk["y"], blk["p"])
            m["window_start"] = t
            out.append(m)
        t += step
    return pd.DataFrame(out)


def append_cv_metrics(rows, csv_path="results/macro_cv_metrics.csv"):
    """Append rows, de-duplicating on (model, technology, fold, scale) keep-last,
    so re-running a single trainer refreshes only its own rows."""
    import os
    new = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        prev = pd.read_csv(csv_path)
        combined = pd.concat([prev, new], ignore_index=True)
    else:
        combined = new
    combined = combined.drop_duplicates(
        subset=["model", "technology", "fold", "scale"], keep="last"
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    combined.to_csv(csv_path, index=False)
    return combined


if __name__ == "__main__":
    # Self-check: perfect prediction ~ 0 error; persistence of a daily cycle is
    # strong; climatology beats the global mean on a seasonal signal.
    idx = pd.date_range("2023-01-01", periods=24 * 90, freq="h", tz="UTC")
    daily = np.sin(2 * np.pi * idx.hour / 24)
    seasonal = np.sin(2 * np.pi * idx.dayofyear / 365)
    y = pd.Series(3 + daily + seasonal + 0.01 * np.random.randn(len(idx)), index=idx)

    assert abs(nmae(y, y)) < 1e-9 and abs(mbe(y, y)) < 1e-9
    assert pearson_r(y, y) > 0.999

    pers = persistence_pred(y, y.index, 24)
    assert pearson_r(y, pers) > 0.9, "persistence should track a daily cycle"

    train, test = y.iloc[: 24 * 60], y.iloc[24 * 60 :]
    clim = climatology_pred(train, test.index)
    assert mae(test, clim) < mae(test, pd.Series(train.mean(), index=test.index))

    # PICP: an interval covering [10th, 90th] pct of a distribution ~ 0.8 coverage,
    # and is wider than a [25th, 75th] interval.
    z = np.random.default_rng(0).normal(size=20000)
    assert abs(picp(z, np.full_like(z, -1.2816), np.full_like(z, 1.2816)) - 0.80) < 0.02
    assert (mpiw_norm(z, np.full_like(z, -1.2816), np.full_like(z, 1.2816))
            > mpiw_norm(z, np.full_like(z, -0.674), np.full_like(z, 0.674)))
    print("metrics.py self-check passed:", all_metrics(test, clim))
