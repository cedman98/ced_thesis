"""Cross-scale municipal validation for Königs Wusterhausen.

Downscaling contract (fixed): macro model predicts a *capacity factor* from
scale-free features; local generation = CF x local_nameplate x k, where k is a
single per-(model,technology) constant calibrated on a held-out split to
reconcile the CF definition and local fleet composition. Calibrated on the
first 40% of the telemetry overlap, evaluated on the remaining 60%
(no hardcoded window).
"""

import json
import logging
import os
import joblib
import numpy as np
import pandas as pd
import torch

from src.validation.kw_features import (
    build_municipal_features, load_municipalities, calibrate_affine, apply_affine,
)
from src.features.schema import FEATURE_COLS, CF_CLIP
from src.models.train_bilstm import BiLSTM
from src.models.train_lightgbm import QUANTILES, quantile_model_path
from src.evaluation import metrics as M

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ACTUAL_COL = {'wind': 'windkraft', 'solar': 'photovoltaik'}
CAL_FRACTION = 0.4  # first 40% calibrates k, rest validates


def predict_cf_tree(model_name, tech, X):
    model = joblib.load(f'models/{model_name}_{tech}.pkl')
    return pd.Series(np.clip(model.predict(X[FEATURE_COLS]), 0.0, CF_CLIP), index=X.index)


def predict_cf_lightgbm_interval(tech, X):
    """LightGBM quantile forecasts: (lower, median, upper) CF series (q10/q50/q90),
    clipped to the training bound and monotone-sorted so the quantiles never cross."""
    P = np.vstack([np.clip(joblib.load(quantile_model_path(tech, q)).predict(X[FEATURE_COLS]), 0.0, CF_CLIP)
                   for q in QUANTILES]).T
    P.sort(axis=1)
    return (pd.Series(P[:, 0], index=X.index), pd.Series(P[:, 1], index=X.index),
            pd.Series(P[:, 2], index=X.index))


def _evaluate_interval(tech, lo_cf, med_cf, hi_cf, actuals, cap_mw, cal_idx, val_idx):
    """Downscale + calibrate the LightGBM median, then apply the SAME affine to the
    lower/upper bounds (scale>0 keeps the interval ordered). Adds PICP + width."""
    y = actuals[ACTUAL_COL[tech]]
    base_med = med_cf * cap_mw * 1000.0
    scale, offset = calibrate_affine(y.reindex(cal_idx), base_med.reindex(cal_idx))
    med_kw = pd.Series(apply_affine(base_med, scale, offset), index=base_med.index)
    lo_kw = pd.Series(apply_affine(lo_cf * cap_mw * 1000.0, scale, offset), index=med_cf.index)
    hi_kw = pd.Series(apply_affine(hi_cf * cap_mw * 1000.0, scale, offset), index=med_cf.index)

    yv = y.reindex(val_idx)
    row = M.all_metrics(yv, med_kw.reindex(val_idx))
    raw = M.all_metrics(yv, base_med.reindex(val_idx))
    row.update(model='lightgbm', technology=tech, scale=round(scale, 4), offset=round(offset, 1),
               nmae_k1=raw['nmae'], mbe_k1=raw['mbe'],
               picp=M.picp(yv, lo_kw.reindex(val_idx), hi_kw.reindex(val_idx)),
               mpiw_norm=M.mpiw_norm(yv, lo_kw.reindex(val_idx), hi_kw.reindex(val_idx)))
    return row, med_kw, lo_kw, hi_kw, {'scale': scale, 'offset': offset, 'cap_mw': cap_mw}


def predict_cf_bilstm(tech, X):
    sc = joblib.load('models/bilstm_scalers.pkl')
    meta = sc['meta']
    model = BiLSTM(meta['input_size'], meta['hidden'], meta['layers'])
    model.load_state_dict(torch.load(f'models/bilstm_{tech}.pth', map_location='cpu'))
    model.eval()
    Xs = sc['X'].transform(X[FEATURE_COLS])
    seq = meta['seq_len']
    if len(Xs) <= seq:
        return pd.Series(dtype=float)
    windows = np.stack([Xs[i:i + seq] for i in range(len(Xs) - seq)])
    with torch.no_grad():
        out = model(torch.tensor(windows, dtype=torch.float32)).squeeze(-1).numpy()
    cf = sc[tech].inverse_transform(out.reshape(-1, 1)).flatten()
    return pd.Series(np.clip(cf, 0.0, CF_CLIP), index=X.index[seq:])


def _evaluate(model_name, tech, cf, actuals, cap_mw, cal_idx, val_idx):
    base_kw = cf * cap_mw * 1000.0  # uncalibrated CF downscaling (scale=1, offset=0)
    y = actuals[ACTUAL_COL[tech]]
    scale, offset = calibrate_affine(y.reindex(cal_idx), base_kw.reindex(cal_idx))
    pred_kw = pd.Series(apply_affine(base_kw, scale, offset), index=base_kw.index)
    yv = y.reindex(val_idx)
    row = M.all_metrics(yv, pred_kw.reindex(val_idx))
    raw = M.all_metrics(yv, base_kw.reindex(val_idx))  # before local calibration
    row.update(model=model_name, technology=tech,
               scale=round(scale, 4), offset=round(offset, 1),
               nmae_k1=raw['nmae'], mbe_k1=raw['mbe'])
    return row, pred_kw, {'scale': scale, 'offset': offset, 'cap_mw': cap_mw}


CONSOLIDATED_PATH = 'results/municipal_validation_metrics.csv'
CALIB_NAME = 'affine_calibration.json'  # per-municipality fitted (scale, offset)


def load_calibration(out_dir):
    """Frozen per-(model,tech) affine params for a municipality, or {} if unfit."""
    p = os.path.join(out_dir, CALIB_NAME)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_calibration(out_dir, calib):
    """Merge fitted affine params into out_dir/affine_calibration.json (keep-last),
    so the tree/BiLSTM and TFT validators can each write their own keys."""
    os.makedirs(out_dir, exist_ok=True)
    merged = load_calibration(out_dir)
    merged.update({k: {kk: round(float(vv), 6) for kk, vv in v.items()} for k, v in calib.items()})
    with open(os.path.join(out_dir, CALIB_NAME), 'w') as f:
        json.dump(merged, f, indent=2)
    return merged


def validate_municipality(muni):
    """Run all tree/BiLSTM models + persistence baseline for one municipality.

    Writes the per-municipality metrics CSV and per-hour prediction parquet into
    its results_dir; returns the metrics DataFrame tagged with `municipality`.
    """
    slug, out_dir = muni['slug'], muni['results_dir']
    logger.info(f"Cross-Scale Municipal Validation ({slug})")
    X, actuals, caps = build_municipal_features(muni['lat'], muni['lon'], muni['telemetry'])
    logger.info(f"[{slug}] Overlap: {len(X)} hourly steps {X.index.min()} -> {X.index.max()}")

    cut = int(len(X) * CAL_FRACTION)
    cal_idx, val_idx = X.index[:cut], X.index[cut:]

    merged = actuals.copy()
    results, calib = [], {}
    for tech in ('wind', 'solar'):
        # LightGBM: quantile regression -> median point forecast + 80% interval.
        lo_cf, med_cf, hi_cf = predict_cf_lightgbm_interval(tech, X)
        row, med_kw, lo_kw, hi_kw, params = _evaluate_interval(
            tech, lo_cf, med_cf, hi_cf, actuals, caps[tech], cal_idx, val_idx)
        results.append(row)
        calib[f'lightgbm_{tech}'] = params
        merged[f'pred_lightgbm_{tech}'] = med_kw
        merged[f'pred_lightgbm_{tech}_lower'] = lo_kw
        merged[f'pred_lightgbm_{tech}_upper'] = hi_kw

        for model_name in ('xgboost', 'ebm'):
            cf = predict_cf_tree(model_name, tech, X)
            row, pred, params = _evaluate(model_name, tech, cf, actuals, caps[tech], cal_idx, val_idx)
            results.append(row)
            calib[f'{model_name}_{tech}'] = params
            merged[f'pred_{model_name}_{tech}'] = pred
        try:
            cf = predict_cf_bilstm(tech, X)
            if len(cf):
                row, pred, params = _evaluate('bilstm', tech, cf, actuals, caps[tech], cal_idx, val_idx)
                results.append(row)
                calib[f'bilstm_{tech}'] = params
                merged[f'pred_bilstm_{tech}'] = pred
        except FileNotFoundError:
            logger.warning("BiLSTM artifacts not found; skipping.")

        # Municipal persistence baseline (t-24h), same validation split.
        pers = actuals[ACTUAL_COL[tech]].shift(24)
        yv = actuals[ACTUAL_COL[tech]].reindex(val_idx)
        base = M.all_metrics(yv, pers.reindex(val_idx))
        base.update(model='persistence', technology=tech, scale=1.0, offset=0.0)
        results.append(base)

    cols = ['municipality', 'model', 'technology', 'scale', 'offset', 'n', 'nmae', 'mbe',
            'corr', 'picp', 'mpiw_norm', 'nmae_k1', 'mbe_k1', 'mae', 'rmse']
    res_df = pd.DataFrame(results)
    res_df['municipality'] = slug
    res_df = res_df[[c for c in cols if c in res_df.columns]]
    print(f"\n--- {slug} Cross-Scale Validation (calibrated, held-out window) ---")
    print(res_df.sort_values(['technology', 'nmae']).to_string(index=False))

    os.makedirs(out_dir, exist_ok=True)
    # KW keeps its historical filenames (notebooks 05/07); others use generic names.
    metrics_name = 'kw_validation_metrics.csv' if slug == 'koenigs_wusterhausen' else 'validation_metrics.csv'
    merged_name = 'kw_multi_model_validation.parquet' if slug == 'koenigs_wusterhausen' else 'multi_model_validation.parquet'
    res_df.to_csv(f'{out_dir}/{metrics_name}', index=False)
    merged.to_parquet(f'{out_dir}/{merged_name}')
    # Persist the fitted per-(model,tech) affine params so future inference applies
    # this municipal bias correction without refitting. TFT rows are appended by
    # tft_municipal_validator into the same file.
    save_calibration(out_dir, calib)
    logger.info(f"[{slug}] Saved metrics to {out_dir}/{metrics_name}")
    return res_df


def build_validation_pipeline():
    all_rows = [validate_municipality(m) for m in load_municipalities()]
    consolidated = pd.concat(all_rows, ignore_index=True)
    os.makedirs('results', exist_ok=True)
    consolidated.to_csv(CONSOLIDATED_PATH, index=False)
    print("\n=== Consolidated Municipal Validation (all configured municipalities) ===")
    print(consolidated.sort_values(['technology', 'municipality', 'nmae']).to_string(index=False))
    logger.info(f"Saved consolidated report to {CONSOLIDATED_PATH}")


if __name__ == '__main__':
    build_validation_pipeline()
