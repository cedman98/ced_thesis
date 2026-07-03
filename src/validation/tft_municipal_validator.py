"""TFT cross-scale municipal validation (KW).

Loads the CF-target TFT checkpoints, predicts the municipal capacity factor from
the shared scale-free features, and downscales with the same
CF x local_nameplate x k contract as the other models. Appends its rows to the
shared KW validation metrics table.
"""

import logging
import os
import numpy as np
import pandas as pd

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from src.validation.kw_features import (
    build_municipal_features, load_municipalities, calibrate_affine, apply_affine,
)
from src.features.schema import TARGET_CF
from src.validation.municipal_validator import (
    ACTUAL_COL, CAL_FRACTION, CONSOLIDATED_PATH, save_calibration,
)
from src.evaluation import metrics as M

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _predict_cf(ckpt, tech, X):
    tft = TemporalFusionTransformer.load_from_checkpoint(ckpt)
    d = X.copy()
    d['time_idx'] = np.arange(len(d))
    d['group'] = '0'
    for t in ('wind', 'solar'):
        d[TARGET_CF[t]] = 0.0  # dummy unknown-real; overwritten by prediction
    ds = TimeSeriesDataSet.from_parameters(tft.dataset_parameters, d, predict=False)
    dl = ds.to_dataloader(train=False, batch_size=256, num_workers=0)
    out = tft.predict(dl, mode="prediction", return_index=True)
    preds, idx = out[0], out[2]
    # x_to_index returns the time_idx of the FIRST decoder step, so the 24-step
    # horizon covers [time_idx, time_idx + 23] and preds[:, 23] belongs to +23.
    cf_1h = np.clip(preds[:, 23].cpu().numpy(), 0.0, 1.0)
    target_time_idx = idx['time_idx'].values + 23
    m = dict(zip(target_time_idx, cf_1h))
    return pd.Series(d['time_idx'].map(m).values, index=d.index).dropna()


def validate_municipality_tft(muni):
    """Append TFT rows to one municipality's metrics CSV + prediction parquet.
    Returns the TFT metrics DataFrame tagged with `municipality`, or None."""
    slug, out_dir = muni['slug'], muni['results_dir']
    logger.info(f"TFT Cross-Scale Municipal Validation ({slug})")
    X, actuals, caps = build_municipal_features(muni['lat'], muni['lon'], muni['telemetry'])
    cut = int(len(X) * CAL_FRACTION)
    cal_idx, val_idx = X.index[:cut], X.index[cut:]

    is_kw = slug == 'koenigs_wusterhausen'
    metrics_path = f"{out_dir}/{'kw_validation_metrics.csv' if is_kw else 'validation_metrics.csv'}"
    merged_path = f"{out_dir}/{'kw_multi_model_validation.parquet' if is_kw else 'multi_model_validation.parquet'}"
    merged = pd.read_parquet(merged_path) if os.path.exists(merged_path) else actuals.copy()

    rows, calib = [], {}
    for tech in ('wind', 'solar'):
        ckpt = f'models/tft/tft_{tech}_production.ckpt'
        if not os.path.exists(ckpt):
            logger.warning(f"missing {ckpt}; skipping {tech}")
            continue
        cf = _predict_cf(ckpt, tech, X)
        base_kw = cf * caps[tech] * 1000.0
        y = actuals[ACTUAL_COL[tech]]
        scale, offset = calibrate_affine(y.reindex(cal_idx), base_kw.reindex(cal_idx))
        pred_kw = pd.Series(apply_affine(base_kw, scale, offset), index=base_kw.index)
        calib[f'tft_{tech}'] = {'scale': scale, 'offset': offset, 'cap_mw': caps[tech]}
        merged[f'pred_tft_{tech}'] = pred_kw
        yv, pv = y.reindex(val_idx), pred_kw.reindex(val_idx)
        row = M.all_metrics(yv, pv)
        raw = M.all_metrics(yv, base_kw.reindex(val_idx))
        row.update(municipality=slug, model='tft', technology=tech,
                   scale=round(scale, 4), offset=round(offset, 1),
                   nmae_k1=raw['nmae'], mbe_k1=raw['mbe'])
        rows.append(row)

    if not rows:
        logger.warning(f"[{slug}] No TFT checkpoints; nothing to do.")
        return None

    save_calibration(out_dir, calib)
    new = pd.DataFrame(rows)
    if os.path.exists(metrics_path):
        prev = pd.read_csv(metrics_path)
        prev = prev[prev['model'] != 'tft']
        combined = pd.concat([prev, new], ignore_index=True)
    else:
        combined = new
    cols = ['municipality', 'model', 'technology', 'scale', 'offset', 'n', 'nmae', 'mbe',
            'corr', 'picp', 'mpiw_norm', 'nmae_k1', 'mbe_k1', 'mae', 'rmse']
    combined['municipality'] = combined.get('municipality', slug)
    combined = combined[[c for c in cols if c in combined.columns]]
    os.makedirs(out_dir, exist_ok=True)
    combined.to_csv(metrics_path, index=False)
    merged.to_parquet(merged_path)

    print(f"\n--- {slug} Validation incl. TFT (calibrated, held-out) ---")
    print(combined.sort_values(['technology', 'nmae']).to_string(index=False))
    return new


def build_tft_validation_pipeline():
    tft_rows = [r for r in (validate_municipality_tft(m) for m in load_municipalities()) if r is not None]
    if not tft_rows or not os.path.exists(CONSOLIDATED_PATH):
        return
    # Merge TFT rows into the consolidated report produced by municipal_validator.
    prev = pd.read_csv(CONSOLIDATED_PATH)
    prev = prev[prev['model'] != 'tft']
    consolidated = pd.concat([prev, pd.concat(tft_rows, ignore_index=True)], ignore_index=True)
    consolidated.to_csv(CONSOLIDATED_PATH, index=False)
    logger.info(f"Merged TFT rows into {CONSOLIDATED_PATH}")


if __name__ == '__main__':
    build_tft_validation_pipeline()
