"""Leave-one-municipality-out transfer of the affine downscaling contract.

§6.5 validates each town with an affine calibration fitted on *that town's own*
telemetry. The 16 unvalidated Brandenburg districts have no telemetry at all, so
the operationally honest question is: does a (scale, offset) fitted at one
municipality work at another?

This fits at one town and applies it verbatim at the other (both directions),
reporting three variants per (target, model, technology):

  oracle        target's own fitted (scale, offset)   -> upper bound (= the §6.5 number)
  transferred   donor's frozen (scale, offset)        -> what the 16 districts get
  uncalibrated  scale=1, offset=0                     -> raw CF x local nameplate

All three are scored on the *same* window — the target's held-out validation
split — so the three rows are directly comparable. The transferred variant
consumes zero target-local data, so scoring it on the target's held-out split
rather than the full overlap costs nothing and buys comparability.

Key diagnostic: an affine map with positive slope cannot change Pearson r, so
`corr` is identical across all three rows by construction. Any degradation under
transfer is therefore *magnitude* (capacity accounting), never *shape* (forecast
skill) — which is the empirical backing for the §7.4 accounting argument.
"""

import logging
import os

import numpy as np
import pandas as pd

from src.evaluation import metrics as M
from src.validation.kw_features import (
    apply_affine, build_municipal_features, calibrate_affine, load_municipalities,
)
from src.validation.municipal_validator import (
    ACTUAL_COL, CAL_FRACTION, load_calibration, predict_cf_bilstm, predict_cf_tree,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUT_PATH = 'results/lomo_transfer.csv'
MODELS = ('lightgbm', 'xgboost', 'ebm', 'bilstm')


def _base_kw(model_name, tech, X, cap_mw):
    """Uncalibrated municipal generation in kW: CF x local nameplate. None if the
    model's artifacts are missing (BiLSTM is optional)."""
    try:
        cf = predict_cf_bilstm(tech, X) if model_name == 'bilstm' else predict_cf_tree(model_name, tech, X)
    except FileNotFoundError:
        logger.warning(f"{model_name} artifacts not found; skipping")
        return None
    return cf * cap_mw * 1000.0 if len(cf) else None


def _row(variant, y_val, base, scale, offset, val_idx):
    pred = pd.Series(apply_affine(base, scale, offset), index=base.index).reindex(val_idx)
    raw = np.asarray(base.reindex(val_idx), float) * scale + offset  # pre-clip
    r = M.all_metrics(y_val, pred)
    r.update(variant=variant, scale_used=scale, offset_used=offset,
             clipped_frac=float(np.mean(raw < 0)) if len(raw) else np.nan)
    return r


def transfer_report(municipalities=None):
    munis = municipalities or load_municipalities()
    if len(munis) < 2:
        raise RuntimeError(f"LOMO needs >=2 configured municipalities, got {len(munis)}")

    built = {}
    for m in munis:
        X, actuals, caps = build_municipal_features(m['lat'], m['lon'], m['telemetry'])
        cut = int(len(X) * CAL_FRACTION)
        built[m['slug']] = dict(m, X=X, actuals=actuals, caps=caps,
                                cal_idx=X.index[:cut], val_idx=X.index[cut:],
                                donor_params=load_calibration(m['results_dir']))
        logger.info(f"[{m['slug']}] overlap {len(X)}h; cal {cut} / val {len(X) - cut}")

    rows = []
    for tgt in built.values():
        X, actuals, caps = tgt['X'], tgt['actuals'], tgt['caps']
        cal_idx, val_idx = tgt['cal_idx'], tgt['val_idx']
        for tech in ('wind', 'solar'):
            y = actuals[ACTUAL_COL[tech]]
            y_val = y.reindex(val_idx)
            for model_name in MODELS:
                base = _base_kw(model_name, tech, X, caps[tech])
                if base is None:
                    continue
                key = f'{model_name}_{tech}'

                # Oracle: the target's own calibration split (the §6.5 setup).
                s_native, o_native = calibrate_affine(y.reindex(cal_idx), base.reindex(cal_idx))
                variants = [_row('oracle', y_val, base, s_native, o_native, val_idx)]
                variants[0].update(donor=tgt['slug'])

                # Transferred: every *other* municipality's frozen params, verbatim.
                for donor in built.values():
                    if donor['slug'] == tgt['slug']:
                        continue
                    p = donor['donor_params'].get(key)
                    if p is None:
                        logger.warning(f"no frozen {key} params at {donor['slug']}; skipping")
                        continue
                    r = _row('transferred', y_val, base, p['scale'], p['offset'], val_idx)
                    r.update(donor=donor['slug'])
                    variants.append(r)

                # Uncalibrated: no local information at all.
                r = _row('uncalibrated', y_val, base, 1.0, 0.0, val_idx)
                r.update(donor='none')
                variants.append(r)

                for r in variants:
                    r.update(target=tgt['slug'], model=model_name, technology=tech,
                             scale_native=round(s_native, 6),
                             scale_ratio=round(r['scale_used'] / s_native, 4) if s_native else np.nan)
                rows += variants

    cols = ['target', 'donor', 'variant', 'model', 'technology', 'n', 'nmae', 'mbe', 'corr',
            'scale_used', 'offset_used', 'scale_native', 'scale_ratio', 'clipped_frac',
            'mae', 'rmse', 'nrmse']
    df = pd.DataFrame(rows)
    df = df[[c for c in cols if c in df.columns]].sort_values(
        ['technology', 'target', 'model', 'variant'])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    logger.info(f"Wrote {len(df)} rows -> {OUT_PATH}")
    return df


if __name__ == '__main__':
    df = transfer_report()
    print("\n=== Leave-one-municipality-out affine transfer ===")
    print(df.to_string(index=False))

    print("\n--- nMAE by variant (mean over models) ---")
    print(df.pivot_table(index=['technology', 'target'], columns='variant',
                         values='nmae').round(3).to_string())

    # The claim the §7.4 argument rests on: an affine map with positive slope is
    # a monotone linear rescaling, so it cannot change Pearson r. Transfer can
    # therefore only break magnitude, never shape. Assert it rather than claim it.
    # (Only where apply_affine's zero-floor never activated — clipping is
    # non-linear and legitimately does move corr.)
    for (tech, tgt, model), g in df.groupby(['technology', 'target', 'model']):
        clean = g[g.clipped_frac == 0]
        if len(clean) < 2:
            continue
        spread = clean['corr'].max() - clean['corr'].min()
        assert spread < 1e-9, f"{tech}/{tgt}/{model}: corr moved under affine transfer ({spread:.2e})"
    print("\nlomo_transfer self-check passed: corr is invariant under affine transfer.")
