"""Split conformal recalibration of prediction intervals (width-normalised CQR).

The LightGBM quantile intervals are calibrated at the macro scale but collapse
under downscaling: the affine contract rescales the interval *width* by `scale`
without re-deriving what that width should be, so municipal PICP lands far below
the 80% nominal. Split conformalized quantile regression fixes this from the
residuals that already exist — no model is retrained.

The conformity score is normalised by the interval half-width:

    w_i = max((hi_i - lo_i) / 2, w_floor)
    E_i = max(lo_i - y_i, y_i - hi_i) / w_i        # < 0 when y is safely inside
    Q   = Quantile_{ceil((n+1)(1-alpha))/n}(E)     # finite-sample corrected
    [lo', hi'] = [lo - Q*w, hi + Q*w]              # Q < 0 legitimately shrinks

Normalising by the width makes the correction *multiplicative*, which preserves
the heteroscedastic shape of the forecast: solar intervals stay near-zero at
night and widen at midday. A plain additive CQR offset would inflate every
night-time hour to the same fixed width, which is both wasteful and visibly
wrong on a reliability plot.

`w_floor` guards the degenerate case (hi == lo, e.g. PV at night): without it
the score is 0/0 when the actual is also zero, and +inf when it is not — the
latter would poison the quantile with a single hour. It is derived from the
calibration data and persisted with Q so fit and apply use an identical floor.

Caveats for the write-up, both of which weaken the theoretical guarantee to an
empirical one:

  1. The calibration and evaluation blocks are contiguous in time, not
     exchangeable draws. This restores *approximately* nominal coverage under an
     assumption of stationarity across the split — not the exact finite-sample
     guarantee conformal theory grants for exchangeable data.
  2. `w_floor` is selected on the calibration split rather than fixed a priori.
     Selection consumes no validation data, but choosing among candidates does
     forfeit the strict a-priori-score condition, so the reported coverage should
     be read as measured on held-out data rather than as guaranteed by theory.
"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ALPHA = 0.2  # 80% nominal interval, matching train_lightgbm.QUANTILES
COVERAGE_PATH = "results/conformal_coverage.csv"


def _arrays(y, lo, hi):
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return y[m], lo[m], hi[m]


# Candidate floors, as multiples of the mean half-width. A tiny floor leaves the
# correction purely multiplicative; a floor far above the typical half-width makes
# w constant, i.e. the correction degenerates to textbook *additive* split-CQR.
# The grid therefore spans both classical regimes and everything between.
FLOOR_GRID = (1e-3, 1e-2, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0, 1024.0)


def _fit_q(y, lo, hi, w, alpha):
    e = np.maximum(lo - y, y - hi) / w
    level = min(np.ceil((len(y) + 1) * (1 - alpha)) / len(y), 1.0)
    return float(np.quantile(e, level, method="higher"))


def fit_conformal(y, lo, hi, alpha=ALPHA, floor_grid=FLOOR_GRID):
    """Fit the interval correction on a calibration split. Returns (Q, w_floor).

    Both numbers must be persisted together: applying a different floor than the
    one used at fit time silently changes the correction.

    `w_floor` is tuned rather than fixed. Every candidate floor is fitted to the
    same nominal coverage on the calibration split by construction, so they are
    compared on the only axis left — mean interval width — and the sharpest wins.
    This matters because the multiplicative form is not always the right one: the
    affine downscaling step rescales the macro interval width by `scale` without
    re-deriving it locally, and where the resulting width no longer tracks the
    local error (municipal solar), a purely multiplicative correction has to
    inflate it enormously to reach coverage. A large floor recovers the additive
    form, which is far sharper there. The selection reads only the calibration
    split, so the validation split remains untouched.
    """
    y, lo, hi = _arrays(y, lo, hi)
    n = len(y)
    if n < 2:
        return 0.0, 0.0
    half = (hi - lo) / 2.0
    mean_half = float(np.mean(half))
    if mean_half <= 0:  # every interval degenerate -> nothing to rescale
        return 0.0, 0.0

    best = None
    for frac in floor_grid:
        w_floor = frac * mean_half
        q = _fit_q(y, lo, hi, np.maximum(half, w_floor), alpha)
        lo_c, hi_c = apply_conformal(lo, hi, q, w_floor)
        width = float(np.mean(hi_c - lo_c))
        if best is None or width < best[0]:
            best = (width, q, w_floor)
    return best[1], best[2]


def apply_conformal(lo, hi, q, w_floor=0.0):
    """Widen (or shrink) an interval by the fitted factor. Lower bound is floored
    at zero — generation is non-negative, same convention as `apply_affine`."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    w = np.maximum((hi - lo) / 2.0, w_floor)
    return np.clip(lo - q * w, 0.0, None), hi + q * w


def append_coverage(rows, csv_path=COVERAGE_PATH):
    """Append before/after coverage rows, de-duplicating on the identity keys so
    re-running either the macro or the municipal side refreshes only its own."""
    new = pd.DataFrame(rows)
    combined = pd.concat([pd.read_csv(csv_path), new], ignore_index=True) \
        if os.path.exists(csv_path) else new
    keys = ["scope", "municipality", "technology", "model", "fold"]
    for k in keys:
        if k not in combined.columns:
            combined[k] = pd.NA
    combined = combined.drop_duplicates(subset=keys, keep="last")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    combined.to_csv(csv_path, index=False)
    return combined


if __name__ == "__main__":
    from src.evaluation.metrics import picp

    rng = np.random.default_rng(0)

    # All fixtures are non-negative, generation-like series: `apply_conformal`
    # floors the lower bound at zero, so a symmetric fixture would be clipped and
    # would not exercise the real code path.

    # 1. Deliberately too-narrow intervals are widened back to ~80% coverage on a
    #    disjoint evaluation split.
    y = rng.gamma(2.0, 500.0, 4000)
    med = float(np.median(y))
    lo, hi = np.full(4000, med * 0.8), np.full(4000, med * 1.2)
    q, wf = fit_conformal(y[:2000], lo[:2000], hi[:2000])
    lo2, hi2 = apply_conformal(lo[2000:], hi[2000:], q, wf)
    cov_before, cov_after = picp(y[2000:], lo[2000:], hi[2000:]), picp(y[2000:], lo2, hi2)
    assert cov_before < 0.5 and abs(cov_after - 0.80) < 0.04, (cov_before, cov_after)

    # 2. Too-wide intervals shrink (Q < 0) rather than being left alone.
    q_wide, _ = fit_conformal(y[:2000], np.zeros(2000), np.full(2000, y.max() * 2))
    assert q_wide < 0, f"over-wide interval should shrink, got Q={q_wide}"

    # 3. Heteroscedastic case (the diurnal PV shape): the correction is
    #    multiplicative, so a zero-width night hour stays near-zero width instead
    #    of being inflated to the mean width.
    amp = np.clip(np.sin(np.linspace(0, 20 * np.pi, 2000)), 0, None)  # half the hours at 0
    yh = np.clip(amp * (1 + 0.5 * rng.normal(size=2000)), 0, None)
    loh, hih = amp * 0.9, amp * 1.1  # far too narrow
    qh, wfh = fit_conformal(yh, loh, hih)
    lo3, hi3 = apply_conformal(loh, hih, qh, wfh)
    assert picp(yh, lo3, hi3) > picp(yh, loh, hih), "conformal must improve coverage here"
    night = amp < 0.01
    assert np.mean(hi3[night] - lo3[night]) < 0.1 * np.mean(hi3 - lo3), \
        "multiplicative correction must not inflate degenerate night-time intervals"

    # 4. When the interval width carries NO information about the local error
    #    (constant error, wildly varying width — what the affine step does to
    #    municipal solar), the selector must fall back to the additive regime:
    #    a large floor, and a far sharper interval than the multiplicative fit.
    wid = rng.uniform(1.0, 100.0, 2000)          # width uncorrelated with error
    yu = 500.0 + rng.normal(0, 50.0, 2000)
    lou, hiu = yu - wid, yu + wid
    q_t, wf_t = fit_conformal(yu, lou, hiu)                     # tuned
    q_m, wf_m = fit_conformal(yu, lou, hiu, floor_grid=(1e-3,))  # forced multiplicative
    w_t = np.subtract(*reversed(apply_conformal(lou, hiu, q_t, wf_t))).mean()
    w_m = np.subtract(*reversed(apply_conformal(lou, hiu, q_m, wf_m))).mean()
    assert wf_t > wf_m and w_t < w_m, f"tuner should pick additive here ({w_t:.1f} vs {w_m:.1f})"
    assert picp(yu, *apply_conformal(lou, hiu, q_t, wf_t)) > 0.78

    # 5. The all-degenerate path (PV at night: lo == hi == y == 0) is a no-op,
    #    not a nan or a divide-by-zero.
    z = np.zeros(100)
    q0, wf0 = fit_conformal(z, z, z)
    assert q0 == 0.0 and wf0 == 0.0
    l0, h0 = apply_conformal(z, z, q0, wf0)
    assert np.all(l0 == 0) and np.all(h0 == 0)

    print(f"conformal self-check passed: PICP {cov_before:.3f} -> {cov_after:.3f} (Q={q:.3f})")
