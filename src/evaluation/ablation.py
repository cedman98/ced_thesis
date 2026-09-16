"""Feature ablation: how much of the forecast skill is the physics prior? (RQ2)

RQ2 is otherwise answered only by attribution — SHAP, EBM importances, TFT
attention — which reports how the model *internally* accounts for its inputs.
That is correlational: a feature can dominate an attribution ranking and still be
replaceable by its correlates. Retraining without it and measuring the error
penalty is the causal, falsifiable version of the same claim.

Three variants, all under the identical purged expanding-window CV used by every
trainer, so the numbers sit directly alongside the main CV table:

    full         the production feature set (schema.FEATURE_COLS)
    no_prior     both physics priors removed -> raw weather + cyclical time only
    prior_only   the technology's own prior, nothing else

`prior_only` earns its runs: with `no_prior` alone a reviewer can object that the
weather features are simply redundant with the prior, so removing it proves
nothing about the prior itself. The three together decompose the skill — what the
prior carries alone, and what the raw weather cannot recover without it.

Trees are rebuilt from the trainers' own parameter dicts (via
export_predictions._builders) so they cannot drift from production. The sequence
models are trained through their real trainers with `save_production=False`, so
a deliberately crippled model can never overwrite a production artifact.

    uv run -m src.evaluation.ablation                  # trees only (~minutes)
    uv run -m src.evaluation.ablation --with-sequence  # + BiLSTM and TFT (GPU, hours)
"""

import argparse
import logging

import numpy as np
import pandas as pd

from src.evaluation import metrics as M
from src.evaluation.export_predictions import _builders
from src.features.schema import (
    CAP_COLS, FEATURE_COLS, TARGET_CF, TARGET_MW, CF_CLIP, tft_known_reals,
)
from src.features.validation_splitter import PurgedExpandingWindowSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUT_CSV = "results/ablation_metrics.csv"
PRIOR_CF = {"wind": "wind_prior_cf", "solar": "solar_prior_cf"}

# feature list per (variant, technology). Kept as callables because prior_only is
# technology-dependent while the other two are not.
VARIANTS = {
    "full": lambda tech: list(FEATURE_COLS),
    "no_prior": lambda tech: [c for c in FEATURE_COLS if not c.endswith("_prior_cf")],
    "prior_only": lambda tech: [PRIOR_CF[tech]],
}


def _splitter():
    return PurgedExpandingWindowSplitter(
        n_splits=4, test_duration=pd.Timedelta(days=365), purge_gap=pd.Timedelta(hours=72)
    )


def run_trees(matrix_path="data/processed/ml_training_matrix.parquet", variants=None):
    df = pd.read_parquet(matrix_path)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    splitter, rows = _splitter(), []

    for variant in (variants or VARIANTS):
        for tech in ("wind", "solar"):
            cols = VARIANTS[variant](tech)
            X, y = df[cols], df[TARGET_CF[tech]]
            for fold, (tr, te) in enumerate(splitter.split(df), start=1):
                for name, make in _builders().items():
                    m = make()
                    m.fit(X.iloc[tr], y.iloc[tr])
                    pred = np.clip(m.predict(X.iloc[te]), 0.0, CF_CLIP)
                    rows += M.cv_rows(f"{name}_{variant}", tech, fold, y.iloc[te], pred,
                                      df[CAP_COLS[tech]].iloc[te], df[TARGET_MW[tech]].iloc[te])
            logger.info(f"{variant}/{tech}: {len(cols)} features {cols}")
    M.append_cv_metrics(rows, csv_path=OUT_CSV)
    return rows


# The TFT decoder takes only its OWN technology's prior as a known-future real
# (schema.tft_known_reals), not both — so its `full` baseline must be that, not
# FEATURE_COLS, or the "penalty" would be measured against a model the thesis
# never trained.
TFT_VARIANTS = {
    "full": tft_known_reals,
    "no_prior": lambda tech: list(FEATURE_COLS[2:]),
    "prior_only": lambda tech: [PRIOR_CF[tech]],
}

# BiLSTM builds one feature matrix for both technologies inside a single call, so
# it cannot express a technology-specific `prior_only`. Rather than redefine that
# variant to mean "both priors" for one model only — which would make the column
# mean two different things in the same table — BiLSTM simply has no prior_only
# row. `full` and `no_prior` are technology-independent and run unchanged.
BILSTM_VARIANTS = ("full", "no_prior")


def run_raw_prior(matrix_path="data/processed/ml_training_matrix.parquet"):
    """The physics prior used directly as the forecast — no learner at all.

    This is the reference point the three variants are otherwise missing.
    `prior_only` still fits a model *on* the prior, so without this row the table
    cannot separate "the prior carries the signal" from "the learner corrects the
    prior". Evaluated on the identical folds; nothing is fitted.
    """
    df = pd.read_parquet(matrix_path)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    rows = []
    for tech in ("wind", "solar"):
        y, prior = df[TARGET_CF[tech]], df[PRIOR_CF[tech]]
        for fold, (_, te) in enumerate(_splitter().split(df), start=1):
            rows += M.cv_rows("physics_prior_raw", tech, fold, y.iloc[te], prior.iloc[te],
                              df[CAP_COLS[tech]].iloc[te], df[TARGET_MW[tech]].iloc[te])
    M.append_cv_metrics(rows, csv_path=OUT_CSV)
    return rows


def run_sequence(variants=None):
    """BiLSTM + TFT under the same variants. Slow (GPU); never touches production
    artifacts because save_production=False."""
    from src.models.train_bilstm import train_and_evaluate_bilstm
    from src.models.train_tft import train_and_evaluate_tft

    for variant in (variants or VARIANTS):
        logger.info(f"=== sequence models, variant={variant} ===")
        if variant in BILSTM_VARIANTS:
            train_and_evaluate_bilstm(feature_cols=VARIANTS[variant]("wind"),
                                      model_name=f"bilstm_{variant}", out_csv=OUT_CSV,
                                      save_production=False)
        train_and_evaluate_tft(known_reals_fn=TFT_VARIANTS[variant],
                               model_name=f"tft_{variant}", out_csv=OUT_CSV,
                               save_production=False)


def summarise(csv_path=OUT_CSV):
    """nMAE per (model, variant, technology) and the penalty for dropping the prior."""
    df = pd.read_csv(csv_path)
    df = df[df["scale"] == "cf"].copy()
    # Variant names themselves contain underscores, so match on the known suffixes
    # (longest first) rather than splitting on the last one.
    def split_name(m):
        for v in sorted(VARIANTS, key=len, reverse=True):
            if m.endswith("_" + v):
                return m[: -(len(v) + 1)], v
        return m, "unknown"

    df[["base", "variant"]] = [split_name(m) for m in df["model"]]

    raw = df[df["model"] == "physics_prior_raw"].groupby("technology")["nmae"].mean()
    piv = df[df["variant"] != "unknown"].pivot_table(
        index=["technology", "base"], columns="variant", values="nmae")
    piv["penalty_%"] = (piv["no_prior"] / piv["full"] - 1) * 100
    if "prior_only" in piv:
        piv["prior_only_gap_%"] = (piv["prior_only"] / piv["full"] - 1) * 100
    if len(raw):
        # The learner-free reference: everything else is measured against it.
        piv["prior_raw"] = [raw.get(t, np.nan) for t, _ in piv.index]
        piv["ml_gain_vs_raw_%"] = (piv["full"] / piv["prior_raw"] - 1) * 100
    return piv.round(4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-sequence", action="store_true",
                    help="also ablate BiLSTM + TFT (GPU, hours)")
    ap.add_argument("--skip-trees", action="store_true",
                    help="sequence models only (the tree run is already done)")
    ap.add_argument("--variants", nargs="*", default=None, choices=list(VARIANTS))
    args = ap.parse_args()

    if not args.skip_trees:
        run_trees(variants=args.variants)
        run_raw_prior()
    if args.with_sequence:
        run_sequence(variants=args.variants)

    piv = summarise()
    print("\n=== Feature ablation: nMAE by variant (CF scale, mean over 4 purged folds) ===")
    print(piv.to_string())

    # The whole RQ2 narrative rests on this ordering. If removing the physics
    # prior ever *helps*, the thesis claim is wrong and this must fail loudly
    # rather than quietly print a surprising table.
    if {"full", "no_prior"}.issubset(piv.columns):
        for (tech, base), r in piv.iterrows():
            assert r["full"] <= r["no_prior"], \
                f"{base}/{tech}: dropping the prior improved nMAE ({r['full']:.4f} -> {r['no_prior']:.4f})"
        print("\nablation self-check passed: the physics prior helps every model on both technologies.")
