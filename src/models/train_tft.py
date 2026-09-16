"""Temporal Fusion Transformer pipeline (Capacity-Factor target, purged CV).

Predicts the wind/solar capacity factor. The GroupNormalizer now normalises a
scale-free CF, so applying the trained normaliser to a municipality no longer
injects the multi-GW macro offset that decoupled local predictions.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE
from pytorch_forecasting.data import GroupNormalizer

from src.features.validation_splitter import PurgedExpandingWindowSplitter
from src.features.schema import TARGET_CF, TARGET_MW, CAP_COLS, tft_known_reals
from src.evaluation import metrics as M

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_ENCODER_LENGTH = 72
MAX_PREDICTION_LENGTH = 24


def _dataset(data, target_col, known_reals):
    return TimeSeriesDataSet(
        data, time_idx="time_idx", target=target_col, group_ids=["group"],
        min_encoder_length=MAX_ENCODER_LENGTH, max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=MAX_PREDICTION_LENGTH, max_prediction_length=MAX_PREDICTION_LENGTH,
        time_varying_known_reals=known_reals, time_varying_unknown_reals=[target_col],
        target_normalizer=GroupNormalizer(groups=["group"]),
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
    )


def train_and_evaluate_tft(matrix_path: str = 'data/processed/ml_training_matrix.parquet',
                           known_reals_fn=tft_known_reals, model_name: str = "tft",
                           out_csv: str = None, oof_path: str = None,
                           save_production: bool = True, seed: int = 42) -> None:
    """Purged-CV TFT training.

    The non-default arguments exist for the feature-ablation driver:
    `known_reals_fn` swaps the decoder's known-future feature list, `model_name`
    tags the metric rows so an ablated run cannot collide with the production
    rows, `out_csv` redirects them, and `save_production=False` stops a
    deliberately crippled model from overwriting the production checkpoint.

    `oof_path` dumps per-timestamp held-out predictions for the significance
    tests. The TFT emits 24 overlapping decoder windows per hour; averaging them
    would give it a free ensembling advantage the single-shot models do not have,
    so only the h=24 step is exported — one clean value per timestamp at exactly
    the day-ahead horizon this thesis targets. Its nMAE therefore differs slightly
    from the CV table, which averages over horizons 1-24.
    """
    # Unseeded, this pipeline moved 5-12% between runs (weight init, dropout,
    # dataloader shuffling) — larger than most of the between-model gaps the Ch.6
    # ranking rests on. Seeding does not remove that variance, it pins one draw so
    # the reported numbers are reproducible from the committed code.
    # ponytail: seeds the run, not bitwise-deterministic; Trainer(deterministic=True)
    # if a fold ever has to be reproduced op-for-op (errors on some cuDNN kernels).
    # `seed` is exposed so the spread itself can be measured across replicates —
    # pinning one draw makes the run reproducible, it does not make the rank stable.
    pl.seed_everything(seed, workers=True)

    df = pd.read_parquet(matrix_path)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df['time_idx'] = np.arange(len(df))
    df['group'] = "0"

    splitter = PurgedExpandingWindowSplitter(
        n_splits=4, test_duration=pd.Timedelta(days=365), purge_gap=pd.Timedelta(hours=72)
    )

    rows, oof = [], []
    for ti, tech in enumerate(("wind", "solar")):
        target_col = TARGET_CF[tech]
        known_reals = known_reals_fn(tech)
        logger.info(f"=== TFT {tech.upper()} (target {target_col}) ===")

        for fold, (train_idx, test_idx) in enumerate(splitter.split(df), start=1):
            # Re-seed per (technology, fold) rather than once per run. Seeding only
            # at entry made a fold's draw depend on everything trained before it —
            # the production refit below sits inside this loop, so solar reproduced
            # only when save_production matched, while wind reproduced either way.
            pl.seed_everything(seed + 100 * ti + fold, workers=True)

            # Early-stopping validation = last 10% of the TRAIN window (with its
            # encoder history). The test fold must never drive epoch selection —
            # it previously served as val_loss, leaking test info into the metrics.
            cut = int(len(train_idx) * 0.9)
            df_fit = df.iloc[train_idx[:cut]].copy()
            df_val = df.iloc[max(0, train_idx[cut] - MAX_ENCODER_LENGTH): train_idx[-1] + 1].copy()
            history_start_idx = test_idx[0] - MAX_ENCODER_LENGTH
            df_test = df.iloc[history_start_idx: test_idx[-1] + 1].copy()

            training_data = _dataset(df_fit, target_col, known_reals)
            validation_data = TimeSeriesDataSet.from_dataset(training_data, df_val, predict=False, stop_randomization=True)
            test_data = TimeSeriesDataSet.from_dataset(training_data, df_test, predict=False, stop_randomization=True)
            train_dl = training_data.to_dataloader(train=True, batch_size=128, num_workers=4, persistent_workers=True)
            val_dl = validation_data.to_dataloader(train=False, batch_size=128, num_workers=4, persistent_workers=True)
            test_dl = test_data.to_dataloader(train=False, batch_size=128, num_workers=4, persistent_workers=True)

            # Explicit monitored checkpoint: without it Lightning's default saves
            # the LAST epoch, which early stopping leaves 3 epochs past the best.
            ckpt_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                                      dirpath="models/tft/cv_tmp", filename=f"{tech}_fold{fold}")
            trainer = pl.Trainer(
                max_epochs=20, accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
                callbacks=[EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, mode="min"), ckpt_cb],
                logger=False, enable_progress_bar=False,
            )
            tft = TemporalFusionTransformer.from_dataset(
                training_data, learning_rate=0.03, hidden_size=32, attention_head_size=4,
                dropout=0.1, hidden_continuous_size=16, loss=RMSE(), reduce_on_plateau_patience=4,
            )
            trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

            best = (TemporalFusionTransformer.load_from_checkpoint(ckpt_cb.best_model_path)
                    if ckpt_cb.best_model_path else tft)
            pred = best.predict(test_dl, return_y=True, return_x=bool(oof_path))
            y_pred = np.clip(pred.output.flatten().cpu().numpy(), 0.0, 1.5)
            y_true = pred.y[0].flatten().cpu().numpy()

            rows += M.cv_rows(model_name, tech, fold, y_true, y_pred)
            logger.info(f"{tech} TFT fold {fold} | CF nMAE {M.nmae(y_true, y_pred):.3f} corr {M.pearson_r(y_true, y_pred):.3f}")

            if oof_path:
                # decoder_time_idx is (n_windows, horizon); flattening it matches
                # the flattening of output/y above, so the last column selects the
                # h=24 step of every window.
                tix = pred.x["decoder_time_idx"].cpu().numpy()
                last = tix[:, -1]
                p_last = y_pred.reshape(tix.shape)[:, -1]
                sub = df.iloc[last]
                oof.append(pd.DataFrame({
                    "timestamp": sub.index, "technology": tech, "fold": fold,
                    "prior_cf": sub[f"{tech}_prior_cf"].to_numpy(),
                    "y_cf": sub[target_col].to_numpy(),
                    "cap_mw": sub[CAP_COLS[tech]].to_numpy(),
                    "y_mw": sub[TARGET_MW[tech]].to_numpy(),
                    "model": model_name, "pred_cf": p_last,
                    "pred_mw": p_last * sub[CAP_COLS[tech]].to_numpy(),
                }))

        if not save_production:
            continue

        # Production model on the full dataset + XAI interpretation. Seeded too,
        # so the shipped checkpoint and its attention figures are reproducible.
        pl.seed_everything(seed + 100 * ti, workers=True)
        logger.info(f"Retraining {tech.upper()} TFT on full dataset...")
        full_dataset = _dataset(df, target_col, known_reals)
        full_dl = full_dataset.to_dataloader(train=True, batch_size=128, num_workers=4)
        trainer_full = pl.Trainer(max_epochs=10, accelerator="gpu" if torch.cuda.is_available() else "cpu",
                                  devices=1, logger=False, enable_progress_bar=False)
        final_tft = TemporalFusionTransformer.from_dataset(
            full_dataset, learning_rate=0.03, hidden_size=32, attention_head_size=4,
            dropout=0.1, hidden_continuous_size=16, loss=RMSE(),
        )
        trainer_full.fit(final_tft, train_dataloaders=full_dl)

        try:
            raw = final_tft.predict(full_dl, mode="raw", return_x=True)
            interp = final_tft.interpret_output(raw.output, reduction="sum")
            figs = final_tft.plot_interpretation(interp)
            doc = Path('documentation'); doc.mkdir(parents=True, exist_ok=True)
            for key, name in [('attention', 'attention'), ('static_variables', 'static'),
                              ('encoder_variables', 'encoder_importance'), ('decoder_variables', 'decoder_importance')]:
                if key in figs:
                    figs[key].savefig(doc / f'tft_{tech}_{name}.png', dpi=200, bbox_inches='tight')
        except Exception as e:
            logger.warning(f"TFT interpretation skipped: {e}")

        models_dir = Path('models/tft'); models_dir.mkdir(parents=True, exist_ok=True)
        trainer_full.save_checkpoint(models_dir / f'tft_{tech}_production.ckpt')
        logger.info(f"Saved TFT {tech} production checkpoint.")

    M.append_cv_metrics(rows, **({"csv_path": out_csv} if out_csv else {}))
    if oof:
        M.append_oof(pd.concat(oof, ignore_index=True), path=oof_path)
        logger.info(f"Saved TFT per-timestamp OOF predictions -> {oof_path}")
    logger.info("Saved TFT macro CV metrics.")


if __name__ == "__main__":
    train_and_evaluate_tft(oof_path="results/macro_oof_predictions.parquet")
