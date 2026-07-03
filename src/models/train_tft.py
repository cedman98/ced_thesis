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
from src.features.schema import TARGET_CF, tft_known_reals
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


def train_and_evaluate_tft(matrix_path: str = 'data/processed/ml_training_matrix.parquet') -> None:
    df = pd.read_parquet(matrix_path)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df['time_idx'] = np.arange(len(df))
    df['group'] = "0"

    splitter = PurgedExpandingWindowSplitter(
        n_splits=4, test_duration=pd.Timedelta(days=365), purge_gap=pd.Timedelta(hours=72)
    )

    rows = []
    for tech in ("wind", "solar"):
        target_col = TARGET_CF[tech]
        known_reals = tft_known_reals(tech)
        logger.info(f"=== TFT {tech.upper()} (target {target_col}) ===")

        for fold, (train_idx, test_idx) in enumerate(splitter.split(df), start=1):
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
            pred = best.predict(test_dl, return_y=True)
            y_pred = np.clip(pred.output.flatten().cpu().numpy(), 0.0, 1.5)
            y_true = pred.y[0].flatten().cpu().numpy()

            rows += M.cv_rows("tft", tech, fold, y_true, y_pred)
            logger.info(f"{tech} TFT fold {fold} | CF nMAE {M.nmae(y_true, y_pred):.3f} corr {M.pearson_r(y_true, y_pred):.3f}")

        # Production model on the full dataset + XAI interpretation.
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

    M.append_cv_metrics(rows)
    logger.info("Saved TFT macro CV metrics.")


if __name__ == "__main__":
    train_and_evaluate_tft()
