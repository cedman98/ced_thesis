"""Temporal Fusion Transformer (TFT) Interpretability Pipeline.

This module extracts the native variable importances and attention patterns from
the trained PyTorch Forecasting TFT checkpoint, producing visualizations for the thesis.
"""

import os
import logging
import io
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer

from src.features.validation_splitter import PurgedExpandingWindowSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def rebuild_validation_dataset(
    target_type: str = 'wind',
    matrix_path: str = 'data/processed/ml_training_matrix.parquet',
    max_encoder_length: int = 72,
    max_prediction_length: int = 24
) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """Rebuilds the training and validation TimeSeriesDataSet objects from the matrix.
    
    Args:
        target_type: Target variable type ('wind' or 'solar').
        matrix_path: Path to the parquet matrix file.
        max_encoder_length: Lookback window step size.
        max_prediction_length: Forecast window step size.
        
    Returns:
        A tuple of (training_data, validation_data) TimeSeriesDataSets.
    """
    logger.info(f"Re-loading Spatio-Temporal Fusion Matrix from {matrix_path}")
    df = pd.read_parquet(matrix_path)
    
    if 'timestamp_utc' in df.columns:
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
        df = df.set_index('timestamp_utc')
    else:
        df.index = pd.to_datetime(df.index, utc=True)

    df = df.sort_index()
    df['time_idx'] = np.arange(len(df))
    df['group'] = "0"
    
    if target_type == 'wind':
        target_col = 'wind_onshore_mw_50hz'
        known_reals = [
            "brandenburg_wind_prior_mw",
            "temperature_2m",
            "surface_pressure",
            "relative_humidity_2m",
            "hour_sin",
            "hour_cos",
            "month_sin",
            "month_cos"
        ]
    elif target_type == 'solar':
        target_col = 'solar_pv_mw_50hz'
        known_reals = [
            "brandenburg_solar_prior_mw",
            "temperature_2m",
            "surface_pressure",
            "relative_humidity_2m",
            "hour_sin",
            "hour_cos",
            "month_sin",
            "month_cos"
        ]
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    # Recreate the splitter and grab the final split index
    splitter = PurgedExpandingWindowSplitter(
        n_splits=4,
        test_duration=pd.Timedelta(days=365),
        purge_gap=pd.Timedelta(hours=72)
    )
    
    splits = list(splitter.split(df))
    train_idx, test_idx = splits[-1]
    
    df_train = df.iloc[train_idx].copy()
    history_start_idx = test_idx[0] - max_encoder_length
    df_test = df.iloc[history_start_idx : test_idx[-1] + 1].copy()
    
    logger.info(f"Instantiating TimeSeriesDataSet matching the validation fold configuration for {target_type.upper()}...")
    training_data = TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",
        target=target_col,
        group_ids=["group"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=[target_col],
        target_normalizer=GroupNormalizer(groups=["group"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    validation_data = TimeSeriesDataSet.from_dataset(training_data, df_test, predict=False, stop_randomization=True)
    return training_data, validation_data


def combine_and_save_importance_plots(
    fig_encoder: plt.Figure,
    fig_decoder: plt.Figure,
    output_path: Path
) -> None:
    """Combines two matplotlib figures vertically using PIL and saves them as a single image.
    
    Args:
        fig_encoder: Matplotlib Figure for encoder variables.
        fig_decoder: Matplotlib Figure for decoder variables.
        output_path: File path to save the combined image to.
    """
    logger.info("Combining encoder and decoder variable importance plots...")
    
    # Save encoder figure to bytes
    buf_enc = io.BytesIO()
    fig_encoder.savefig(buf_enc, format='png', dpi=300, bbox_inches='tight')
    buf_enc.seek(0)
    img_enc = Image.open(buf_enc)
    
    # Save decoder figure to bytes
    buf_dec = io.BytesIO()
    fig_decoder.savefig(buf_dec, format='png', dpi=300, bbox_inches='tight')
    buf_dec.seek(0)
    img_dec = Image.open(buf_dec)
    
    # Determine the canvas dimension
    w1, h1 = img_enc.size
    w2, h2 = img_dec.size
    max_width = max(w1, w2)
    total_height = h1 + h2
    
    # Paste images vertically
    combined = Image.new('RGB', (max_width, total_height), color='white')
    combined.paste(img_enc, ((max_width - w1) // 2, 0))
    combined.paste(img_dec, ((max_width - w2) // 2, h1))
    
    # Save output
    combined.save(output_path, format='PNG')
    logger.info(f"Combined importance plots successfully saved to {output_path}")


def explain_tft_model() -> None:
    """Extracts internal weights from the TFT checkpoint and generates explainability plots."""
    # Ensure folders exist
    doc_dir = Path('documentation')
    results_dir = Path('results')
    doc_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for target_type in ['wind', 'solar']:
        checkpoint_path = Path(f'models/tft/tft_{target_type}_production.ckpt')
        if not checkpoint_path.exists():
            logger.warning(f"TFT checkpoint not found at {checkpoint_path}. Skipping explainability for {target_type}.")
            continue
            
        logger.info(f"Generating explanations for TFT model ({target_type.upper()})...")
        # Rebuild the dataset setup
        _, validation_data = rebuild_validation_dataset(target_type=target_type)
        val_dataloader = validation_data.to_dataloader(train=False, batch_size=128, num_workers=0)
        
        # Load model
        logger.info(f"Loading Temporal Fusion Transformer from {checkpoint_path}")
        best_tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
        best_tft.eval()
        
        # Step 3: Generate predictions and calculate interpretation dictionary
        logger.info("Generating predictions on the validation set...")
        raw_predictions = best_tft.predict(val_dataloader, return_x=True, mode="raw", return_index=True)
        
        logger.info("Extracting raw interpretation weights from output...")
        # Handle the Prediction object safely by accessing its .output attribute if present
        if hasattr(raw_predictions, 'output'):
            out_data = raw_predictions.output
        else:
            out_data = raw_predictions
            
        interpretation = best_tft.interpret_output(out_data, reduction="sum")
        
        # Step 4: Generate plots
        logger.info("Generating matplotlib figures using native plot_interpretation method...")
        figs = best_tft.plot_interpretation(interpretation)
        
        # Save variable importance plots
        if 'encoder_variables' in figs and 'decoder_variables' in figs:
            importance_path = doc_dir / f'tft_{target_type}_importance.png'
            combine_and_save_importance_plots(
                fig_encoder=figs['encoder_variables'],
                fig_decoder=figs['decoder_variables'],
                output_path=importance_path
            )
        else:
            logger.warning("Could not find encoder or decoder variables in the plotted interpretation.")
            # Attempt fallback to save individual plots
            if 'encoder_variables' in figs:
                figs['encoder_variables'].savefig(doc_dir / f'tft_{target_type}_encoder_importance.png', dpi=300, bbox_inches='tight')
            if 'decoder_variables' in figs:
                figs['decoder_variables'].savefig(doc_dir / f'tft_{target_type}_decoder_importance.png', dpi=300, bbox_inches='tight')

        # Save attention plot
        if 'attention' in figs:
            attention_path = results_dir / f'tft_{target_type}_attention.png'
            figs['attention'].savefig(attention_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved attention plot successfully to {attention_path}")
        else:
            logger.warning("Could not find attention weight plots in interpretation.")
            
        # Clean up figure resources to avoid memory leaks
        for fig in figs.values():
            plt.close(fig)


if __name__ == "__main__":
    explain_tft_model()
