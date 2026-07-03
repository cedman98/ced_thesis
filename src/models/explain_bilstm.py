"""DeepSHAP Explainability Module for the Bi-LSTM Model.

This script uses SHAP (SHapley Additive exPlanations) DeepExplainer (based on DeepLIFT)
to explain the predictions of the PyTorch Bi-LSTM wind model. It quantifies how the
neural network weights the physical priors compared to meteorological factors.
"""

import os
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt

# Import custom modules from src
from src.models.train_bilstm import BiLSTM, TimeSeriesDataset
from src.features.validation_splitter import PurgedExpandingWindowSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data_and_create_test_tensors(
    matrix_path: str = 'data/processed/ml_training_matrix.parquet',
    sequence_length: int = 24
) -> Tuple[torch.Tensor, pd.DataFrame]:
    """Loads the spatio-temporal fusion matrix and constructs the scaled test tensors.
    
    Args:
        matrix_path: Path to the parquet matrix file.
        sequence_length: Sequence window length for LSTM.
        
    Returns:
        A tuple containing:
            - test_tensors: A PyTorch tensor of shape (num_samples, sequence_length, num_features)
            - X_test_raw: Pandas DataFrame containing the raw test feature slice (to preserve column names).
    """
    logger.info(f"Loading data from {matrix_path}")
    df = pd.read_parquet(matrix_path)
    
    if 'timestamp_utc' in df.columns:
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
        df = df.set_index('timestamp_utc')
    else:
        df.index = pd.to_datetime(df.index, utc=True)
        
    target_wind_col = 'wind_onshore_mw_50hz'
    target_solar_col = 'solar_pv_mw_50hz'
    
    y_wind = df[[target_wind_col]]
    X = df.drop(columns=[target_wind_col, target_solar_col])
    
    # Instantiate the purged expanding window splitter
    splitter = PurgedExpandingWindowSplitter(
        n_splits=4,
        test_duration=pd.Timedelta(days=365),
        purge_gap=pd.Timedelta(hours=72)
    )
    
    # Retrieve the final cross-validation fold
    splits = list(splitter.split(df))
    train_idx, test_idx = splits[-1]
    
    X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
    y_w_train_raw, y_w_test_raw = y_wind.iloc[train_idx], y_wind.iloc[test_idx]
    
    # Fit scaling matrices strictly on the training fold to prevent leakage
    scaler_X = StandardScaler()
    scaler_y_wind = StandardScaler()
    
    X_train_sc = scaler_X.fit_transform(X_train_raw)
    X_test_sc = scaler_X.transform(X_test_raw)
    
    y_w_train_sc = scaler_y_wind.fit_transform(y_w_train_raw).flatten()
    y_w_test_sc = scaler_y_wind.transform(y_w_test_raw).flatten()
    
    # Create the sequence dataset for the test fold
    test_ds = TimeSeriesDataset(X_test_sc, y_w_test_sc, sequence_length=sequence_length)
    
    # Extract all sequence tensors in a single batch
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
    test_tensors, _ = next(iter(test_loader))
    
    logger.info(f"Generated scaled test tensor of shape {test_tensors.shape}")
    return test_tensors, X_test_raw


def explain_bilstm_wind() -> None:
    """Explains PyTorch Bi-LSTM wind model predictions using SHAP DeepExplainer."""
    # Ensure output directories exist
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1 & 2: Load the scaled testing dataset tensors and metadata
    sequence_length = 24
    test_tensors, X_test_raw = load_data_and_create_test_tensors(sequence_length=sequence_length)
    
    # Step 3: Load the trained PyTorch model and weights
    input_dim = X_test_raw.shape[1]
    logger.info(f"Instantiating Bi-LSTM model with input dimension: {input_dim}")
    
    model = BiLSTM(input_size=input_dim, hidden_size=64, num_layers=2)
    model_path = 'models/bilstm_wind.pth'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model weights not found at {model_path}. Please train the model first.")
        
    device = torch.device('cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info("Loaded trained weights and set model to eval mode.")
    
    # Sample background and test distributions
    # DeepExplainer is computationally intensive, so we sample a smaller test set
    np.random.seed(42)
    torch.manual_seed(42)
    
    num_samples = test_tensors.shape[0]
    bg_size = 500
    test_size = 100
    
    bg_indices = np.random.choice(num_samples, size=bg_size, replace=False)
    test_indices = np.random.choice(num_samples, size=test_size, replace=False)
    
    background_tensors = test_tensors[bg_indices].to(device)
    sampled_test_tensors = test_tensors[test_indices].to(device)
    
    logger.info(f"Sampled {bg_size} background tensors and {test_size} test tensors.")
    
    # Initialize DeepExplainer
    logger.info("Initializing shap.DeepExplainer...")
    explainer = shap.DeepExplainer(model, background_tensors)
    
    # Compute SHAP values
    logger.info("Calculating SHAP values (with check_additivity=False to support PyTorch LSTM layer)...")
    shap_values = explainer.shap_values(sampled_test_tensors, check_additivity=False)
    
    # Handle list wrapping or output dimension in SHAP outputs
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    # The output structure is typically (samples, sequence_length, features, outputs) or (samples, sequence_length, features)
    if len(shap_values.shape) == 4 and shap_values.shape[-1] == 1:
        shap_values = np.squeeze(shap_values, axis=-1)
        
    logger.info(f"Raw SHAP values array shape: {shap_values.shape}")
    
    # Average SHAP values across the sequence dimension (dim/axis=1) to get overall feature importances
    shap_values_2d = np.mean(shap_values, axis=1)
    logger.info(f"Aggregated (averaged) 2D SHAP values shape: {shap_values_2d.shape}")
    
    # Prepare corresponding feature values for plotting
    # We average the features of the test sequences across the sequence dimension (axis=1) to match the SHAP values
    test_features_2d = sampled_test_tensors.cpu().numpy().mean(axis=1)
    
    # Preserve feature/column names from X to satisfy Explainable AI (XAI) Compliance
    feature_names = list(X_test_raw.columns)
    test_features_df = pd.DataFrame(test_features_2d, columns=feature_names)
    
    # Step 4: Generate a shap.summary_plot and save it
    logger.info("Generating SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    
    # We pass show=False to allow customizing the plot before saving
    shap.summary_plot(
        shap_values_2d,
        test_features_df,
        feature_names=feature_names,
        show=False
    )
    
    plt.title("Bi-LSTM Model - Average SHAP Feature Importance (Wind Onshore)", fontsize=16, pad=20)
    plt.tight_layout()
    
    output_plot_path = results_dir / 'bilstm_wind_shap.png'
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Successfully generated and saved SHAP summary plot to {output_plot_path}")


if __name__ == '__main__':
    explain_bilstm_wind()
