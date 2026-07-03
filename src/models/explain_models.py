"""SHAP Explainability Module for ML Models.

This script uses SHAP (SHapley Additive exPlanations) to explain the output of the
LightGBM models. It quantifies the impact of physical priors versus meteorological
factors on the predictions, visually proving the logic learned by the tree ensembles.
"""

import os
import logging
from pathlib import Path
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_shap_explanations() -> None:
    """Calculates SHAP values for wind and solar models and generates summary plots."""
    
    # 1. Ensure the output directory exists
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Load the production dataset
    matrix_path = 'data/processed/ml_training_matrix.parquet'
    logger.info(f"Loading Spatio-Temporal Fusion Matrix from {matrix_path}")
    df = pd.read_parquet(matrix_path)
    
    if 'timestamp_utc' in df.columns:
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
        df = df.set_index('timestamp_utc')
    else:
        df.index = pd.to_datetime(df.index, utc=True)
        
    # 3. Separate features from targets
    target_wind_col = 'wind_onshore_mw_50hz'
    target_solar_col = 'solar_pv_mw_50hz'
    
    X = df.drop(columns=[target_wind_col, target_solar_col])
    
    # 4. Take a representative sample for computational efficiency
    sample_size = 2000
    if len(X) > sample_size:
        logger.info(f"Sampling {sample_size} rows uniformly from the feature matrix for SHAP calculation.")
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        logger.info(f"Dataset has {len(X)} rows, which is less than {sample_size}. Using the full dataset.")
        X_sample = X

    # 5. Load Models
    wind_model_path = Path('models/lightgbm_wind.pkl')
    solar_model_path = Path('models/lightgbm_solar.pkl')
    
    if not wind_model_path.exists() or not solar_model_path.exists():
        logger.error(f"Models not found at {wind_model_path} or {solar_model_path}. Did you run the training pipeline first?")
        return
    
    logger.info("Loading trained LightGBM production models.")
    wind_model = joblib.load(wind_model_path)
    solar_model = joblib.load(solar_model_path)
    
    # 6. Wind Explanation Generation
    logger.info("Instantiating SHAP TreeExplainer for the Wind Model...")
    wind_explainer = shap.TreeExplainer(wind_model)
    logger.info("Calculating SHAP values for the Wind Model...")
    wind_shap_values = wind_explainer.shap_values(X_sample)
    
    logger.info("Generating SHAP summary plot for the Wind Model...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(wind_shap_values, X_sample, show=False)
    # The summary plot automatically draws on the current figure.
    plt.title("SHAP Feature Importance - Wind Model", fontsize=16, pad=15)
    plt.tight_layout()
    
    wind_plot_path = results_dir / 'wind_shap_summary.png'
    plt.savefig(wind_plot_path, dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()
    logger.info(f"Wind SHAP summary plot successfully saved to {wind_plot_path}")
    
    # 7. Solar Explanation Generation
    logger.info("Instantiating SHAP TreeExplainer for the Solar Model...")
    solar_explainer = shap.TreeExplainer(solar_model)
    logger.info("Calculating SHAP values for the Solar Model...")
    solar_shap_values = solar_explainer.shap_values(X_sample)
    
    logger.info("Generating SHAP summary plot for the Solar Model...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(solar_shap_values, X_sample, show=False)
    plt.title("SHAP Feature Importance - Solar Model", fontsize=16, pad=15)
    plt.tight_layout()
    
    solar_plot_path = results_dir / 'solar_shap_summary.png'
    plt.savefig(solar_plot_path, dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()
    logger.info(f"Solar SHAP summary plot successfully saved to {solar_plot_path}")


if __name__ == '__main__':
    generate_shap_explanations()
