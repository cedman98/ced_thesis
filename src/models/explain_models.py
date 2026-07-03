"""SHAP explainability for the production LightGBM capacity-factor models.

Quantifies the impact of the physical priors versus meteorological factors on
the median (q50) point forecast. Uses the exact training feature contract
(schema.FEATURE_COLS) — the models were trained on those scale-free columns
and nothing else, so passing any other column set to TreeExplainer is invalid.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.features.schema import FEATURE_COLS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_shap_explanations(
    matrix_path: str = 'data/processed/ml_training_matrix.parquet',
    sample_size: int = 2000,
) -> None:
    """SHAP summary plots for the wind and solar LightGBM median models."""
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading feature matrix from {matrix_path}")
    df = pd.read_parquet(matrix_path)
    X = df[FEATURE_COLS]
    X_sample = X.sample(n=min(sample_size, len(X)), random_state=42)
    logger.info(f"SHAP sample: {X_sample.shape[0]} rows x {X_sample.shape[1]} features")

    for tech in ('wind', 'solar'):
        model_path = Path(f'models/lightgbm_{tech}.pkl')
        if not model_path.exists():
            logger.error(f"{model_path} not found — run the training pipeline first.")
            continue
        model = joblib.load(model_path)
        logger.info(f"Computing SHAP values for the {tech} model...")
        shap_values = shap.TreeExplainer(model).shap_values(X_sample)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f"SHAP Feature Importance — LightGBM {tech} (capacity factor)",
                  fontsize=15, pad=15)
        plt.tight_layout()
        out = results_dir / f'lightgbm_{tech}_shap_summary.png'
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved {out}")


if __name__ == '__main__':
    generate_shap_explanations()
