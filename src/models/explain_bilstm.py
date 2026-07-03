"""DeepSHAP explainability for the Bi-LSTM capacity-factor models.

Uses shap.DeepExplainer (DeepLIFT) on the trained PyTorch Bi-LSTM. Sequences
are built from the last purged CV test fold with the *persisted* production
scalers and the training feature contract (schema.FEATURE_COLS); per-feature
importance is the SHAP value averaged over the 24-step input sequence.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.models.train_bilstm import BiLSTM, TimeSeriesDataset
from src.features.validation_splitter import PurgedExpandingWindowSplitter
from src.features.schema import FEATURE_COLS, TARGET_CF

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def explain_bilstm(
    tech: str = 'wind',
    matrix_path: str = 'data/processed/ml_training_matrix.parquet',
    bg_size: int = 500,
    n_explain: int = 100,
) -> None:
    """SHAP summary plot for one technology's Bi-LSTM, saved to results/."""
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)

    scalers = joblib.load('models/bilstm_scalers.pkl')
    meta = scalers['meta']

    df = pd.read_parquet(matrix_path)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    # Last purged CV fold's test window = the most recent held-out year.
    splitter = PurgedExpandingWindowSplitter(
        n_splits=4, test_duration=pd.Timedelta(days=365), purge_gap=pd.Timedelta(hours=72))
    _, test_idx = list(splitter.split(df))[-1]

    Xte = scalers['X'].transform(df[FEATURE_COLS].iloc[test_idx])
    yte = scalers[tech].transform(df[TARGET_CF[tech]].iloc[test_idx].values.reshape(-1, 1)).flatten()
    ds = TimeSeriesDataset(Xte, yte, sequence_length=meta['seq_len'])
    tensors, _ = next(iter(DataLoader(ds, batch_size=len(ds), shuffle=False)))
    logger.info(f"[{tech}] built {tensors.shape[0]} test sequences of shape {tuple(tensors.shape[1:])}")

    model = BiLSTM(meta['input_size'], meta['hidden'], meta['layers'])
    model.load_state_dict(torch.load(f'models/bilstm_{tech}.pth', map_location='cpu'))
    model.eval()

    rng = np.random.default_rng(42)
    bg = tensors[rng.choice(len(tensors), size=min(bg_size, len(tensors)), replace=False)]
    te = tensors[rng.choice(len(tensors), size=min(n_explain, len(tensors)), replace=False)]

    logger.info(f"[{tech}] computing DeepSHAP values ({len(bg)} background, {len(te)} explained)...")
    shap_values = shap.DeepExplainer(model, bg).shap_values(te, check_additivity=False)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]

    # Average over the sequence dimension -> one importance row per sample.
    sv2d = shap_values.mean(axis=1)
    feat2d = pd.DataFrame(te.numpy().mean(axis=1), columns=FEATURE_COLS)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv2d, feat2d, show=False)
    plt.title(f"Bi-LSTM DeepSHAP importance — {tech} (capacity factor)", fontsize=15, pad=15)
    plt.tight_layout()
    out = results_dir / f'bilstm_{tech}_shap.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {out}")


if __name__ == '__main__':
    for tech in ('wind', 'solar'):
        explain_bilstm(tech)
