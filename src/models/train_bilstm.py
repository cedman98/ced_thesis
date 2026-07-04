"""Bi-LSTM benchmark (Capacity-Factor target, purged CV).

Sequence model over the scale-free CF features. The target is a capacity factor,
so the StandardScaler offset is now physically meaningful at *any* scale (macro
CF mean ~= municipal CF mean) instead of the multi-GW offset that previously
made local predictions anti-correlate. Feature + target scalers are persisted so
the municipal validator can run inference.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from src.features.validation_splitter import PurgedExpandingWindowSplitter
from src.features.schema import FEATURE_COLS, TARGET_CF, TARGET_MW, CAP_COLS, CF_CLIP
from src.evaluation import metrics as M

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEQ_LEN = 24
HIDDEN = 64
LAYERS = 2


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length=SEQ_LEN):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx):
        return self.X[idx:idx + self.sequence_length], self.y[idx + self.sequence_length]


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN, num_layers=LAYERS, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


def train_model(model, train_loader, val_loader, epochs=20, patience=3, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    best_loss, best_state, patience_counter = float('inf'), None, 0
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx).squeeze(-1), by)
            loss.backward()
            optimizer.step()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                vl += criterion(model(bx).squeeze(-1), by).item() * bx.size(0)
        vl /= len(val_loader.dataset)
        if vl < best_loss:
            best_loss, best_state, patience_counter = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict_cf(model, loader, scaler_y, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for bx, _ in loader:
            preds.extend(model(bx.to(device)).squeeze(-1).cpu().numpy())
    return scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()


def train_and_evaluate_bilstm(matrix_path: str = 'data/processed/ml_training_matrix.parquet') -> None:
    df = pd.read_parquet(matrix_path)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    X = df[FEATURE_COLS]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    splitter = PurgedExpandingWindowSplitter(
        n_splits=4, test_duration=pd.Timedelta(days=365), purge_gap=pd.Timedelta(hours=72)
    )

    rows = []
    for tech in ("wind", "solar"):
        y = df[TARGET_CF[tech]].values.reshape(-1, 1)
        for fold, (tr, te) in enumerate(splitter.split(df), start=1):
            sx, sy = StandardScaler(), StandardScaler()
            Xtr = sx.fit_transform(X.iloc[tr]); Xte = sx.transform(X.iloc[te])
            ytr = sy.fit_transform(y[tr]).flatten(); yte = sy.transform(y[te]).flatten()

            # Early-stopping validation = last 10% of the TRAIN window. The test
            # fold must never drive epoch selection (that leaked test information
            # into the reported CV metrics).
            cut = int(len(Xtr) * 0.9)
            tl = DataLoader(TimeSeriesDataset(Xtr[:cut], ytr[:cut]), batch_size=256, shuffle=True)
            vl = DataLoader(TimeSeriesDataset(Xtr[cut:], ytr[cut:]), batch_size=256, shuffle=False)
            te_dl = DataLoader(TimeSeriesDataset(Xte, yte), batch_size=256, shuffle=False)
            model = train_model(BiLSTM(len(FEATURE_COLS)), tl, vl)

            pred_cf = np.clip(_predict_cf(model, te_dl, sy, device), 0.0, CF_CLIP)
            # Targets/caps aligned: dataset emits target at idx+SEQ_LEN.
            y_cf = df[TARGET_CF[tech]].iloc[te].values[SEQ_LEN:]
            cap = df[CAP_COLS[tech]].iloc[te].values[SEQ_LEN:]
            y_mw = df[TARGET_MW[tech]].iloc[te].values[SEQ_LEN:]
            rows += M.cv_rows("bilstm", tech, fold, y_cf, pred_cf, cap, y_mw)
            logger.info(f"{tech} BiLSTM fold {fold} | CF nMAE {M.nmae(y_cf, pred_cf):.3f} corr {M.pearson_r(y_cf, pred_cf):.3f}")

    M.append_cv_metrics(rows)

    # Production models + scalers (scalers are required for municipal inference).
    os.makedirs('models', exist_ok=True)
    scalers = {}
    sx = StandardScaler(); Xf = sx.fit_transform(X)
    scalers['X'] = sx
    for tech in ("wind", "solar"):
        sy = StandardScaler()
        yf = sy.fit_transform(df[TARGET_CF[tech]].values.reshape(-1, 1)).flatten()
        loader = DataLoader(TimeSeriesDataset(Xf, yf), batch_size=256, shuffle=True)
        model = train_model(BiLSTM(len(FEATURE_COLS)), loader, loader, epochs=12, patience=12)
        torch.save(model.state_dict(), f'models/bilstm_{tech}.pth')
        scalers[tech] = sy
    scalers['meta'] = {'seq_len': SEQ_LEN, 'input_size': len(FEATURE_COLS), 'hidden': HIDDEN, 'layers': LAYERS, 'features': FEATURE_COLS}
    joblib.dump(scalers, 'models/bilstm_scalers.pkl')
    logger.info("Saved BiLSTM models + scalers.")


if __name__ == "__main__":
    train_and_evaluate_bilstm()
