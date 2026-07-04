# Checkpoint 07: ML Model Training Pipelines & XAI Integration

**Date**: 2026-06-29  
**Project**: Brandenburg Energy Forecast System (Bachelor's Thesis)  
**Cooperation**: RITS Project

---

## 1. Objectives Completed

In this checkpoint, we transitioned from data preparation to modeling. We integrated a diverse suite of machine learning architectures—ranging from gradient-boosted trees to recurrent and attention-based neural networks. Crucially, in strict adherence to the thesis's Explainable AI (XAI) requirements, post-hoc interpretability frameworks were deeply integrated into the evaluation loops.

### Part A: Machine Learning Engines
We implemented four distinct modeling architectures to benchmark tabular ML against sequence-to-sequence deep learning for spatial downscaling:
* **LightGBM & XGBoost**: Configured as strong baseline tabular models for direct multi-step forecasting (`src/models/train_lightgbm.py`).
* **Explainable Boosting Machines (EBM)**: Deployed via the `interpret` library as a glass-box GA²M (Generalized Additive Model) architecture to separate main effects and pairwise interactions (`src/models/train_ebm.py`).
* **BiLSTM**: Added a Bidirectional Long Short-Term Memory architecture to explicitly model temporal autocorrelations (`src/models/train_bilstm.py`).
* **Temporal Fusion Transformer (TFT)**: Implemented as the advanced sequential baseline (`src/models/train_tft.py`), utilizing self-attention mechanisms to explicitly separate static metadata, historical variables, and known future covariates (NWP).

### Part B: XAI & Transparency Compliance
To defend the models' reasoning:
* **SHAP Integration**: Integrated TreeExplainer and DeepExplainer across LightGBM, XGBoost, and BiLSTM pipelines (`src/models/explain_models.py`, `src/models/explain_bilstm.py`) to verify that physical priors behave monotonically.
* **TFT Attention Weights**: Extracted the internal attention heads for TFT (`src/models/explain_tft.py`) to map encoder lookback reliance.

### Part C: Evaluation Framework & Municipal Dashboard
* **Metrics Engine**: Created robust standardized error metric modules (`src/evaluation/metrics.py`) tracking MAE, RMSE, Pearson correlation, and Bias.
* **Expanding Window Validation**: Enforced the purged rolling cross-validation schema during all training sweeps to eliminate look-ahead leakage.
* **Municipal Validation Dashboard**: Built an interactive Streamlit dashboard (`dashboard.py`, `src/dashboard/forecast_service.py`) to visually benchmark the downscaled municipal predictions against the scraped actuals for Königs Wusterhausen and Nauen.

---

## 2. Next Steps

With all models fully trained, XAI structures preserved, and evaluation loops functional, the project now advances to the final phase: performing a comprehensive analytical review of model performance, spatial downscaling gaps, and feature attributions (which is documented in Checkpoint 08).
