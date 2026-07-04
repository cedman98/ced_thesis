# Checkpoint 12: EDA and Modeling Notebook Reorganization

**Date**: 2026-07-03  
**Project**: Brandenburg Energy Forecast System (Bachelor's Thesis)  
**Cooperation**: RITS Project

---

## 1. Overview

As the thesis progressed towards its final stages, the exploratory data analysis (EDA) and model prototyping workflows outgrew their initial scratchpads. To ensure reproducibility and logical flow for the final academic submission, we completely reorganized, expanded, and sequentially prefixed the Jupyter Notebooks in the `notebooks/` directory.

## 2. Notebook Sequence & Changes

The previous monolithic or scattered notebooks were split into a logical 11-step pipeline covering the entire thesis workflow:

- **`01_brandenburg_asset_mapping.ipynb`**: Initial geospatial mapping of renewable assets (Wind & Solar) across Brandenburg using MaStR coordinates.
- **`02_clustered_assets_mapping.ipynb`**: Implementation and visualization of geospatial clustering for assets to reduce dimensionality before weather ingestion.
- **`03_ml_matrix_eda.ipynb`**: Exploratory Data Analysis of the final ML-ready feature matrix (including capacity factor distributions and cyclical time encodings).
- **`04_model_performance_comparison.ipynb`**: Head-to-head evaluation of the machine learning architectures (LightGBM, XGBoost, EBM, BiLSTM, TFT) using Purged Cross-Validation.
- **`05_municipal_validation_dashboard.ipynb`**: Scratchpad prototyping for the Streamlit municipal validation metrics.
- **`06_macro_performance.ipynb`**: Deep dive into the models' performance at the aggregate 50Hertz TSO macro-level (showing optimal model calibration before downscaling).
- **`07_spatial_downscaling.ipynb`**: Core thesis analysis focusing on translating macro-learned capacity factor maps to the municipal scale.
- **`08_xai_and_features.ipynb`**: Explainable AI visualization layer (SHAP, interpret, and TFT attention matrices) to justify the physical priors.
- **`09_physics_vs_ml_residuals.ipynb`**: Analysis of the residuals between the theoretical physics simulation and the ML corrections.
- **`10_municipal_scope_comparison.ipynb`**: Extended comparison of downscaling performance across multiple scopes (e.g., Königs Wusterhausen vs. Nauen).
- **`11_brandenburg_state_forecast.ipynb`**: Final simulated rollout of the 24-hour ahead forecasts for the entire state of Brandenburg.

## 3. Improvements

- **Kernel and Environment**: Updated kernel specifications across all notebooks to ensure they run cleanly in the unified `uv` environment.
- **Narrative Flow**: The sequential prefixing (`01_` to `11_`) now perfectly maps to the thesis chapters, providing examiners with a reproducible, step-by-step codebase to follow alongside the written document.
