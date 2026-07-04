# Checkpoint 11: Municipal Validation Dashboard Implementation

**Date**: 2026-07-02  
**Project**: Brandenburg Energy Forecast System (Bachelor's Thesis)  
**Cooperation**: RITS Project

---

## 1. Overview

In this phase, we implemented a dedicated municipal validation dashboard using Streamlit. This dashboard bridges the gap between the macro-scale training data (50Hertz level) and the municipal-level telemetry (Königs Wusterhausen and Nauen) we collect via our scraper. It provides an interactive interface for evaluating the spatial downscaling performance of our models.

## 2. Key Additions and Changes

### Streamlit Dashboard Application (`dashboard.py`)
- Created the main entry point for the interactive web dashboard.
- Implemented visualization panels for comparing model predictions (LightGBM, XGBoost, EBM, TFT, BiLSTM) against actual telemetry.
- Added visual analysis of downscaled target gaps (over-forecasting and under-forecasting metrics).

### Forecast Service Layer (`src/dashboard/forecast_service.py`)
- Developed a dedicated service to handle prediction loading, metric calculations, and formatting data for the Streamlit UI.
- Minor tweaks were subsequently introduced to accommodate NWP (Numerical Weather Prediction) gap analysis metrics, aligning the service with expanding window cross-validation updates.

### Administrative Mapping Data (`src/data/districts.py`)
- Introduced mapping structures and geographic boundaries for Brandenburg's districts (Landkreise/Gemeinden) to allow spatial localization within the dashboard.
- Refactored administrative mappings during subsequent reliability fixes to handle missing spatial nodes and improve integration with the Open-Meteo ingestion pipelines.

## 3. Impact on Thesis

The dashboard allows examiners and project stakeholders to visually inspect the central problem of the thesis: the spatial downscaling mismatch. By providing an interactive UI, we successfully demonstrate that while macro-forecasting performs well, the translation to the municipal scale retains systematic bias (as analyzed in previous checkpoints).
