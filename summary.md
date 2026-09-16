# Comprehensive Thesis Summary: Predicting Regional Renewable Power Generation in Brandenburg

This document provides a fully fleshed-out summary of the entire thesis, chapter by chapter, section by section. All figures are noted within their corresponding sections.

---

## Chapter 1: Introduction

**Note on Figures:** There are no figures explicitly referenced or included in Chapter 1.

### 1.1 Motivation: the Energiewende Reaches the Distribution Grid
This section details the structural shift in Germany's power system due to the "Energiewende" (energy transition). Generation is transitioning from a few large, centrally dispatched thermal plants to hundreds of thousands of small, weather-driven, and geographically dispersed renewable units (primarily wind and photovoltaics). This shift pushes the operational burden from the transmission system operator (TSO) down to distribution system operators (DSOs). As a result, congestion, voltage excursions, and reverse power flows are increasingly occurring at the distribution level. Anticipating this generation day-ahead is crucial for DSOs and municipalities to manage operations.

### 1.2 The Gap: an Observation Asymmetry
The author identifies a significant "observation asymmetry" in the available data. While the regional TSO publishes high-quality, aggregate generation data for the entire control zone, there is virtually no publicly available generation data at the municipal or district levels. A forecasting model trained on the macro data learns absolute magnitudes and cannot be natively shrunk to apply to a municipality. Conversely, training separate models for municipalities is impossible due to the lack of local training labels. There is a need for a scale-invariant formulation to transfer macro-scale relationships down to unmeasured local scales.

### 1.3 Problem Statement
The thesis addresses a core problem: Can aggregate renewable generation, published only at the control-zone level, be downscaled to create 24-hour-ahead forecasts at the municipal and district levels? The solution must be achieved using only publicly available data (national asset registry and open weather services) along with a minimal amount of local telemetry for calibration. 

### 1.4 Research Questions
The problem is broken down into three research questions:
*   **RQ1:** Does a capacity-factor target (representing fractional utilization) successfully transfer across spatial scales where a raw megawatt target cannot? 
*   **RQ2:** How much of the forecasting skill is contributed by a bottom-up physics prior (a deterministic function of weather and installed hardware) compared to the machine learning applied on top of it? 
*   **RQ3:** What forecast accuracy is achievable at the municipal level, and what limits it?

### 1.5 Contributions
The chapter outlines six main contributions of the thesis:
1.  **A scale-invariant capacity-factor formulation**
2.  **Physics-informed feature priors**
3.  **An affine downscaling contract**
4.  **A leak-free evaluation protocol**
5.  **Interpretability and honest uncertainty reporting**
6.  **State-wide operational scaling**

### 1.6 Thesis Outline
The final section provides a roadmap of the upcoming chapters (Chapters 2 to 8).

---

## Chapter 2: Background and Related Work

**Note on Figures:** There are no figures referenced in Chapter 2.

### 2.1 Renewable Generation Forecasting: A Taxonomy
*   **2.1.1 Model Class: Physical, Statistical, and Hybrid approaches:** Contrasts deterministic physical models with data-driven statistical models. Hybrid models combine these, using physics for structural bounds and ML for corrections.
*   **2.1.2 Forecast Horizon:** Distinguishes between nowcasting, intra-day, and day-ahead forecasting. The thesis operates on a day-ahead horizon, heavily relying on numerical weather prediction (NWP).
*   **2.1.3 Forecast Form: Point and Probabilistic:** Differentiates single-value point estimates from probabilistic intervals that characterize uncertainty.
*   **2.1.4 Positioning:** Positions the thesis directly within this taxonomy: a hybrid physics-ML model, issuing point and probabilistic forecasts, at a day-ahead horizon.

### 2.2 Wind Power Generation Fundamentals
*   **2.2.1 From Wind Speed to Power: The Physical Relationship:** Explains the cubic dependency of available power on wind speed.
*   **2.2.2 The Power Curve:** Describes the empirical manufacturer power curve mapping wind speed to output. 
*   **2.2.3 Hub-Height Extrapolation:** Details how wind speeds are extrapolated to turbine hub heights.
*   **2.2.4 Site-Specific and Fleet-Level Factors:** Explores limitations of the ideal physical model, noting omitted variables such as wake losses and curtailment.

### 2.3 Solar PV Generation Fundamentals
*   **2.3.1 The Irradiance-to-Power Chain:** Breaks down the calculation steps: horizontal irradiance (GHI) to plane-of-array (POA) irradiance to DC power, and finally to AC power.
*   **2.3.2 Standard Test Conditions and the Nameplate Rating:** Explains the STC laboratory benchmark used for rating solar modules.
*   **2.3.3 Clear-Sky Models:** Introduces deterministic models that isolate cloudless irradiance levels.
*   **2.3.4 The Performance Ratio:** Details how operational loss elements are aggregated into a fixed performance ratio (PR).

### 2.4 Spatial Downscaling and Upscaling in Forecasting
*   **2.4.1 Top-Down Scaling:** Discusses statistical downscaling frameworks adapted from climate science.
*   **2.4.2 Disaggregation and Inverse Methods:** Covers methods partitioning aggregate region data onto constituent nodes.
*   **2.4.3 Bottom-Up Modelling:** Describes reconstructing total generation by driving a known inventory of local assets with local weather variables.
*   **2.4.4 Upscaling and the sub-regional gap:** Details the inverse issue—estimating macro outputs from sampled regional observations.

### 2.5 Hybrid Physics-Informed Machine Learning and Residual Learning
*   **2.5.1 The Complementary Weaknesses:** Summarizes how the strengths of physical models offset the weaknesses of ML models and vice versa.
*   **2.5.2 A Taxonomy of Coupling:** Catalogs four hybrid layouts.
*   **2.5.3 Residual Learning:** Focuses on the adopted approach: using the physical baseline as the anchor and letting the learner predict the discrepancy (residual).

### 2.6 Forecasting Evaluation Methods and Time-Series Validation
*   **2.6.1 Why Standard Cross-Validation Fails on Time Series:** Explains why a purged, expanding-window scheme with a temporal embargo is essential.
*   **2.6.2 Quantile Regression and the Pinball Loss:** Explains the use of pinball loss functions to estimate non-parametric intervals.
*   **2.6.3 Scoring Probabilistic Forecasts:** Discusses tracking interval reliability (PICP) alongside sharpness (MPIW).
*   **2.6.4 Point Metrics and Reference Baselines:** Emphasizes beating naive baselines (persistence and climatology).
*   **2.6.5 Explainability for Time-Series Models:** Spotlights tools used to attribute skill post-hoc, including SHAP values, EBMs, and attention weights.

### 2.7 The Data Landscape and Public Sources
*   **2.7.1 Asset Registries:** Discusses the German Marktstammdatenregister (MaStR) used for bottom-up priors.
*   **2.7.2 Transmission:** Covers macro-scale output labels published by operators like 50Hertz.
*   **2.7.3 Meteorological Data:** Contrasts historical reanalysis weather data (e.g., ERA5) with live numerical weather prediction (NWP).
*   **2.7.4 Municipal Telemetry:** Stresses the scarcity of sub-regional actuals (ground truth), motivating the thesis.

### 2.8 Research Gap and Positioning
Summarizes the overarching gap: Forecasting is well-established for full control zones and single turbine assets, but undeveloped for municipal boundaries due to lack of ground truth. 

---

## Chapter 3: Data

### 3.1 The Data Landscape: an Asymmetry Problem
Introduces the core data challenge: the asymmetry in the availability of renewable generation data. Four primary datasets are introduced: MaStR (fleet definition), 50Hertz actuals (training target), Open-Meteo (meteorological drivers), and E.ON Energiemonitor (calibration ground truth). Includes a reference to **Table 3.1** summarizing the datasets.

### 3.2 Asset Registry and Spatial Scope: MaStR
Details the use of the Marktstammdatenregister (MaStR) to define the physical fleet. Includes a reference to **Table 3.3**, highlighting the active wind and solar units in Brandenburg, underscoring the structural contrast between large wind turbines and numerous small solar rooftops. 

### 3.3 Macro Generation Actuals: 50Hertz
The dataset includes onshore wind and photovoltaic generation at a 15-minute resolution, which is cleaned and standardized. The model deliberately never sees municipal targets during training, making downscaling a true transfer learning problem.

### 3.4 Meteorological Drivers: Open-Meteo, Reanalysis versus Forecast
Weather is the sole time-varying exogenous driver. Data is drawn from Open-Meteo via the Archive API (ERA5 reanalysis) for training and Forecast API (NWP) for operational simulation. Addresses the "train-deploy discrepancy" (NWP gap).

### 3.5 Municipal Telemetry: E.ON Energiemonitor, the Only Ground Truth
The sole source of local ground truth, offering telemetry for exactly two towns in Brandenburg (Königs Wusterhausen and Nauen). This data is strictly reserved for fitting and validating the downscaling calibration.

### 3.6 Capacity Normalization: from Megawatts to Capacity Factor
Explains the transformation of the learning target from raw megawatts to a scale-free capacity factor. 
*   **Figure 3.1:** Illustrates the continuous growth of installed nameplate capacity in the 50Hertz zone from 2000 to 2026. 
The capacity factor is normalized by a causally-computed "effective capacity" ($C_t$) to prevent data leakage.

### 3.7 Data Quality, Coverage, and Provenance Discipline
Summarizes how datasets are joined into a unified grid and recaps the project's strict data provenance. Highlights six key quality risks, including the mismatch between registry nameplate and metered footprint, and the scarcity of municipal ground truth.

---

## Chapter 4: Methodology

### 4.1 Target Formulation: the Capacity Factor
Defines the core target variable as a scale-free capacity factor (CF) instead of absolute megawatt generation, enabling zero-shot spatial transfer to municipal scales (addressing RQ1). The first leakage guard is introduced: normalization employs a causal trailing 99.9th percentile.

### 4.2 Physics Priors and Feature Engineering
*   **4.2.1 Wind Prior:** Wind speeds are extrapolated to hub heights and mapped to power using manufacturer power curves, yielding generation priors.
*   **4.2.2 Solar Prior:** Solar PV output is modelled using irradiance scaled to Standard Test Conditions and adjusted by a performance ratio.
*   **4.2.3 Normalisation to Capacity Factors (Second Leakage Guard):** Wind and solar megawatt priors are divided by their respective fleet nameplate capacities to yield dimensionless capacity factor priors. Absolute magnitudes are stripped.
*   **4.2.4 Temporal Features:** Includes spatially averaged weather factors and cyclic temporal embeddings (sine/cosine encodings).
*   **4.2.5 The Feature Contract:** A central `schema.py` enforces consistency between state-level training and local-level inference.

### 4.3 Geospatial Clustering
The asset fleet is clustered to preserve local micro-meteorology efficiently. Wind and utility-scale solar farms use DBSCAN clustering, while distributed rooftop solar uses a fixed spatial grid. 

### 4.4 Models
Tests five distinct model architectures on the capacity-factor target: LightGBM (production choice), XGBoost, Explainable Boosting Machine (EBM), Bidirectional LSTM, and Temporal Fusion Transformer (TFT). 

### 4.5 Evaluation Protocol
Standard random splitting fails due to autocorrelation.
*   **Figure 4.1:** Illustrates the Purged Expanding-Window Cross-Validation Scheme (the third leakage guard), showing expanding training windows and fixed-length testing windows separated by a 72-hour purge gap.

### 4.6 The Spatial Downscaling Contract
Defines how scale-free capacity factor predictions are translated into absolute megawatt generation (addressing RQ3) via an affine transformation: `Local = scale * (CF * Nameplate) + offset`. The fourth leakage guard ensures calibration occurs strictly on a chronologically-held-out split of telemetry.

### 4.7 Summary
Reaffirms the four-point leakage discipline and summarizes the methodology pipeline.

---

## Chapter 5: Implementation

### 5.1 Architectural Overview
The system is built as a batch data-engineering pipeline divided into seven logical layers, orchestrated in `main.py`.
*   **Figure 5.1:** Visualizes the end-to-end data flow from public data sources through the scale-free feature matrix and model training, to cross-scale validation and operational serving.

### 5.2 Module Map
Details the division of responsibilities across the pipeline's modules, referencing **Table 5.1**. A strict boundary is maintained between executable modules and exploratory notebooks. Includes a standalone scraper service for municipal telemetry.

### 5.3 Structural Contracts
Relies on three core contracts for leak-free evaluation: Canonical Schema (`schema.py`), Unified Feature Construction via callbacks, and Frozen Calibration Artifacts (constants and affine parameters saved and reused verbatim).

### 5.4 Dual-Mode Operation: Reanalysis and Forecast
The codebase operates in two modes: training models on historical data (reanalysis branch) and producing operational forecasts using live weather data (forecast branch). This quantifies the "NWP gap".

### 5.5 Reproducibility and Tooling
Maintained using the `uv` package manager. Configurable properties are centralized, and model training is fully deterministic. Core modules contain embedded self-checks.

### 5.6 Operational Artefacts
The final layer uses trained models for operational forecasting. The service pulls NWP forecasts, runs quantile models, and downscales to megawatt generation bands. Assets are assigned to all 18 Brandenburg districts. 

### 5.7 Summary
Summarizes how the pipeline structurally enforces the scale-free and leak-free guarantees proposed in the methodology.

---

## Chapter 6: Results and Evaluation

### 6.1 Macro model skill
*   **Figure 6.1:** Compares macro cross-validation metrics across five ML models and naive baselines. LightGBM demonstrates strong capabilities and is chosen for production.
*   **Figure 6.2:** Overlays actual 24-hour wind and solar generation profiles against LightGBM out-of-fold forecasts.
*   **Figure 6.3:** Hexbin density plots showing predicted versus actual capacity factor regression.

### 6.2 Do the models earn their place? Baseline comparison
The models heavily outperform "persistence" and "climatology" baselines. Machine learning drops nMAE for wind 3.3 times below persistence. 

### 6.3 Interpretability and the physics prior
Answers RQ2 by showing the engineered physics prior heavily drives forecast skill. 
*   **Figure 6.4:** SHAP beeswarm plot for the wind model showing prior value importance.
*   **Figure 6.5:** SHAP beeswarm plot for the solar model showing prior value importance.
*   **Figure 6.6:** EBM feature interaction heatmaps for wind and solar features.
*   **Figure 6.7:** TFT wind model attention weights showing near-uniform historical attention.

### 6.4 Physics prior versus learned residual
*   **Figure 6.8:** Hexbin plots of physical prior capacity factor against actual capacity factor. 
*   **Figure 6.9:** Heatmap of the correlation between ML residuals and meteorological features.
*   **Figure 6.10:** Hexbin density plots showing the dependency of ML residuals on variables like temperature and surface pressure. The ML residual successfully recovers unmodelled physical gaps (e.g., wake effects).

### 6.5 Cross-scale downscaling to municipalities
Tests applying models to unseen municipalities (RQ3). Downscaling works well for solar but errors double for wind.
*   **Figure 6.11:** Scale gap evaluation bar charts for wind and solar.
*   **Figure 6.12:** Affine slope stability plots, showing slopes vary wildly by fleet due to accounting mismatches.
*   **Figure 6.13:** One-week validation time series in Königs Wusterhausen.
*   **Figure 6.14:** Scatter plots mapping predicted vs actual power generation in Königs Wusterhausen.

### 6.6 Probabilistic performance and interval coverage
*   **Figure 6.15:** Reliability diagram (PICP) evaluating the LightGBM 80% prediction intervals. Solar macro approaches the nominal target, but wind under-covers. Municipal intervals collapse severely.
*   **Figure 6.16:** Density plots comparing prediction error (bias) distributions for municipal test cases.

### 6.7 Operational realism: the NWP gap
*   **Figure 6.17:** Bar charts quantifying the performance penalty when using live NWP forecast weather instead of historical ERA5 reanalysis weather.

### 6.8 State-wide scaling across the 18 districts
*   **Figure 6.18:** A choropleth map displaying the 24-hour forecasted aggregate renewable generation across the 18 municipal districts in Brandenburg.

### 6.9 Summary
Answers the core research questions: capacity-factor formulation is necessary for cross-scale transfer (RQ1), physics prior carries the predictive signal (RQ2), and municipal downscaling works well for solar but faces accounting hurdles for wind (RQ3).

---

## Chapter 7: Discussion

**Note on Figures:** There are no figures referenced in Chapter 7.

### 7.1 Why It Works, and What RQ1 Established
The forecasting system acts as a "weather-to-power translator". It is scale-free (predicting capacity factor), allowing state-level models to predict for municipal fleets. The capacity-factor formulation successfully transfers the temporal signal across spatial scales.

### 7.2 The Physics Prior as the Load-Bearing Feature
Addressing RQ2, the physics prior is the dominant driver of the model. The gradient-boosted ensemble fits the residual—the gap between the idealized physical response and reality. This explains why complex sequence models did not vastly outperform simpler ones.

### 7.3 Municipal Accuracy and Its Ceiling
Answering RQ3, downscaling affects wind and solar asymmetrically. Solar forecasting is effectively solved locally, but wind errors double. This is driven by solar fleet homogeneity and the roughly linear physics of solar vs. the cubic nonlinearity of wind to spatial averages.

### 7.4 Resolving the Wind Scale Divergence
*   **7.4.1 The Claim:** The affine calibration scalar for wind diverged massively between the two pilot municipalities.
*   **7.4.2 Why the Divergence Is Accounting and Not Skill:** This is an accounting discrepancy. The wind models still achieved very high temporal correlations; only the absolute magnitude was off.
*   **7.4.3 What the Scale Is Made Of:** The 8 km geographic radius sweeps in huge neighboring wind farms that the local distribution meter cannot actually see (as they are connected to the transmission grid). 
*   **7.4.4 Consequences:** The scale parameter cannot be blindly transferred. The state-wide forecast is a relative shape product, not a calibrated absolute megawatt prediction.

### 7.5 Threats to Validity
Consolidates caveats: Reanalysis optimism (NWP errors are the operationally honest figures), telemetry sparsity (only two towns), heuristic turbine matching, effective-capacity proxy undershoot, lack of spatial hold-out for calibration, interval under-coverage, and weak circularity in training data.

### 7.6 Relation to the Wider Literature
The thesis performs *downscaling* (distributing a regional total to unmetered sub-regions) rather than conventional upscaling. By using physics priors as ML features, it secures the data-efficiency of physical models while letting ML correct registry errors.

### 7.7 Generalisability
The methodology is generic and transferable, but its binding constraint is the availability of a granular asset registry (like MaStR) to build the physics priors.

---

## Chapter 8: Conclusion

**Note on Figures:** There are no figures presented or referenced in this chapter.

### 8.1 Summary of Work
Outlines the gap addressed: the lack of public sub-aggregate signals for renewable generation. The system successfully built a scale-invariant, leak-free, physics-informed hybrid model scaled down to municipalities.

### 8.2 Answers to the research questions
*   **8.2.1 RQ1: Transfer of a capacity-factor target across spatial scales:** Successfully transfers the shape of the generation signal, proving scale-freeness is a necessary condition for downscaling.
*   **8.2.2 RQ2: The contribution of the physics prior relative to pure machine learning:** The physics prior strongly dominates, while the ML layer serves as a necessary residual corrector.
*   **8.2.3 RQ3: Achievable municipal accuracy and its limits:** Solar downscaling is usable at the municipal level, whereas wind downscaling currently suffers from capacity-accounting mismatches, sparse ground truth, and optimistic reanalysis evaluations.

### 8.3 Future Work
Identifies deferred items: 1) Reconciliation of metered footprint with the registry, 2) Feature ablation study, 3) Expansion of municipal ground truth, 4) Conformal prediction for intervals, 5) Training on forecast weather, 6) Replacement of the effective-capacity proxy, and 7) Operational deployment.

### 8.4 Closing Statement
The proposed forecasting system is generic and relies only on a public asset registry, transmission actuals, and limited metered local data. It provides a scalable route to downscale aggregate control zone signals to the local level.
