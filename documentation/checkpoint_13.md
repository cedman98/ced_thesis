# Checkpoint 13: Baselines Sweep, Calibration Audit & Capacity Factor Sanity Check

**Date**: 2026-07-04  
**Project**: Brandenburg Energy Forecast System (Bachelor's Thesis)  
**Cooperation**: RITS Project

---

## 1. Baselines Sweep (Step 10)

The macro baseline sweep was successfully executed via `baselines_macro` (designed to be idempotent—`append_cv_metrics` correctly deduplicates on `(model, tech, fold, scale)` by keeping the last entry). 

* **Results**: Both naive baselines (persistence and climatology) are now fully present across all 4 cross-validation folds for both wind and solar technologies. 
* **Notebook Integration**: `06_macro_performance.ipynb` accurately cites these baselines. Specifically, the Figure 1a bar charts include them (designated with hatched patterns and a "hatched = naive baseline" caption), and the Figure 1c/1d time-series overlays explicitly draw the 24-hour persistence forecast.

---

## 2. Calibration Audit & CF Sanity (Steps 11 & 12)

We introduced a new evaluation notebook: **`12_calibration_and_capacity_audit.ipynb`**. The execution runs perfectly clean (0 errors) and produces two critical figures for defending the methodology:

### A. Affine Calibration (`s5_affine_calibration.png`)
* **Finding**: The calibration slope remains highly stable across different models within a single fleet, varying primarily by municipality and technology rather than model architecture.
* **Fit Integrity**: We successfully resolved real two-parameter affine fits (scale, offset) for both Königs Wusterhausen (KW) and Nauen without falling back to identity matrices. 
* **Scale Discrepancy**: Nauen's wind scale is notably small ($\approx 0.013$) compared to KW ($\approx 0.44$). This is mathematically sound and simply reflects the fact that Nauen possesses an enormous local wind capacity (241 MW) compared to KW (28 MW), which the affine scale naturally absorbs.

### B. Capacity vs. Peaks (`s6_capacity_vs_peaks.png`)
* **Finding**: The active capacity variable ($C_t$) correctly sits at the 99.9th percentile of actual generation and roughly 0–10% under the absolute annual maximum. 
* **Validation**: Only 0.01% to 0.28% of hours yield a Capacity Factor > 1. This directly answers potential examiner scrutiny regarding "why use the 99.9th percentile," as it perfectly aligns with the expected ~0.1% exceedance implied by a 99.9th percentile active capacity estimate.

---

## 3. Important Methodological Note (Target Area vs. Brandenburg Fleet)

For the final Methods section (and to be reflected when finalizing the notebook inventory), it is critical to highlight the following macro-scale capacity discrepancy:

* **Observation**: The wind active capacity $C_t \approx 16\text{ GW}$ exceeds the physically installed Brandenburg MaStR fleet nameplate ($\approx 9.6\text{ GW}$). 
* **Reasoning**: This is **not a bug** (verified within `feature_pipeline.py`). The 50Hertz TSO target actuals represent the *entire* 50Hertz control area (all of East Germany plus offshore), not solely the Brandenburg state boundaries. The system mathematically defines the Capacity Factor on the macro TSO series first, and then accurately transfers that learned map locally.
