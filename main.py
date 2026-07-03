"""Energy Forecast System - Main Orchestration Entrypoint

This script coordinates the data pipelines and modeling layers for 
the Brandenburg rolling 24-hour ahead energy forecast.
"""

import logging
import os
import pandas as pd
from src.data.boundary_fetcher import fetch_brandenburg_boundary, load_config
from src.data.mastr_bulk_parser import run_parsing_pipeline
from src.features.geospatial_clustering import run_geospatial_clustering
from src.data.weather_ingestion import run_weather_ingestion


def setup_logging() -> None:
    """Configures project-level logging standard."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def run_forecast_mode() -> None:
    """Operational NWP branch: refresh forecast weather + rebuild the forecast
    feature matrix for inference. Assumes the reanalysis pipeline has already
    produced clusters, solar geometry and 50Hertz targets (they are mode-invariant).
    Models are trained on reanalysis and applied unchanged to these features.
    """
    logger = logging.getLogger(__name__)
    if not os.path.exists("data/processed/wind_clusters.csv"):
        raise FileNotFoundError(
            "forecast_mode requires the reanalysis pipeline outputs (clusters/targets). "
            "Run `uv run main.py` (reanalysis) once first."
        )
    logger.info("[forecast_mode] Ingesting Open-Meteo NWP forecast weather...")
    run_weather_ingestion("conf/config.yaml", mode="forecast")
    logger.info("[forecast_mode] Building forecast feature matrix...")
    from src.features.feature_pipeline import build_feature_matrix
    build_feature_matrix(mode="forecast")
    logger.info("[forecast_mode] Done. Quantify the NWP gap with "
                "`uv run -m src.evaluation.nwp_gap`.")


def main(mode: str = "reanalysis") -> None:
    """Execute the pipeline steps."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info(f"Initializing Energy Forecast System (Brandenburg) [mode={mode}]")

    if mode == "forecast":
        run_forecast_mode()
        return

    # Load configuration
    config_path = "conf/config.yaml"
    logger.info(f"Loading configuration from {config_path}...")
    config = load_config(config_path)
    
    # Step 1: Ensure Brandenburg administrative boundary is fetched
    logger.info("Step 1: Checking / fetching Brandenburg administrative boundary...")
    fetch_brandenburg_boundary(config)
    
    # Step 2: Run streaming spatial parser if outputs do not already exist
    wind_raw_path = "data/raw/mastr_wind_brandenburg_raw.csv"
    solar_raw_path = "data/raw/mastr_solar_brandenburg_raw.csv"
    
    if os.path.exists(wind_raw_path) and os.path.exists(solar_raw_path):
        logger.info("Step 2: Raw parsed MaStR datasets already exist. Skipping XML streaming parser stage.")
    else:
        logger.info("Step 2: Executing streaming spatial parser for MaStR bulk XML data...")
        run_parsing_pipeline(config_path)
    
    # Step 3: Run geospatial clustering to optimize weather API centroids
    logger.info("Step 3: Executing geospatial clustering to optimize weather API centroids...")
    run_geospatial_clustering(config_path)
    
    # Load resulting files to report detailed cluster statistics
    wind_clusters = pd.read_csv("data/processed/wind_clusters.csv")
    solar_clusters = pd.read_csv("data/processed/solar_clusters.csv")
    
    num_wind_clusters = len(wind_clusters)
    num_solar_utility = len(solar_clusters[solar_clusters["track_type"] == "utility"])
    num_solar_distributed = len(solar_clusters[solar_clusters["track_type"] == "distributed"])
    num_solar_total = len(solar_clusters)
    
    logger.info(
        f"Geospatial Clustering stage finished. Created:\n"
        f"  - {num_wind_clusters} Wind clusters\n"
        f"  - {num_solar_total} Solar clusters total ({num_solar_utility} utility, {num_solar_distributed} distributed)"
    )
    
    # Step 3b: Pre-calculate deterministic Solar Geometry & Clear-Sky Irradiance
    logger.info("Step 3b: Pre-calculating deterministic Solar Geometry & Clear-Sky Irradiance...")
    from src.features.solar_geometry import run_solar_geometry_pipeline
    run_solar_geometry_pipeline(config_path)
    
    # Step 3c: Harmonize 50Hertz macro-target generation actuals
    logger.info("Step 3c: Harmonizing 50Hertz macro-target generation actuals...")
    from src.data.target_harmonization import run_target_harmonization
    run_target_harmonization(config_path)
    
    # Step 4: Run weather data ingestion
    logger.info("Step 4: Executing historical weather data ingestion...")
    run_weather_ingestion(config_path)
    
    # Step 5: Assemble the capacity-factor ML training matrix (physical priors +
    # spatial weather + cyclical time, all scale-free).
    logger.info("Step 5: Building capacity-factor ML training matrix...")
    from src.features.feature_pipeline import build_feature_matrix
    build_feature_matrix()

    # Step 6: Naive baselines (persistence + climatology) — the bar ML must clear.
    logger.info("Step 6: Computing macro naive baselines...")
    from src.evaluation.baselines_macro import run as run_macro_baselines
    run_macro_baselines()

    # Step 7: Model training (all persist held-out macro CV metrics to results/).
    logger.info("Step 7: Training models on capacity-factor target...")
    from src.models.train_lightgbm import train_and_evaluate_models
    train_and_evaluate_models()
    from src.models.train_comparisons import train_and_evaluate_comparisons
    train_and_evaluate_comparisons()
    from src.models.train_bilstm import train_and_evaluate_bilstm
    train_and_evaluate_bilstm()
    from src.models.train_tft import train_and_evaluate_tft
    train_and_evaluate_tft()

    # Step 8: Cross-scale municipal validation (rolling, calibrated downscaling).
    logger.info("Step 8: Running cross-scale municipal validation...")
    from src.validation.municipal_validator import build_validation_pipeline
    build_validation_pipeline()
    from src.validation.tft_municipal_validator import build_tft_validation_pipeline
    build_tft_validation_pipeline()

    logger.info("Pipeline execution finished successfully.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Brandenburg energy forecast pipeline.")
    p.add_argument(
        "--mode", choices=["reanalysis", "forecast"], default="reanalysis",
        help="reanalysis: full train pipeline on ERA5 actuals. "
             "forecast: refresh NWP weather + build forecast feature matrix.",
    )
    main(p.parse_args().mode)


