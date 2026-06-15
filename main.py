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

def main() -> None:
    """Execute the pipeline steps."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Initializing Energy Forecast System (Brandenburg)")
    
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
    
    logger.info("Pipeline execution finished successfully.")


if __name__ == "__main__":
    main()


