"""
Energy Forecast System - Main Orchestration Entrypoint

This script coordinates the data pipelines and modeling layers for 
the Brandenburg rolling 24-hour ahead energy forecast.
"""

import logging
from src.ingestion.mastr_ingestion import main as run_mastr_ingestion
from src.features.geospatial_clustering import run_clustering_pipeline
from src.ingestion.weather_ingestion import run_weather_ingestion

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
    
    # Step 1: Ingest MaStR Data for Asset Topology
    logger.info("Executing MaStR Ingestion Pipeline...")
    run_mastr_ingestion()
    
    # Step 2: Process Geospatial Data and Calculate Centroids
    logger.info("Executing Geospatial Clustering Pipeline...")
    run_clustering_pipeline()
    
    # Step 3: Download Historical Weather for Clusters
    logger.info("Executing Historical Weather Ingestion...")
    run_weather_ingestion()
    
    logger.info("Pipeline execution finished successfully.")

if __name__ == "__main__":
    main()
