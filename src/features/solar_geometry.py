"""Solar Geometry and Clear-Sky Irradiance Calculation Engine.

This module pre-calculates apparent solar position (zenith, azimuth) and 
theoretical clear-sky Global Horizontal Irradiance (GHI) for renewable 
energy resource centroids (solar clusters) in Brandenburg, Germany.
The calculations are deterministic and cover a specified multi-year timeframe 
at 15-minute intervals.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pvlib
from tqdm import tqdm

from src.data.boundary_fetcher import load_config

logger = logging.getLogger(__name__)


def process_single_cluster(
    cluster_id: str,
    lat: float,
    lon: float,
    start_time: str,
    end_time: str,
    tz: str,
    freq: str,
    output_dir: Path,
) -> Tuple[str, bool, str]:
    """Calculates solar geometry and clear-sky GHI for a single cluster and saves to Parquet.

    It generates a datetime index, constructs a Location object, calls get_solarposition
    to fetch solar elevation, zenith, and azimuth, uses the Ineichen clear-sky model,
    and saves the output to a Parquet file.

    Args:
        cluster_id: Unique identifier for the solar cluster.
        lat: Latitude of the cluster centroid.
        lon: Longitude of the cluster centroid.
        start_time: Start timestamp of the index (local time).
        end_time: End timestamp of the index (local time).
        tz: Timezone string (e.g. 'Europe/Berlin').
        freq: Frequency offset string (e.g. '15min').
        output_dir: Path to directory to save the resulting Parquet file.

    Returns:
        A tuple of (cluster_id, success_boolean, status_message) indicating the status of the process.

    Raises:
        This function catches all exceptions internally to prevent pool worker failure and returns
        the error message in the status tuple.
    """
    try:
        # Generate timezone-aware datetime index
        datetime_idx = pd.date_range(
            start=start_time, end=end_time, freq=freq, tz=tz
        )

        # Create Location object (altitude is automatically resolved via coordinate lookup)
        loc = pvlib.location.Location(latitude=lat, longitude=lon, tz=tz)

        # Compute solar position (apparent zenith, elevation, azimuth)
        solar_pos = pvlib.solarposition.get_solarposition(
            time=datetime_idx,
            latitude=loc.latitude,
            longitude=loc.longitude,
            altitude=loc.altitude,
        )

        # Compute clearsky GHI using Ineichen model (optimized by passing precomputed solar position)
        clearsky = loc.get_clearsky(
            times=datetime_idx,
            model="ineichen",
            solar_position=solar_pos,
        )

        # Prepare final DataFrame with clean identifiers
        result_df = pd.DataFrame(
            {
                "solar_zenith": solar_pos["apparent_zenith"],
                "solar_azimuth": solar_pos["azimuth"],
                "ghi_clearsky": clearsky["ghi"],
            },
            index=datetime_idx,
        )
        result_df.index.name = "timestamp"

        # Save to Parquet
        output_file = output_dir / f"{cluster_id}.parquet"
        result_df.to_parquet(output_file)

        return cluster_id, True, f"Successfully processed and saved to {output_file}"

    except Exception as e:
        return cluster_id, False, f"Failed: {str(e)}"


def run_solar_geometry_pipeline(
    config_path: str = "conf/config.yaml",
    solar_clusters_path: str = "data/processed/solar_clusters.csv",
    output_dir: str = "data/processed/solar_geometry",
    start_time: str = "2022-01-01 00:00:00",
    end_time: str = "2026-06-01 00:00:00",
    tz: str = "Europe/Berlin",
    freq: str = "15min",
    max_workers: Optional[int] = None,
) -> None:
    """Executes the solar geometry and clear-sky GHI calculations for all centroids in parallel.

    Args:
        config_path: Path to the YAML configuration file.
        solar_clusters_path: Path to the solar clusters CSV file.
        output_dir: Directory path where Parquet files will be stored.
        start_time: Start of the target time range.
        end_time: End of the target time range.
        tz: Timezone of the datetime index.
        freq: Frequency offset string of the index.
        max_workers: Number of processes to run in parallel. Defaults to CPU count - 1.

    Raises:
        FileNotFoundError: If the solar clusters CSV file is missing.
        ValueError: If required columns are missing in the clusters CSV.
    """
    logger.info("Starting Solar Geometry and Clear-Sky calculations...")

    # Load config if needed
    try:
        load_config(config_path)
    except Exception as e:
        logger.warning(f"Could not load config file: {e}. Proceeding with default arguments.")

    # Ensure output directory exists
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Read clusters
    if not os.path.exists(solar_clusters_path):
        raise FileNotFoundError(f"Solar clusters CSV not found at {solar_clusters_path}")

    clusters_df = pd.read_csv(solar_clusters_path)
    
    if "cluster_id" not in clusters_df.columns or \
       "centroid_lat" not in clusters_df.columns or \
       "centroid_lon" not in clusters_df.columns:
        raise ValueError(
            "solar_clusters.csv must contain columns 'cluster_id', 'centroid_lat', and 'centroid_lon'"
        )

    # Clean missing values
    clusters_df = clusters_df.dropna(subset=["cluster_id", "centroid_lat", "centroid_lon"]).copy()
    unique_clusters = clusters_df.drop_duplicates(subset=["cluster_id"])
    num_clusters = len(unique_clusters)
    
    logger.info(f"Loaded {num_clusters} solar clusters to process.")

    # Set up parallel workers
    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        max_workers = max(1, cpu_count - 1)

    logger.info(
        f"Using {max_workers} processes to calculate solar geometry from "
        f"{start_time} to {end_time} ({freq} interval)."
    )

    success_count = 0
    failure_count = 0
    start_wall = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for _, row in unique_clusters.iterrows():
            c_id = str(row["cluster_id"])
            lat = float(row["centroid_lat"])
            lon = float(row["centroid_lon"])

            future = executor.submit(
                process_single_cluster,
                cluster_id=c_id,
                lat=lat,
                lon=lon,
                start_time=start_time,
                end_time=end_time,
                tz=tz,
                freq=freq,
                output_dir=out_path,
            )
            futures[future] = c_id

        # Wrap in tqdm to visualize progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating Solar Geometry"):
            c_id = futures[future]
            try:
                c_id_res, success, msg = future.result()
                if success:
                    success_count += 1
                else:
                    failure_count += 1
                    logger.error(f"Cluster {c_id} calculation failed: {msg}")
            except Exception as exc:
                failure_count += 1
                logger.error(f"Cluster {c_id} generated an exception: {exc}")

    elapsed = time.time() - start_wall
    logger.info(
        f"Solar Geometry Calculation Complete.\n"
        f"  - Total Elapsed: {elapsed:.2f} seconds\n"
        f"  - Successfully processed: {success_count} / {num_clusters}\n"
        f"  - Failures: {failure_count} / {num_clusters}"
    )

    if failure_count > 0:
        logger.warning(f"{failure_count} clusters failed to compute. Check the error log above.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    run_solar_geometry_pipeline()
