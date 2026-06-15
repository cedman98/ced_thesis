"""Geospatial Clustering Pipeline for Renewable Asset Fleets.

This module implements spatial clustering algorithms to group wind and solar
assets in Brandenburg. Grouping assets optimizes weather API calls while
preserving micro-meteorological variation.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from src.data.boundary_fetcher import load_config

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0088  # Mean earth radius in kilometers


def cluster_wind_fleet(
    df: pd.DataFrame, cluster_radius_km: float
) -> pd.DataFrame:
    """Groups wind turbines into clusters using DBSCAN with Haversine metric.

    For each cluster, computes the capacity-weighted centroid using gross_power
    as the weight. Sums the total capacity in MW and counts constituent assets.

    Args:
        df: Input DataFrame containing wind turbines with 'latitude',
          'longitude', and 'gross_power' columns.
        cluster_radius_km: Maximum distance in kilometers between two samples
          for one to be considered as in the neighborhood of the other.

    Returns:
        A DataFrame summarizing the clusters with columns:
        ['cluster_id', 'centroid_lat', 'centroid_lon', 'total_capacity_mw',
        'asset_count']
    """
    if df.empty:
        logger.warning("Empty wind turbine DataFrame provided for clustering.")
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "centroid_lat",
                "centroid_lon",
                "total_capacity_mw",
                "asset_count",
            ]
        )

    # Ensure clean numeric columns and no NaN coordinates
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df["gross_power"] = pd.to_numeric(df["gross_power"], errors="coerce").fillna(0.0)

    # Convert coordinates to radians (required for sklearn haversine metric)
    # The order MUST be [latitude, longitude] for sklearn haversine
    coords = np.radians(df[["latitude", "longitude"]].values)
    eps = cluster_radius_km / EARTH_RADIUS_KM

    # DBSCAN with min_samples=1 ensures every turbine is part of some cluster
    db = DBSCAN(eps=eps, min_samples=1, metric="haversine")
    df["cluster_label"] = db.fit_predict(coords)

    # Calculate capacity-weighted centroids and aggregate stats
    cluster_records = []
    grouped = df.groupby("cluster_label")
    for label, group in grouped:
        total_power_kw = group["gross_power"].sum()
        total_capacity_mw = total_power_kw / 1000.0
        asset_count = len(group)

        if total_power_kw > 0:
            centroid_lat = np.sum(group["latitude"] * group["gross_power"]) / total_power_kw
            centroid_lon = np.sum(group["longitude"] * group["gross_power"]) / total_power_kw
        else:
            # Fallback to simple mean if total capacity is zero
            centroid_lat = group["latitude"].mean()
            centroid_lon = group["longitude"].mean()

        cluster_records.append(
            {
                "cluster_id": f"wind_{label}",
                "centroid_lat": float(centroid_lat),
                "centroid_lon": float(centroid_lon),
                "total_capacity_mw": float(total_capacity_mw),
                "asset_count": int(asset_count),
            }
        )

    result_df = pd.DataFrame(cluster_records)
    logger.info(
        f"Wind Fleet: Clustered {len(df)} turbines into {len(result_df)} clusters "
        f"using DBSCAN (radius={cluster_radius_km} km)."
    )
    return result_df


def cluster_solar_utility(
    df: pd.DataFrame, cluster_radius_km: float = 2.0
) -> pd.DataFrame:
    """Clusters utility-scale (ground-mounted) solar installations using DBSCAN.

    Args:
        df: Input solar DataFrame containing utility installations (is_ground_mounted=True).
        cluster_radius_km: Maximum distance in kilometers for DBSCAN eps.

    Returns:
        DataFrame containing utility solar clusters.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "centroid_lat",
                "centroid_lon",
                "total_capacity_mw",
                "asset_count",
                "track_type",
            ]
        )

    # Convert coordinates to radians
    coords = np.radians(df[["latitude", "longitude"]].values)
    eps = cluster_radius_km / EARTH_RADIUS_KM

    db = DBSCAN(eps=eps, min_samples=1, metric="haversine")
    df = df.copy()
    df["cluster_label"] = db.fit_predict(coords)

    cluster_records = []
    grouped = df.groupby("cluster_label")
    for label, group in grouped:
        total_power_kw = group["gross_power"].sum()
        total_capacity_mw = total_power_kw / 1000.0
        asset_count = len(group)

        if total_power_kw > 0:
            centroid_lat = np.sum(group["latitude"] * group["gross_power"]) / total_power_kw
            centroid_lon = np.sum(group["longitude"] * group["gross_power"]) / total_power_kw
        else:
            centroid_lat = group["latitude"].mean()
            centroid_lon = group["longitude"].mean()

        cluster_records.append(
            {
                "cluster_id": f"solar_utility_{label}",
                "centroid_lat": float(centroid_lat),
                "centroid_lon": float(centroid_lon),
                "total_capacity_mw": float(total_capacity_mw),
                "asset_count": int(asset_count),
                "track_type": "utility",
            }
        )

    return pd.DataFrame(cluster_records)


def cluster_solar_distributed(
    df: pd.DataFrame, grid_resolution: float = 0.1
) -> pd.DataFrame:
    """Groups distributed solar installations using a static geographic grid.

    Rounds coordinates to the nearest grid resolution, groups by grid cell, and
    aggregates capacity. The grid cell center serves as the centroid.

    Args:
        df: Input solar DataFrame containing distributed installations.
        grid_resolution: Resolution of the grid in degrees (e.g. 0.1).

    Returns:
        DataFrame containing distributed solar grid cells.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "centroid_lat",
                "centroid_lon",
                "total_capacity_mw",
                "asset_count",
                "track_type",
            ]
        )

    df = df.copy()
    # Round coordinates to the nearest increment of grid_resolution, with high precision
    df["grid_lat"] = np.round(np.round(df["latitude"] / grid_resolution) * grid_resolution, 6)
    df["grid_lon"] = np.round(np.round(df["longitude"] / grid_resolution) * grid_resolution, 6)

    # Group by rounded coordinates
    grouped = df.groupby(["grid_lat", "grid_lon"])
    cluster_records = []

    for i, ((grid_lat, grid_lon), group) in enumerate(grouped):
        total_power_kw = group["gross_power"].sum()
        total_capacity_mw = total_power_kw / 1000.0
        asset_count = len(group)

        cluster_records.append(
            {
                "cluster_id": f"solar_dist_{i}",
                "centroid_lat": float(grid_lat),
                "centroid_lon": float(grid_lon),
                "total_capacity_mw": float(total_capacity_mw),
                "asset_count": int(asset_count),
                "track_type": "distributed",
            }
        )

    return pd.DataFrame(cluster_records)


def run_geospatial_clustering(
    config_path: str = "conf/config.yaml",
    wind_radius_override: Optional[float] = None,
) -> None:
    """Orchestrates the geospatial clustering pipeline.

    Loads raw wind and solar CSVs, performs the respective clustering strategies,
    consolidates outputs, and saves summary files to the data/processed directory.

    Args:
        config_path: Path to the configuration YAML file.
        wind_radius_override: Optional radius override in km for the wind fleet clustering.
    """
    config = load_config(config_path)

    # 1. Load configuration variables
    open_meteo_conf = config.get("open_meteo", {})
    grid_resolution = open_meteo_conf.get("grid_resolution", 0.1)

    # Use override if provided, otherwise config value, otherwise default 3.0
    if wind_radius_override is not None:
        wind_radius = wind_radius_override
        logger.info(f"Using manual wind cluster radius override: {wind_radius} km")
    else:
        wind_radius = open_meteo_conf.get("cluster_radius_km", 3.0)
        logger.info(f"Using wind cluster radius from configuration: {wind_radius} km")

    # 2. Input file paths
    wind_raw_path = Path("data/raw/mastr_wind_brandenburg_raw.csv")
    solar_raw_path = Path("data/raw/mastr_solar_brandenburg_raw.csv")

    if not wind_raw_path.exists():
        raise FileNotFoundError(f"Missing raw wind data: {wind_raw_path}")
    if not solar_raw_path.exists():
        raise FileNotFoundError(f"Missing raw solar data: {solar_raw_path}")

    # 3. Cluster Wind Fleet
    logger.info(f"Loading raw wind data from: {wind_raw_path}")
    wind_df = pd.read_csv(wind_raw_path)
    wind_clusters_df = cluster_wind_fleet(wind_df, wind_radius)

    # 4. Cluster Solar Fleet (Dual-Track)
    logger.info(f"Loading raw solar data from: {solar_raw_path}")
    solar_df = pd.read_csv(solar_raw_path)

    # Ensure valid inputs
    solar_df = solar_df.dropna(subset=["latitude", "longitude"]).copy()
    solar_df["gross_power"] = pd.to_numeric(solar_df["gross_power"], errors="coerce").fillna(0.0)

    # Map is_ground_mounted to boolean correctly
    if solar_df["is_ground_mounted"].dtype == object:
        is_ground_mounted_bool = solar_df["is_ground_mounted"].astype(str).str.lower() == "true"
    else:
        is_ground_mounted_bool = solar_df["is_ground_mounted"].astype(bool)

    # Split into Track A (Utility) and Track B (Distributed)
    utility_df = solar_df[is_ground_mounted_bool].copy()
    distributed_df = solar_df[~is_ground_mounted_bool].copy()

    logger.info(
        f"Solar Fleet: Split {len(solar_df)} assets into {len(utility_df)} utility-scale "
        f"and {len(distributed_df)} distributed rooftop assets."
    )

    # Track A: DBSCAN clustering with strict 2.0 km radius
    utility_radius = 2.0
    logger.info(f"Clustering utility solar assets using DBSCAN (radius={utility_radius} km)...")
    utility_clusters_df = cluster_solar_utility(utility_df, utility_radius)

    # Track B: Grid clustering using 0.1 degree grid
    logger.info(f"Clustering distributed solar assets using grid-aggregation (resolution={grid_resolution} deg)...")
    distributed_clusters_df = cluster_solar_distributed(distributed_df, grid_resolution)

    # Combine Solar Tracks
    solar_clusters_df = pd.concat([utility_clusters_df, distributed_clusters_df], ignore_index=True)

    # 5. Export results
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    wind_out_path = processed_dir / "wind_clusters.csv"
    solar_out_path = processed_dir / "solar_clusters.csv"

    wind_clusters_df.to_csv(wind_out_path, index=False, encoding="utf-8")
    solar_clusters_df.to_csv(solar_out_path, index=False, encoding="utf-8")

    logger.info(
        f"Consolidated and exported wind clusters to {wind_out_path} "
        f"({len(wind_clusters_df)} clusters, representing {wind_clusters_df['asset_count'].sum()} assets)."
    )
    logger.info(
        f"Consolidated and exported solar clusters to {solar_out_path} "
        f"({len(solar_clusters_df)} clusters: {len(utility_clusters_df)} utility, "
        f"{len(distributed_clusters_df)} distributed; representing {solar_clusters_df['asset_count'].sum()} assets)."
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    run_geospatial_clustering()
