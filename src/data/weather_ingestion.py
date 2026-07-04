"""Historical Weather Data Ingestion Module using Open-Meteo.

This module downloads hourly historical weather data for geospatial grid nodes
in Brandenburg. It maps cluster coordinates to 0.1-degree weather grid cells,
performs rate-limit-aware and resumable batch downloads, and writes results
directly as compressed Parquet files.
"""

import datetime
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests

from src.data.boundary_fetcher import load_config

logger = logging.getLogger(__name__)


def snap_coordinates(lat: float, lon: float) -> Tuple[float, float]:
    """Snaps a coordinate to the closest 0.1 decimal degree resolution grid.

    Args:
        lat: Latitude of the coordinate.
        lon: Longitude of the coordinate.

    Returns:
        A tuple of (snapped_lat, snapped_lon) rounded to 1 decimal place.
    """
    return round(lat, 1), round(lon, 1)


def load_state(state_path: Path) -> Dict[str, Any]:
    """Loads the ingestion state from a JSON file.

    If the file does not exist, initializes a default state dictionary.

    Args:
        state_path: Path to the JSON state file.

    Returns:
        A dictionary containing the current ingestion state.
    """
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
                if "completed_nodes" not in state:
                    state["completed_nodes"] = []
                if "daily_calls_count" not in state:
                    state["daily_calls_count"] = 0
                if "last_call_date" not in state:
                    state["last_call_date"] = ""
                logger.info(
                    f"Loaded existing ingestion state with {len(state['completed_nodes'])} completed grid nodes."
                )
                return state
        except json.JSONDecodeError as e:
            logger.warning(
                f"Could not parse state file {state_path} due to error: {e}. Reinitializing state."
            )
    return {"completed_nodes": [], "daily_calls_count": 0, "last_call_date": ""}


def save_state(state_path: Path, state: Dict[str, Any]) -> None:
    """Saves the ingestion state atomically to a JSON file.

    Args:
        state_path: Path to the JSON state file.
        state: The state dictionary to save.
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = state_path.with_suffix(".json.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)
        # Atomically replace state file
        temp_path.replace(state_path)
    except Exception as e:
        logger.error(f"Failed to save ingestion state to {state_path}: {e}")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def _rate_limit_sleep(reason: str, batch_label: str, retry_delay: float) -> float:
    """Sleep out a 429 according to its reason; returns the next backoff delay.

    Daily/hourly quota exhaustion sleeps until the quota window resets (with a
    small safety margin); anything else backs off exponentially.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    if "Daily API request limit exceeded" in reason:
        until = (now + datetime.timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
    elif "Hourly API request limit exceeded" in reason:
        until = (now + datetime.timedelta(hours=1)).replace(minute=1, second=0, microsecond=0)
    else:
        logger.warning(
            f"API returned 429 Rate Limit error on {batch_label}. "
            f"Sleeping for {retry_delay} seconds before retrying..."
        )
        time.sleep(retry_delay)
        return retry_delay * 2  # exponential backoff
    sleep_seconds = (until - now).total_seconds()
    logger.warning(
        f"{reason} on {batch_label}. Sleeping for {sleep_seconds:.1f} seconds "
        f"(until {until.isoformat()}) before retrying..."
    )
    time.sleep(sleep_seconds)
    return retry_delay


def _429_reason(response) -> str:
    try:
        return response.json().get("reason", "")
    except Exception:
        return ""


def print_resume_instructions(batch_idx: int) -> None:
    """Outputs clear instructions to the log/console on how to resume execution.

    Args:
        batch_idx: The current batch index where execution was paused.
    """
    logger.info("=" * 80)
    logger.info("WEATHER INGESTION PAUSED GRACEFULLY")
    logger.info(f"The execution stopped at batch index: {batch_idx}")
    logger.info("To resume the ingestion pipeline:")
    logger.info("1. Wait for rate limits to clear or wait for the next day if the daily limit was hit.")
    logger.info("2. Simply re-run the main pipeline or weather ingestion script.")
    logger.info("   It will automatically read '.ingestion_state.json' and resume from the pending grid nodes.")
    logger.info("=" * 80)


def run_weather_ingestion(
    config_path: str = "conf/config.yaml",
    start_date: str = "2022-01-01",
    end_date: str = "2026-06-01",
    mode: str = "reanalysis",
) -> None:
    """Orchestrates weather data ingestion for all unique grid locations.

    Reads geospatial cluster coordinates from wind and solar clusters, maps them
    to 0.1-degree grid cells, filters out already downloaded grid nodes, and
    queries Open-Meteo in batches of 5 locations. Writes timeseries weather data
    as compressed Parquet files and creates a mapping index for downstream tasks.

    Two modes select the data source and (isolated) output directory:
      - ``reanalysis``: Open-Meteo archive (ERA5 actuals) -> data/processed/weather/
      - ``forecast``:   Open-Meteo historical-forecast archive (NWP model runs) ->
                        data/processed/weather_forecast/
    Both are strictly UTC. The historical-forecast endpoint is the archived
    output of the forecast model over the same date range, which is exactly what
    lets us quantify the NWP gap against reanalysis on identical timestamps.

    Args:
        config_path: Path to the configuration YAML file.
        start_date: Start date of query (YYYY-MM-DD), defaults to '2022-01-01'.
        end_date: End date of query (YYYY-MM-DD), defaults to '2026-06-01'.
        mode: 'reanalysis' or 'forecast'.

    Raises:
        FileNotFoundError: If the input cluster files do not exist.
        RuntimeError: If critical ingestion errors occur.
    """
    if mode not in ("reanalysis", "forecast"):
        raise ValueError(f"mode must be 'reanalysis' or 'forecast', got {mode!r}")
    config = load_config(config_path)
    open_meteo_config = config.get("open_meteo", {})
    hourly_variables = open_meteo_config.get("variables", [])

    if not hourly_variables:
        raise ValueError("No target weather variables found in open_meteo.variables configuration.")

    # 1. Setup paths
    processed_dir = Path("data/processed")
    wind_clusters_path = processed_dir / "wind_clusters.csv"
    solar_clusters_path = processed_dir / "solar_clusters.csv"
    
    weather_out_dir = processed_dir / ("weather" if mode == "reanalysis" else "weather_forecast")
    weather_subdir = weather_out_dir.name  # used to stamp per-node paths in the mapping index
    grid_nodes_dir = weather_out_dir / "grid_nodes"
    
    # Ensure output directories exist
    grid_nodes_dir.mkdir(parents=True, exist_ok=True)

    state_path = weather_out_dir / ".ingestion_state.json"
    mapping_path = weather_out_dir / "mapping_index.json"

    # Lock mechanism to prevent concurrent runs
    import atexit
    lock_path = weather_out_dir / ".lock"
    if lock_path.exists():
        try:
            with open(lock_path, "r", encoding="utf-8") as lf:
                old_pid = int(lf.read().strip())
            os.kill(old_pid, 0)
            logger.info(f"Another ingestion process (PID {old_pid}) is already running. Exiting.")
            return
        except (ValueError, OSError):
            logger.info("Removing stale lock file.")
            try:
                lock_path.unlink()
            except Exception:
                pass

    with open(lock_path, "w", encoding="utf-8") as lf:
        lf.write(str(os.getpid()))

    def cleanup_lock():
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass

    atexit.register(cleanup_lock)

    # 2. Verify cluster files exist
    if not wind_clusters_path.exists():
        raise FileNotFoundError(f"Missing wind cluster file: {wind_clusters_path}")
    if not solar_clusters_path.exists():
        raise FileNotFoundError(f"Missing solar cluster file: {solar_clusters_path}")

    # 3. Load clusters and map to grid cells
    wind_df = pd.read_csv(wind_clusters_path)
    solar_df = pd.read_csv(solar_clusters_path)

    cluster_mapping = {}
    unique_grid_points = set()

    def map_clusters(df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            c_id = str(row["cluster_id"])
            c_lat = float(row["centroid_lat"])
            c_lon = float(row["centroid_lon"])
            
            grid_lat, grid_lon = snap_coordinates(c_lat, c_lon)
            unique_grid_points.add((grid_lat, grid_lon))
            
            cluster_mapping[c_id] = {
                "centroid_lat": c_lat,
                "centroid_lon": c_lon,
                "grid_lat": grid_lat,
                "grid_lon": grid_lon,
                "parquet_path": f"data/processed/{weather_subdir}/grid_nodes/node_{grid_lat:.1f}_{grid_lon:.1f}.parquet"
            }

    map_clusters(wind_df)
    map_clusters(solar_df)

    # Export mapping index
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(cluster_mapping, f, indent=4)
    logger.info(f"Saved structural mapping index for {len(cluster_mapping)} clusters to {mapping_path}")

    # Convert to sorted list of grid points to ensure deterministic processing
    unique_points_list = sorted(list(unique_grid_points))
    total_points = len(unique_points_list)
    logger.info(f"Identified {total_points} unique grid nodes across Brandenburg.")

    # 4. Load state and filter pending points
    state = load_state(state_path)
    
    # Check date to handle daily call counter reset
    current_date = datetime.date.today().isoformat()
    if state.get("last_call_date") != current_date:
        state["last_call_date"] = current_date
        state["daily_calls_count"] = 0
        save_state(state_path, state)

    completed_nodes = set(state.get("completed_nodes", []))
    pending_points = [
        pt for pt in unique_points_list
        if f"{pt[0]:.1f}_{pt[1]:.1f}" not in completed_nodes
    ]
    total_pending = len(pending_points)

    if total_pending == 0:
        logger.info("All unique weather grid cells already ingested. weather_ingestion is complete.")
        return

    logger.info(
        f"Ingestion resume state: {total_points - total_pending}/{total_points} completed. "
        f"{total_pending} remaining."
    )

    # 5. Group unique coordinates into batches of 5 locations
    batch_size = 5
    batches = [pending_points[i : i + batch_size] for i in range(0, len(pending_points), batch_size)]

    # Resolve URL & API Key for commercial or default free usage. Reanalysis hits
    # the ERA5 archive; forecast hits the historical-forecast archive (past NWP
    # model runs over the same date range).
    # ponytail: forecast mode uses the historical-forecast archive so the gap
    # study aligns on real timestamps; for a live operational run swap to
    # open_meteo.forecast_url with forecast_days=1.
    api_key = os.environ.get("OPEN_METEO_API_KEY") or open_meteo_config.get("api_key")
    if mode == "forecast":
        base_url = (
            "https://customer-historical-forecast-api.open-meteo.com/v1/forecast"
            if api_key else "https://historical-forecast-api.open-meteo.com/v1/forecast"
        )
    else:
        base_url = (
            "https://customer-api.open-meteo.com/v1/archive"
            if api_key else "https://archive-api.open-meteo.com/v1/archive"
        )
    logger.info(f"[{mode}] Using Open-Meteo endpoint: {base_url} (api_key={'yes' if api_key else 'no'})")

    # Rate limiting thresholds
    DAILY_CALL_LIMIT = 10000
    FRACTIONAL_THRESHOLD = 0.95
    DAILY_LIMIT_THRESHOLD = int(DAILY_CALL_LIMIT * FRACTIONAL_THRESHOLD)

    # 6. Fetch batch data
    for idx, batch in enumerate(batches):
        # Join coordinates into comma-separated strings
        batch_lats_str = ",".join(f"{lat:.1f}" for lat, lon in batch)
        batch_lons_str = ",".join(f"{lon:.1f}" for lat, lon in batch)

        # Check fractional daily threshold limit
        if not api_key and state["daily_calls_count"] >= DAILY_LIMIT_THRESHOLD:
            logger.warning(
                f"Daily API call count ({state['daily_calls_count']}) has hit the "
                f"daily free fractional threshold ({DAILY_LIMIT_THRESHOLD}). Gracefully pausing."
            )
            print_resume_instructions(idx)
            return

        params = {
            "latitude": batch_lats_str,
            "longitude": batch_lons_str,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(hourly_variables),
            "timezone": "UTC",
        }
        if api_key:
            params["apikey"] = api_key

        logger.info(
            f"Fetching batch {idx + 1}/{len(batches)} with {len(batch)} locations (lat: {batch_lats_str[:40]}...)"
        )

        success = False
        max_retries = 5
        retry_delay = 15.0
        response = None
        failures = 0  # network errors / unexplained 429s; quota sleeps don't count

        state["daily_calls_count"] += 1
        state["last_call_date"] = current_date
        save_state(state_path, state)

        # Quota 429s (hourly/daily/minutely limit) sleep until the window resets
        # and retry indefinitely — a 4-year x 430-node backfill *will* exhaust the
        # free hourly quota many times over; that must never abort the run.
        while failures < max_retries:
            try:
                response = requests.get(base_url, params=params, timeout=60)

                if response.status_code == 429:
                    reason = _429_reason(response)
                    retry_delay = _rate_limit_sleep(
                        reason, f"batch {idx + 1}/{len(batches)}", retry_delay)
                    if "request limit exceeded" not in reason:
                        failures += 1  # unexplained 429: bounded retries
                    continue

                response.raise_for_status()
                success = True
                break

            except requests.exceptions.RequestException as e:
                status_code = e.response.status_code if e.response is not None else "Unknown"
                logger.warning(
                    f"Network error on batch {idx + 1}/{len(batches)}, failure {failures + 1}/{max_retries} (HTTP status: {status_code}): {e}."
                )
                if status_code == 429:
                    reason = _429_reason(e.response) if e.response is not None else ""
                    retry_delay = _rate_limit_sleep(
                        reason, f"batch {idx + 1}/{len(batches)}", retry_delay)
                    if "request limit exceeded" not in reason:
                        failures += 1
                    continue
                failures += 1
                logger.warning("Sleeping for 5.0 seconds before retrying...")
                time.sleep(5.0)

        if not success:
            logger.error(f"Failed to fetch batch {idx + 1}/{len(batches)} after {max_retries} attempts.")
            print_resume_instructions(idx)
            return

        try:
            data = response.json()

            # Wrap dict in list if querying a single location
            if isinstance(data, dict):
                data = [data]

            # Process each response entry
            for (lat, lon), loc_data in zip(batch, data):
                hourly = loc_data.get("hourly", {})
                if not hourly or "time" not in hourly:
                    logger.warning(f"No hourly weather data in response for ({lat:.1f}, {lon:.1f}). Skipping.")
                    continue

                df = pd.DataFrame(hourly)
                df["time"] = pd.to_datetime(df["time"])

                # Write as snappy compressed Parquet
                out_path = grid_nodes_dir / f"node_{lat:.1f}_{lon:.1f}.parquet"
                df.to_parquet(out_path, index=False, compression="snappy")

                node_key = f"{lat:.1f}_{lon:.1f}"
                if node_key not in state["completed_nodes"]:
                    state["completed_nodes"].append(node_key)

            # Save state after successfully completing batch
            save_state(state_path, state)
            logger.info(f"Successfully processed batch {idx + 1}/{len(batches)}.")

            # Rate limit spacing delay (2.0 seconds) to prevent slamming Open-Meteo
            time.sleep(2.0)

        except Exception as e:
            logger.error(f"Error processing data for batch {idx + 1}/{len(batches)}: {e}")
            print_resume_instructions(idx)
            return

    logger.info("All weather data ingestion completed successfully!")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    p = argparse.ArgumentParser(description="Open-Meteo weather ingestion.")
    p.add_argument("--mode", choices=["reanalysis", "forecast"], default="reanalysis")
    p.add_argument("--start-date", default="2022-01-01")
    p.add_argument("--end-date", default="2026-06-01")
    args = p.parse_args()
    run_weather_ingestion(start_date=args.start_date, end_date=args.end_date, mode=args.mode)
