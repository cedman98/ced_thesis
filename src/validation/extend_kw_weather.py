"""Extend municipal weather nodes past the training cut-off so each configured
municipality has multi-week overlap with its E.ON telemetry instead of the few
days that training-era weather allowed.

Loops over every municipality in conf/config.yaml (scraper.regions), derives the
0.1-snapped weather nodes around it (same rule as build_municipal_features), and
back-fills each node from its current max timestamp up to the municipality's
latest telemetry hour. Macro training is unaffected (feature_pipeline inner-joins
on targets that end earlier, dropping these rows).
"""

import logging
import os
import time
import numpy as np
import pandas as pd
import requests

from src.validation.kw_features import (
    load_municipalities, load_edis_telemetry, haversine_distance, RADIUS_KM)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
NODE_DIR = "data/processed/weather/grid_nodes"


def municipal_nodes(lat, lon, radius_km=RADIUS_KM):
    """0.1-snapped (lat, lon) weather nodes covering assets within radius_km."""
    nodes = set()
    for tech in ('wind', 'solar'):
        df = pd.read_csv(f'data/raw/mastr_{tech}_brandenburg_raw.csv')
        d = haversine_distance(df['latitude'], df['longitude'], lat, lon)
        near = df[d <= radius_km]
        nodes.update(zip(near['latitude'].round(1), near['longitude'].round(1)))
    return sorted(nodes)


def extend_node(lat, lon, end_date):
    path = f"{NODE_DIR}/node_{lat}_{lon}.parquet"
    if not os.path.exists(path):
        return False
    existing = pd.read_parquet(path)
    existing['time'] = pd.to_datetime(existing['time'], utc=True)
    start = existing['time'].max().date()
    if str(start) >= end_date:
        return False  # already current
    variables = [c for c in existing.columns if c != 'time']

    r = requests.get(ARCHIVE, params={
        "latitude": lat, "longitude": lon, "start_date": str(start), "end_date": end_date,
        "hourly": ",".join(variables), "timezone": "UTC",
    }, timeout=60)
    r.raise_for_status()
    h = r.json().get("hourly", {})
    if not h.get("time"):
        logger.warning(f"no data for {lat},{lon}")
        return False
    new = pd.DataFrame(h)
    new['time'] = pd.to_datetime(new['time'], utc=True)
    # ERA5(T) lags realtime by a few days: the API pads recent hours with nulls.
    # Drop all-NaN rows so the node's max timestamp stays at real data and the
    # next run re-requests those hours instead of skipping them forever.
    new = new.dropna(how='all', subset=variables)
    if new.empty:
        return False
    merged = (pd.concat([existing, new[existing.columns]], ignore_index=True)
              .drop_duplicates(subset='time', keep='last').sort_values('time'))
    merged.to_parquet(path, index=False)
    logger.info(f"{lat},{lon}: -> {merged['time'].max()}")
    time.sleep(0.5)
    return True


def extend():
    for m in load_municipalities():
        end_date = str(load_edis_telemetry(m['telemetry']).index.max().date())
        nodes = municipal_nodes(m['lat'], m['lon'])
        logger.info(f"[{m['slug']}] extending {len(nodes)} nodes through {end_date}")
        n = sum(extend_node(lat, lon, end_date) for lat, lon in nodes)
        logger.info(f"[{m['slug']}] updated {n} nodes.")


def extend_all(end_date=None):
    """Extend EVERY archived grid node through yesterday (or `end_date`).

    This is the cron entrypoint: the batch ingester's completed_nodes state is
    binary, so once all nodes were fetched a nightly run_weather_ingestion is a
    permanent no-op and the archive silently stops growing. Extension appends
    from each node's own max timestamp instead.
    """
    import glob
    end_date = end_date or str((pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=1)).date())
    paths = sorted(glob.glob(f"{NODE_DIR}/node_*.parquet"))
    logger.info(f"extending {len(paths)} grid nodes through {end_date}")
    n, failed = 0, 0
    for p in paths:
        lat, lon = os.path.basename(p)[len('node_'):-len('.parquet')].split('_')
        # One flaky call must not abort the remaining nodes; a skipped node is
        # simply retried on the next nightly run (it extends from its own max).
        try:
            n += extend_node(lat, lon, end_date)
        except requests.RequestException as e:
            failed += 1
            logger.warning(f"{lat},{lon}: skipped ({e})")
    logger.info(f"updated {n}/{len(paths)} nodes ({failed} failed, will retry next run).")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Extend archived weather nodes.")
    p.add_argument("--all", action="store_true",
                   help="extend every grid node through yesterday (cron mode); "
                        "default extends only the municipal validation nodes")
    p.add_argument("--end-date", default=None, help="YYYY-MM-DD (only with --all)")
    args = p.parse_args()
    extend_all(args.end_date) if args.all else extend()
