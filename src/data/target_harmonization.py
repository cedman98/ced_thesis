"""Target Data Harmonization Pipeline for 50Hertz Actual Macro Generation.

This script parses, cleans, and standardizes historical actual wind (onshore) 
and solar (PV) power generation time series from 50Hertz TSO data, 
resolving Daylight Saving Time (DST) transitions, handling missing values, 
and exporting a harmonized 15-minute resolution Parquet file in UTC.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from src.data.boundary_fetcher import load_config

logger = logging.getLogger(__name__)


def interpolate_short_gaps(s: pd.Series, max_gap: int = 3) -> pd.Series:
    """Interpolates NaN gaps in a time series only if the gap size is <= max_gap.

    Leading and trailing NaNs are strictly preserved and not interpolated.

    Args:
        s: Input pandas Series.
        max_gap: Maximum consecutive NaNs to interpolate. Defaults to 3.

    Returns:
        The Series with short NaN gaps interpolated.
    """
    if s.empty or s.isna().all():
        return s

    first_valid = s.first_valid_index()
    last_valid = s.last_valid_index()
    
    valid_slice = s.loc[first_valid:last_valid]
    is_nan = valid_slice.isna()
    
    # Cumulative sum of non-NaN values creates unique group IDs for each block of contiguous NaNs
    nan_counts = is_nan.groupby((~is_nan).cumsum()).transform('sum')
    
    # Linear interpolation
    interpolated_slice = valid_slice.interpolate(method='linear')
    
    # Only keep interpolated values where the original gap size was <= max_gap
    filled_slice = interpolated_slice.where(nan_counts <= max_gap, valid_slice)
    
    result = s.copy()
    result.loc[first_valid:last_valid] = filled_slice
    return result


def parse_50hertz_csv(filepath: Path, is_wind: bool) -> pd.DataFrame:
    """Parses and standardizes a single 50Hertz actual generation CSV file.

    Handles UTF-16-LE encoding, skips metadata header rows, resolves local
    timestamps in the Europe/Berlin timezone (DST-aware), and cleans float values.

    Args:
        filepath: Path to the CSV file.
        is_wind: True if processing a wind file, False for solar.

    Returns:
        A DataFrame indexed by 'timestamp_utc' containing a single cleaned generation column.
    """
    logger.info(f"Parsing 50Hertz {'wind' if is_wind else 'solar'} file: {filepath.name}")
    
    # Read CSV skipping the first 4 metadata rows
    df = pd.read_csv(filepath, sep=';', encoding='utf-16-le', skiprows=4)
    
    # Clean whitespace in column headers
    df.columns = df.columns.str.strip()
    
    # Standardize start time column name
    df = df.rename(columns={'Von': 'von'})
    
    if 'Datum' not in df.columns or 'von' not in df.columns:
        raise ValueError(f"Missing required time columns 'Datum' or 'von' in {filepath}")

    # Identify value column
    if is_wind:
        if 'Onshore MW' not in df.columns:
            # Fallback check for case variations
            alt_cols = [c for c in df.columns if 'onshore' in c.lower()]
            if alt_cols:
                val_col = alt_cols[0]
            else:
                raise ValueError(f"Missing 'Onshore MW' column in wind file: {filepath}")
        else:
            val_col = 'Onshore MW'
    else:
        if 'MW' not in df.columns:
            raise ValueError(f"Missing 'MW' column in solar file: {filepath}")
        val_col = 'MW'

    # Filter out empty rows
    df = df.dropna(subset=['Datum', 'von']).copy()

    # Clean numeric representations (replace comma decimals and placeholders)
    val_series = df[val_col].astype(str).str.strip().str.replace(',', '.', regex=False)
    val_series = val_series.replace({
        '-': np.nan, 'n.v.': np.nan, 'N.V.': np.nan,
        'nv': np.nan, 'NV': np.nan, 'nan': np.nan, '': np.nan
    })
    df['value_clean'] = pd.to_numeric(val_series, errors='coerce')

    # Construct and parse local datetimes
    naive_dt_str = df['Datum'].str.strip() + ' ' + df['von'].str.strip()
    naive_dts = pd.to_datetime(naive_dt_str, format='%d.%m.%Y %H:%M')

    # Localize to Europe/Berlin (DST-aware) and convert to UTC
    localized_dts = naive_dts.dt.tz_localize('Europe/Berlin', ambiguous='infer')
    df['timestamp_utc'] = localized_dts.dt.tz_convert('UTC').dt.tz_localize(None)

    # Dedup and set index
    res_df = df[['timestamp_utc', 'value_clean']].copy()
    res_df = res_df.drop_duplicates(subset=['timestamp_utc'])
    res_df = res_df.set_index('timestamp_utc')

    return res_df


def run_target_harmonization(
    config_path: str = "conf/config.yaml",
    wind_dir: str = "data/wind/50hertz",
    solar_dir: str = "data/solar",
    output_path: str = "data/processed/actual_generation_50hertz.parquet",
    start_date: str = "2022-01-01 00:00:00",
    end_date: str = "2026-06-01 00:00:00",
) -> None:
    """Executes target actual generation harmonization pipeline.

    Args:
        config_path: Path to the YAML configuration file.
        wind_dir: Directory containing 50Hertz wind CSVs.
        solar_dir: Directory containing 50Hertz solar CSVs.
        output_path: Filepath where the output Parquet will be saved.
        start_date: Target start local time string.
        end_date: Target end local time string.
    """
    logger.info("Initializing 50Hertz macro-target harmonization pipeline...")

    # Load configuration
    try:
        load_config(config_path)
    except Exception as e:
        logger.warning(f"Could not load config file: {e}. Proceeding with defaults.")

    # Define target index range in UTC
    target_idx_local = pd.date_range(
        start=start_date, end=end_date, freq="15min", tz="Europe/Berlin"
    )
    target_idx_utc = target_idx_local.tz_convert("UTC").tz_localize(None)

    # 1. Process Wind Files
    wind_path = Path(wind_dir)
    wind_files = list(wind_path.glob("Windenergie_Hochrechnung_*.csv"))
    if not wind_files:
        raise FileNotFoundError(f"No wind CSV files found in {wind_dir}")

    wind_dfs = []
    for f in wind_files:
        # We only process years within range (2022 to 2026)
        match = re.search(r"(\d{4})", f.name)
        if match:
            year = int(match.group(1))
            if 2022 <= year <= 2026:
                wind_dfs.append(parse_50hertz_csv(f, is_wind=True))

    if not wind_dfs:
        raise ValueError("No wind CSV files within range (2022-2026) were parsed.")
        
    wind_master = pd.concat(wind_dfs).sort_index()
    wind_master = wind_master.rename(columns={'value_clean': 'wind_onshore_mw_50hz'})

    # 2. Process Solar Files
    solar_path = Path(solar_dir)
    solar_files = list(solar_path.glob("Solarenergie_Hochrechnung_*.csv"))
    if not solar_files:
        raise FileNotFoundError(f"No solar CSV files found in {solar_dir}")

    solar_dfs = []
    for f in solar_files:
        match = re.search(r"(\d{4})", f.name)
        if match:
            year = int(match.group(1))
            if 2022 <= year <= 2026:
                solar_dfs.append(parse_50hertz_csv(f, is_wind=False))

    if not solar_dfs:
        raise ValueError("No solar CSV files within range (2022-2026) were parsed.")

    solar_master = pd.concat(solar_dfs).sort_index()
    solar_master = solar_master.rename(columns={'value_clean': 'solar_pv_mw_50hz'})

    # 3. Consolidate and Reindex on the target range
    consolidated_df = pd.DataFrame(index=target_idx_utc)
    consolidated_df = consolidated_df.join(wind_master, how='left')
    consolidated_df = consolidated_df.join(solar_master, how='left')

    # 4. Handle missing entries / short gaps (<= 3 indices)
    logger.info("Interpolating short gaps (<= 3 steps)...")
    consolidated_df['wind_onshore_mw_50hz'] = interpolate_short_gaps(consolidated_df['wind_onshore_mw_50hz'], max_gap=3)
    consolidated_df['solar_pv_mw_50hz'] = interpolate_short_gaps(consolidated_df['solar_pv_mw_50hz'], max_gap=3)

    # Format output columns
    consolidated_df = consolidated_df.reset_index().rename(columns={'index': 'timestamp_utc'})

    # Ensure correct parent directory exists for export
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 5. Export to Parquet
    logger.info(f"Exporting consolidated time-series to: {output_path}")
    consolidated_df.to_parquet(out_file, index=False)
    logger.info("Target data harmonization pipeline completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    run_target_harmonization()
