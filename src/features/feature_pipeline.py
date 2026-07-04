import pandas as pd
import numpy as np
import json
import logging
import os
from src.models.wind_physics_transformer import WindPowerCalculator

# Configure robust logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Representative modern turbine used to build a *per-unit* wind power curve
# (dimensionless CF), which is then scaled by each cluster's nameplate.
REF_TURBINE_KW = 3000.0
# Generic PV performance ratio (soiling, temperature, inverter, wiring losses).
SOLAR_PERFORMANCE_RATIO = 0.80

def load_and_downsample_targets(filepath: str = 'data/processed/actual_generation_50hertz.parquet') -> pd.DataFrame:
    """Loads 15-min actual generation data and downsamples to 1-hour resolution."""
    logger.info(f"Loading and downsampling targets from {filepath}")
    df = pd.read_parquet(filepath)
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
    df = df.set_index('timestamp_utc')
    df_hourly = df.resample('1h').mean()
    return df_hourly

def calculate_regional_wind_prior(weather_dir: str = 'data/processed/weather') -> pd.DataFrame:
    """Calculates macro physical wind prior by aggregating power from all wind clusters."""
    logger.info("Calculating regional wind prior...")
    clusters_df = pd.read_csv('data/processed/wind_clusters.csv')

    with open(f'{weather_dir}/mapping_index.json', 'r') as f:
        mapping = json.load(f)
        
    calculator = WindPowerCalculator()
    all_priors = []
    
    for _, row in clusters_df.iterrows():
        cluster_id = row['cluster_id']
        capacity_mw = row['total_capacity_mw']
        capacity_kw = capacity_mw * 1000.0
        
        if cluster_id not in mapping:
            continue
            
        weather_path = mapping[cluster_id]['parquet_path']
        if not os.path.exists(weather_path):
            continue
            
        weather_df = pd.read_parquet(weather_path)
        weather_df['time'] = pd.to_datetime(weather_df['time'], utc=True)
        weather_df = weather_df.set_index('time')
        
        if 'wind_speed_100m' not in weather_df.columns:
            logger.warning(f"'wind_speed_100m' not found in {weather_path}")
            continue
            
        # Open-Meteo outputs wind speed in km/h by default; divide by 3.6 to convert to m/s
        v_100m = weather_df['wind_speed_100m'].values / 3.6

        # Per-unit power curve (CF in [0,1]) from a representative turbine, then
        # scaled by the *whole cluster* nameplate. The old code passed the cluster
        # nameplate as gross_power, which matched a single ~10 MW curve and capped a
        # 366 MW / 153-asset cluster at 10 MW (~30x under-scaled prior).
        try:
            ref_kw = calculator.calculate_power(
                v_100m=v_100m,
                hub_height=100.0,
                manufacturer_id=None,
                type_designation="Generic",
                gross_power=REF_TURBINE_KW,
                rotor_diameter=None
            )
        except Exception as e:
            logger.error(f"Failed to calculate wind power for cluster {cluster_id}: {e}")
            continue

        per_unit = np.clip(ref_kw / REF_TURBINE_KW, 0.0, 1.0)
        cluster_prior = pd.DataFrame(
            {'power_mw': per_unit * capacity_mw},
            index=weather_df.index
        )
        all_priors.append(cluster_prior)
        
    if not all_priors:
        logger.warning("No wind priors could be computed.")
        return pd.DataFrame()
        
    total_wind_prior = pd.concat(all_priors, axis=1).sum(axis=1)
    total_wind_prior.name = 'brandenburg_wind_prior_mw'
    return pd.DataFrame(total_wind_prior)

def calculate_regional_solar_prior(weather_dir: str = 'data/processed/weather') -> pd.DataFrame:
    """Calculates macro physical solar prior by aggregating power from all solar clusters."""
    logger.info("Calculating regional solar prior...")
    clusters_df = pd.read_csv('data/processed/solar_clusters.csv')

    with open(f'{weather_dir}/mapping_index.json', 'r') as f:
        mapping = json.load(f)
        
    all_priors = []
    
    for _, row in clusters_df.iterrows():
        cluster_id = row['cluster_id']
        capacity_mw = row['total_capacity_mw']
        
        if cluster_id not in mapping:
            continue
            
        weather_path = mapping[cluster_id]['parquet_path']
        if not os.path.exists(weather_path):
            continue
            
        weather_df = pd.read_parquet(weather_path)
        weather_df['time'] = pd.to_datetime(weather_df['time'], utc=True)
        weather_df = weather_df.set_index('time')
        
        geom_path = f"data/processed/solar_geometry/{cluster_id}.parquet"
        if os.path.exists(geom_path):
            geom_df = pd.read_parquet(geom_path)
            geom_df.index = pd.to_datetime(geom_df.index, utc=True)
            geom_hourly = geom_df.resample('1h').mean()
            merged = weather_df.join(geom_hourly, how='inner')
        else:
            merged = weather_df
            
        if 'shortwave_radiation' not in merged.columns:
            logger.warning(f"'shortwave_radiation' not found for cluster {cluster_id}")
            continue
            
        # Physical solar prior: shortwave (W/m^2) normalised to STC (1000 W/m^2)
        # yields the DC capacity factor, times a generic performance ratio. The old
        # code omitted the /1000, inflating the prior ~1000x (max 1.19e6 MW).
        power_mw = (merged['shortwave_radiation'] / 1000.0) * capacity_mw * SOLAR_PERFORMANCE_RATIO
        
        cluster_prior = pd.DataFrame(
            {'power_mw': power_mw},
            index=merged.index
        )
        all_priors.append(cluster_prior)
        
    if not all_priors:
        logger.warning("No solar priors could be computed.")
        return pd.DataFrame()
        
    total_solar_prior = pd.concat(all_priors, axis=1).sum(axis=1)
    total_solar_prior.name = 'brandenburg_solar_prior_mw'
    return pd.DataFrame(total_solar_prior)

def calculate_mean_weather_features(weather_dir: str = 'data/processed/weather') -> pd.DataFrame:
    """Calculates hourly spatial average for temperature, pressure, and humidity across all grids."""
    logger.info("Calculating mean regional weather features...")
    with open(f'{weather_dir}/mapping_index.json', 'r') as f:
        mapping = json.load(f)
        
    unique_parquets = list(set([v['parquet_path'] for v in mapping.values()]))
    weather_dfs = []
    cols = ['temperature_2m', 'surface_pressure', 'relative_humidity_2m']
    
    for path in unique_parquets:
        if not os.path.exists(path):
            continue
            
        df = pd.read_parquet(path)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.set_index('time')
        
        available_cols = [c for c in cols if c in df.columns]
        if available_cols:
            weather_dfs.append(df[available_cols])
            
    if not weather_dfs:
        logger.warning("No weather grid data found to compute regional means.")
        return pd.DataFrame()
        
    combined = pd.concat(weather_dfs)
    regional_mean = combined.groupby(combined.index).mean()
    return regional_mean

def _effective_capacity(y: pd.Series):
    """Data-driven installed-capacity proxy: CAUSAL trailing-365-day robust peak
    (p99.9) of generation, shifted one step so cap_t only uses data strictly
    before t. The previous per-calendar-year percentile normalised every test
    timestamp with a capacity computed from that same year's (future) data —
    look-ahead inside the CV test folds.

    cummax enforces a physically monotone fleet (capacity does not shrink) and
    keeps the CF stable through becalmed periods. Tracks solar's commissioning
    growth and wind's near-flat capacity without an external registry. Replace
    with official 50Hertz installed-capacity figures via cf_normalization.json
    if/when they are available. The first ~30 days have no capacity estimate
    (NaN) and are dropped with the rest of the matrix NaNs.
    """
    cap = (y.sort_index()
            .rolling('365D', min_periods=24 * 30).quantile(0.999)
            .shift(1).cummax())
    per_year = cap.groupby(cap.index.year).last().dropna()  # year-end snapshot (reporting only)
    return cap, per_year


def build_feature_matrix(mode: str = 'reanalysis') -> None:
    """Assembles the final Spatio-Temporal Fusion Matrix for ML training.

    mode='reanalysis' reads ERA5 actuals (data/processed/weather/) and writes
    ml_training_matrix.parquet. mode='forecast' reads the NWP forecast archive
    (data/processed/weather_forecast/) and writes ml_training_matrix_forecast.parquet.
    The 50Hertz target is identical in both, so the only difference between the
    two matrices is the weather that drives the physical priors + weather means —
    exactly the quantity the NWP-gap study isolates.
    """
    if mode not in ('reanalysis', 'forecast'):
        raise ValueError(f"mode must be 'reanalysis' or 'forecast', got {mode!r}")
    weather_dir = 'data/processed/weather' if mode == 'reanalysis' else 'data/processed/weather_forecast'
    logger.info(f"Building Spatio-Temporal Fusion Matrix [{mode}] from {weather_dir}...")

    targets = load_and_downsample_targets()
    wind_prior = calculate_regional_wind_prior(weather_dir)
    solar_prior = calculate_regional_solar_prior(weather_dir)
    weather_mean = calculate_mean_weather_features(weather_dir)
    
    matrix = targets.copy()
    logger.info(f"Starting matrix with targets: {matrix.shape[0]} rows")
    
    matrix = matrix.join(wind_prior, how='inner')
    logger.info(f"After joining wind_prior: {matrix.shape[0]} rows")
    
    matrix = matrix.join(solar_prior, how='inner')
    logger.info(f"After joining solar_prior: {matrix.shape[0]} rows")
    
    matrix = matrix.join(weather_mean, how='inner')
    logger.info(f"After joining weather_mean: {matrix.shape[0]} rows")
    
    logger.info("Adding time-based cyclical features...")
    hour = matrix.index.hour
    month = matrix.index.month
    
    matrix['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    matrix['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    matrix['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    matrix['month_cos'] = np.cos(2 * np.pi * month / 12.0)

    # --- Scale-invariant reformulation -------------------------------------
    # Express the physical priors as capacity factors (prior_mw / fleet_mw) and
    # the 50Hertz targets as capacity factors (mw / effective_capacity). Both
    # target and features are now dimensionless and transfer unchanged from the
    # macro zone to a single municipality, which is what makes downscaling work.
    logger.info("Normalising targets and priors to capacity factors...")
    wind_fleet_mw = pd.read_csv('data/processed/wind_clusters.csv')['total_capacity_mw'].sum()
    solar_fleet_mw = pd.read_csv('data/processed/solar_clusters.csv')['total_capacity_mw'].sum()

    matrix['wind_prior_cf'] = (matrix['brandenburg_wind_prior_mw'] / wind_fleet_mw).clip(0, 1.5)
    matrix['solar_prior_cf'] = (matrix['brandenburg_solar_prior_mw'] / solar_fleet_mw).clip(0, 1.5)

    wind_cap, wind_cap_year = _effective_capacity(matrix['wind_onshore_mw_50hz'])
    solar_cap, solar_cap_year = _effective_capacity(matrix['solar_pv_mw_50hz'])
    matrix['wind_eff_capacity_mw'] = wind_cap
    matrix['solar_eff_capacity_mw'] = solar_cap
    matrix['wind_cf_50hz'] = (matrix['wind_onshore_mw_50hz'] / wind_cap).clip(0, 1.5)
    matrix['solar_cf_50hz'] = (matrix['solar_pv_mw_50hz'] / solar_cap).clip(0, 1.5)

    # Persist the normalisation constants as the single source of truth for the
    # municipal validators (CF-definition reconciliation) and reproducibility.
    norm = {
        'wind': {
            'fleet_capacity_mw': float(wind_fleet_mw),
            'eff_capacity_mw_by_year': {int(k): float(v) for k, v in wind_cap_year.items()},
        },
        'solar': {
            'fleet_capacity_mw': float(solar_fleet_mw),
            'eff_capacity_mw_by_year': {int(k): float(v) for k, v in solar_cap_year.items()},
        },
        'solar_performance_ratio': SOLAR_PERFORMANCE_RATIO,
        'ref_turbine_kw': REF_TURBINE_KW,
    }
    # Only the reanalysis run owns the canonical CF normalisation constants; a
    # forecast run must not clobber the source of truth the validators rely on.
    if mode == 'reanalysis':
        with open('data/processed/cf_normalization.json', 'w') as f:
            json.dump(norm, f, indent=2)

    initial_rows = matrix.shape[0]
    matrix = matrix.dropna()
    dropped_rows = initial_rows - matrix.shape[0]

    logger.info(f"Dropped {dropped_rows} rows containing NaNs.")

    output_path = (
        'data/processed/ml_training_matrix.parquet' if mode == 'reanalysis'
        else 'data/processed/ml_training_matrix_forecast.parquet'
    )
    logger.info(f"Final matrix shape: {matrix.shape}. Saving to {output_path}")
    matrix.to_parquet(output_path)
    logger.info("Master feature matrix successfully saved.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build the ML feature matrix.")
    p.add_argument("--mode", choices=["reanalysis", "forecast"], default="reanalysis")
    build_feature_matrix(p.parse_args().mode)
