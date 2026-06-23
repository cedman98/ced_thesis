import pandas as pd
import numpy as np
import json
import os
import logging
from typing import List

# Configure logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_actual_generation(filepath: str) -> pd.DataFrame:
    """Loads and preprocesses the 50Hertz actual generation data.
    
    Args:
        filepath: Path to the parquet file.
        
    Returns:
        pd.DataFrame: DataFrame with a UTC datetime index.
    """
    logging.info(f"Loading actual generation data from {filepath}")
    df = pd.read_parquet(filepath)
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
    df = df.set_index('timestamp_utc')
    return df

def generate_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates rolling means and cyclical time encodings.
    
    Args:
        df: DataFrame containing weather and physical features.
        
    Returns:
        pd.DataFrame: DataFrame augmented with lag and cyclical features.
    """
    # 1. Rolling 24-hour means
    # Assumes the DataFrame index is sorted datetime and frequency is regular
    if 'wind_speed_10m' in df.columns:
        df['wind_speed_10m_roll24'] = df['wind_speed_10m'].rolling('24h').mean()
    if 'shortwave_radiation' in df.columns:
        df['shortwave_radiation_roll24'] = df['shortwave_radiation'].rolling('24h').mean()
        
    # 2. Time-based cyclical encodings
    hour = df.index.hour
    month = df.index.month
    
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * month / 12.0)
    
    return df

def load_and_merge_cluster_data(mapping_file: str, sample_size: int = 5) -> pd.DataFrame:
    """Loops through clusters and merges weather and physical features on the time index.
    
    Args:
        mapping_file: Path to the JSON mapping index.
        sample_size: Number of clusters to process (for testing).
        
    Returns:
        pd.DataFrame: A concatenated long-format DataFrame of cluster features.
    """
    logging.info(f"Loading mapping index from {mapping_file}")
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
        
    # Prioritize solar clusters for the dummy test since geometry data exists
    clusters = [k for k in mapping.keys() if 'solar' in k][:sample_size]
    if not clusters:
        # Fallback to any cluster
        clusters = list(mapping.keys())[:sample_size]
        
    cluster_dfs: List[pd.DataFrame] = []
    
    for cluster_id in clusters:
        logging.info(f"Processing cluster: {cluster_id}")
        
        # Load corresponding weather grid node data
        weather_path = mapping[cluster_id]['parquet_path']
        if not os.path.exists(weather_path):
            logging.warning(f"Weather file {weather_path} not found. Skipping {cluster_id}.")
            continue
            
        weather_df = pd.read_parquet(weather_path)
        weather_df['time'] = pd.to_datetime(weather_df['time'], utc=True)
        weather_df = weather_df.set_index('time')
        
        # Resample weather from 1-hour to 15-min frequency (forward-fill)
        weather_df = weather_df.resample('15min').ffill()
        
        # Load physical data (Solar Geometry or Wind Power Curves)
        phys_df = None
        if 'solar' in cluster_id:
            phys_path = f"data/processed/solar_geometry/{cluster_id}.parquet"
        else:
            phys_path = f"data/processed/wind_power_curves/{cluster_id}.parquet"
            
        if os.path.exists(phys_path):
            phys_df = pd.read_parquet(phys_path)
            # Physical features usually use 'timestamp', convert to timezone-aware UTC
            phys_df.index = pd.to_datetime(phys_df.index, utc=True)
            phys_df.index.name = 'time'
        else:
            logging.warning(f"Physical feature file {phys_path} not found. Proceeding with weather only.")
            
        # Merge weather and physical features on time index
        if phys_df is not None:
            cluster_df = pd.merge(weather_df, phys_df, left_index=True, right_index=True, how='inner')
        else:
            cluster_df = weather_df
            
        # Add spatial/cluster identification
        cluster_df['cluster_id'] = cluster_id
        
        # Append to master list
        cluster_dfs.append(cluster_df)
        
    if not cluster_dfs:
        raise ValueError("No cluster data was successfully loaded.")
        
    # Concatenate all clusters into a single long-format matrix
    final_features_df = pd.concat(cluster_dfs)
    
    # Generate temporal lag features efficiently on the combined structure
    # Because we're computing rolling metrics, we group by cluster to avoid bleeding values
    logging.info("Generating lag and cyclical features...")
    # Sort first to ensure chronological rolling
    final_features_df = final_features_df.sort_index()
    
    # Apply rolling by cluster
    def apply_lags(group):
        return generate_lag_features(group)
        
    final_features_df = final_features_df.groupby('cluster_id', group_keys=False).apply(apply_lags)
    
    return final_features_df

def build_feature_matrix() -> None:
    """Main orchestrator function to build the complete ML training matrix."""
    actual_gen_path = 'data/processed/actual_generation_50hertz.parquet'
    mapping_path = 'data/processed/weather/mapping_index.json'
    output_path = 'data/processed/ml_ready_matrix_sample.parquet'
    
    # 1. Load actual target generation
    actual_df = load_actual_generation(actual_gen_path)
    
    # 2 & 3 & 4. Load cluster features and generate lags
    features_df = load_and_merge_cluster_data(mapping_path, sample_size=5)
    
    # 5. Join actual generation targets onto the feature matrix
    # Both DataFrames have UTC time index
    features_df.index.name = 'time'
    actual_df.index.name = 'time'
    
    logging.info("Joining target variables (actual generation) to the feature matrix...")
    final_matrix = features_df.join(actual_df, how='inner')
    
    # Save the dummy-tested matrix
    logging.info(f"Saving final ML matrix to {output_path} with shape {final_matrix.shape}")
    final_matrix.to_parquet(output_path)
    logging.info("Master feature matrix successfully saved.")

if __name__ == '__main__':
    build_feature_matrix()
