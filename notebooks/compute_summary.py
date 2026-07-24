import pandas as pd
import numpy as np

# Load data
wind_df = pd.read_csv('/home/ced/bachelor/final/data/raw/mastr_wind_brandenburg_raw.csv')
solar_df = pd.read_csv('/home/ced/bachelor/final/data/raw/mastr_solar_brandenburg_raw.csv')

# Wind stats
wind_count = len(wind_df)
# gross_power is in kW in MaStR
wind_total_gw = wind_df['gross_power'].sum() / 1e6
wind_median_kw = wind_df['gross_power'].median()
wind_q25_kw = wind_df['gross_power'].quantile(0.25)
wind_q75_kw = wind_df['gross_power'].quantile(0.75)
wind_hub_height_median = wind_df['hub_height'].median()
wind_hub_height_min = wind_df['hub_height'].min()
wind_hub_height_max = wind_df['hub_height'].max()

# Solar stats
solar_count = len(solar_df)
solar_total_gw = solar_df['gross_power'].sum() / 1e6
solar_median_kw = solar_df['gross_power'].median()
solar_q25_kw = solar_df['gross_power'].quantile(0.25)
solar_q75_kw = solar_df['gross_power'].quantile(0.75)

# For solar only: share utility vs. rooftop
if 'is_ground_mounted' in solar_df.columns:
    # is_ground_mounted might have NaNs, dropna for mean
    utility_share = solar_df['is_ground_mounted'].dropna().astype(bool).mean() * 100
    rooftop_share = 100 - utility_share
    solar_mounting_str = f"Utility: {utility_share:.1f}%, Rooftop: {rooftop_share:.1f}%"
else:
    solar_mounting_str = "N/A"

# Build rows
data = [
    {
        'Technology': 'Wind',
        'Asset count': wind_count,
        'Total nameplate GW': round(wind_total_gw, 2),
        'Median asset size (kW)': wind_median_kw,
        'IQR asset size (kW)': f"{wind_q25_kw} - {wind_q75_kw}",
        'Hub-height range or median': f"Median: {wind_hub_height_median}m (Range: {wind_hub_height_min}m - {wind_hub_height_max}m)",
        'Mounting/Location': 'N/A'
    },
    {
        'Technology': 'Solar',
        'Asset count': solar_count,
        'Total nameplate GW': round(solar_total_gw, 2),
        'Median asset size (kW)': solar_median_kw,
        'IQR asset size (kW)': f"{solar_q25_kw} - {solar_q75_kw}",
        'Hub-height range or median': 'N/A',
        'Mounting/Location': solar_mounting_str
    }
]

df_summary = pd.DataFrame(data)

out_path = '/home/ced/bachelor/final/data/interim/mastr_summary.csv'
import os
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df_summary.to_csv(out_path, index=False)
print(f"Successfully wrote summary to {out_path}")
print("\n--- CSV Content ---")
print(df_summary.to_csv(index=False))
