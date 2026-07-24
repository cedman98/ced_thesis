import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set a nice theme
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def process_capacity(df, type_name):
    df['commissioning_date'] = pd.to_datetime(df['commissioning_date'], errors='coerce')
    df['final_decommission_date'] = pd.to_datetime(df['final_decommission_date'], errors='coerce')
    
    # Additions
    additions = df[['commissioning_date', 'gross_power']].copy()
    additions = additions.rename(columns={'commissioning_date': 'date', 'gross_power': 'power_kw'})
    
    # Decommissions
    decomms = df.dropna(subset=['final_decommission_date'])[['final_decommission_date', 'gross_power']].copy()
    decomms = decomms.rename(columns={'final_decommission_date': 'date', 'gross_power': 'power_kw'})
    decomms['power_kw'] = -decomms['power_kw']
    
    # Combine
    events = pd.concat([additions, decomms])
    events = events.dropna(subset=['date'])
    
    # Aggregate by month
    events['month'] = events['date'].dt.to_period('M').dt.to_timestamp()
    
    monthly = events.groupby('month')['power_kw'].sum().reset_index()
    monthly = monthly.sort_values('month')
    monthly['cumulative_gw'] = monthly['power_kw'].cumsum() / 1e6
    monthly['tech'] = type_name
    return monthly

def main():
    solar_path = 'data/raw/mastr_solar_brandenburg_raw.csv'
    wind_path = 'data/raw/mastr_wind_brandenburg_raw.csv'
    
    df_solar = pd.read_csv(solar_path)
    df_wind = pd.read_csv(wind_path)
    
    monthly_solar = process_capacity(df_solar, 'Solar (PV)')
    monthly_wind = process_capacity(df_wind, 'Wind (Onshore)')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.step(monthly_solar['month'], monthly_solar['cumulative_gw'], label='Solar (PV)', where='post', linewidth=2, color='#e67e22')
    ax.step(monthly_wind['month'], monthly_wind['cumulative_gw'], label='Wind (Onshore)', where='post', linewidth=2, color='#2980b9')
    
    ax.set_title('Cumulative Installed Nameplate Capacity in Brandenburg', fontsize=14, weight='bold')
    ax.set_xlabel('Commissioning Year', fontsize=12)
    ax.set_ylabel('Installed Capacity (GW)', fontsize=12)
    
    ax.legend(title='Technology', fontsize=11)
    
    # Set x-axis limit from 2000 to present
    ax.set_xlim(pd.Timestamp('2000-01-01'), pd.Timestamp('2026-07-01'))
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    out_path = 'results/commissioning_trend.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_path}")

if __name__ == '__main__':
    main()
