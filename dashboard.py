"""Brandenburg State-Wide Generation Forecast — operational dashboard.

Run:  uv run streamlit run dashboard.py

Select any of the 18 Brandenburg Kreise (14 Landkreise + 4 kreisfreie Städte) or the
state total aggregate. The app serves the pre-computed batch forecast
(results/forecasts/brandenburg_state_forecast.parquet from notebook 11) when present,
and otherwise runs the 24h pipeline live against the Open-Meteo forecast API for the
selected district. Median (P50) is the point forecast; the shaded band is the
80% (P10–P90) interval.
"""

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.districts import district_registry, load_district_fleets
from src.dashboard.forecast_service import forecast_district

st.set_page_config(page_title="Brandenburg Generation Forecast", page_icon="⚡", layout="wide")

WIND = {"line": "#0e7490", "band": "rgba(14,116,144,0.18)"}
SOLAR = {"line": "#d97706", "band": "rgba(217,119,6,0.18)"}
AGG = "Brandenburg (Total Aggregate)"
PARQUET = "results/forecasts/brandenburg_state_forecast.parquet"
MW_COLS = ["wind_lower_mw", "wind_median_mw", "wind_upper_mw",
           "solar_lower_mw", "solar_median_mw", "solar_upper_mw"]


@st.cache_data(ttl=1800, show_spinner=False)
def load_state_parquet(mtime):
    """Pre-computed batch forecast, or None. `mtime` keys the cache to the file."""
    if not os.path.exists(PARQUET):
        return None
    return pd.read_parquet(PARQUET)


@st.cache_resource(show_spinner="Assigning MaStR assets to districts…")
def get_fleets():
    return load_district_fleets()


@st.cache_data(ttl=1800, show_spinner=False)
def live_district(name, issue_hour):
    fc, caps, _ = forecast_district(name, get_fleets()[name], pause=0.0)
    return fc, caps


def band_figure(fc, tech, style, title):
    t = fc.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=fc[f'{tech}_upper_mw'], line=dict(width=0),
                             showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=t, y=fc[f'{tech}_lower_mw'], fill='tonexty', fillcolor=style['band'],
                             line=dict(width=0), name='80% interval (P10–P90)', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=t, y=fc[f'{tech}_median_mw'], line=dict(color=style['line'], width=2.6),
                             name='Median forecast (P50)',
                             hovertemplate='%{x|%a %H:%M} UTC<br>%{y:.1f} MW<extra></extra>'))
    fig.update_layout(
        title=dict(text=title, font=dict(size=17)),
        template='plotly_white', height=340, margin=dict(l=10, r=10, t=42, b=10),
        yaxis_title='MW', xaxis_title=None, hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.update_yaxes(rangemode='tozero')
    return fig


def render(fc, subtitle, caps=None, aggregate=False):
    st.caption(subtitle)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Peak wind (P50)", f"{fc['wind_median_mw'].max():,.1f} MW")
    k2.metric("Peak solar (P50)", f"{fc['solar_median_mw'].max():,.1f} MW")
    k3.metric("Wind energy (24 h)", f"{fc['wind_median_mw'].sum():,.0f} MWh")
    k4.metric("Solar energy (24 h)", f"{fc['solar_median_mw'].sum():,.0f} MWh")
    st.plotly_chart(band_figure(fc, 'wind', WIND, "Wind generation"), width='stretch')
    st.plotly_chart(band_figure(fc, 'solar', SOLAR, "Solar generation"), width='stretch')
    if aggregate:
        st.caption(
            "**Aggregate note:** the state band is the sum of district P10/P90, i.e. the "
            "*comonotonic* (perfect-correlation) aggregation — a conservative upper bound on "
            "the true interval, since spatial forecast errors partly decorrelate."
        )
    with st.expander("Forecast table (MW)"):
        show = fc.copy()
        show.index = show.index.strftime('%Y-%m-%d %H:%M UTC')
        st.dataframe(show[MW_COLS].round(2), width='stretch')


# ---- Sidebar ----------------------------------------------------------------
names = [n for n, _, _ in district_registry()]
st.sidebar.title("⚡ Forecast controls")
choice = st.sidebar.selectbox("Region", [AGG] + sorted(names))

parquet = load_state_parquet(os.path.getmtime(PARQUET) if os.path.exists(PARQUET) else 0)
issue_hour = pd.Timestamp.now(tz='UTC').floor('h')
src_note = ("pre-computed batch (notebook 11)" if parquet is not None else "live Open-Meteo NWP")
st.sidebar.caption(f"Source: {src_note}")
st.sidebar.caption("Uncalibrated districts use identity downscaling (physics-trust); "
                   "fit an affine k per district once DSO telemetry exists.")

# ---- Main -------------------------------------------------------------------
st.title("Brandenburg — 24 h Generation Forecast")

try:
    if choice == AGG:
        if parquet is not None:
            fc = parquet.groupby(level="time_utc")[MW_COLS].sum()
        else:
            frames = []
            prog = st.progress(0.0, "Running live state-wide forecast…")
            fleets = get_fleets()
            for i, (name, fleet) in enumerate(fleets.items(), 1):
                f, _, _ = forecast_district(name, fleet, pause=0.5)
                frames.append(f[MW_COLS]); prog.progress(i / len(fleets), f"{name} ({i}/18)")
            prog.empty()
            fc = pd.concat(frames).groupby(level=0).sum()
        render(fc, f"All 18 Kreise summed · issued {issue_hour:%Y-%m-%d %H:%M} UTC · {src_note}",
               aggregate=True)
    else:
        if parquet is not None and choice in parquet.index.get_level_values("district"):
            fc = parquet.xs(choice, level="district")
            caps = {"wind": fc["wind_nameplate_mw"].iloc[0], "solar": fc["solar_nameplate_mw"].iloc[0]}
        else:
            fc, caps = live_district(choice, issue_hour)
        st.sidebar.markdown("**Installed capacity**")
        st.sidebar.metric("Wind", f"{caps['wind']:,.1f} MW")
        st.sidebar.metric("Solar", f"{caps['solar']:,.1f} MW")
        render(fc, f"{choice} · issued {issue_hour:%Y-%m-%d %H:%M} UTC · "
                   f"window {fc.index.min():%a %H:%M} → {fc.index.max():%a %H:%M} UTC", caps)
except Exception as e:
    st.error(f"Could not build forecast for {choice}: {e}")
    st.stop()

st.caption(
    "Median (P50) is the point forecast; the shaded band is the 80% prediction interval from the "
    "LightGBM quantile models. Intervals are indicative — quantiles are trained on the smoother "
    "50Hertz zonal signal, so local coverage runs below nominal (see PICP in the validation report)."
)
