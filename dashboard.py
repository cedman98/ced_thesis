"""Brandenburg State-Wide Generation Forecast — operational dashboard.

Run:  uv run streamlit run dashboard.py

Select any of the 18 Brandenburg Kreise (14 Landkreise + 4 kreisfreie Städte) or the
state total aggregate. The app serves the pre-computed batch forecast
(results/forecasts/brandenburg_state_forecast.parquet from notebook 11) when present,
and otherwise runs the 24h pipeline live against the Open-Meteo forecast API for the
selected district. Median (P50) is the point forecast; the shaded band is the
80% (P10–P90) interval.
"""

import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.districts import district_registry, load_district_fleets, load_district_boundaries
from src.dashboard.forecast_service import forecast_district, local_day_window

st.set_page_config(page_title="Brandenburg Generation Forecast", page_icon="⚡", layout="wide")

WIND = {"line": "#0e7490", "band": "rgba(14,116,144,0.18)"}
SOLAR = {"line": "#d97706", "band": "rgba(217,119,6,0.18)"}
AGG = "Brandenburg (Total Aggregate)"
LOCAL = "Europe/Berlin"  # display tz only; all data/model stay UTC. DST-safe (CEST +2 / CET +1).
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
def live_district(name, window):
    fc, caps, _ = forecast_district(name, get_fleets()[name], pause=0.0, window=window)
    return fc, caps


@st.cache_data(ttl=1800, show_spinner=False)
def load_skill():
    """Macro CV nMAE + PICP per (model, tech), CF scale only. The macro model is
    what every district runs (identity downscaling), so this is their shared skill."""
    m = pd.read_csv("results/macro_cv_metrics.csv")
    m = m[m["scale"] == "cf"]
    return m.groupby(["model", "technology"]).agg(nmae=("nmae", "mean"), picp=("picp", "mean"))


@st.cache_data(ttl=1800, show_spinner=False)
def load_muni_picp():
    """Realized LightGBM interval coverage from the KW+Nauen validation telemetry."""
    v = pd.read_csv("results/municipal_validation_metrics.csv")
    return v[v["model"] == "lightgbm"].groupby("technology")["picp"].mean()


@st.cache_resource(show_spinner=False)
def get_boundaries():
    return load_district_boundaries()


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


def stacked_figure(fc):
    """Combined renewable stack (wind + solar P50) — the headline total."""
    t = fc.index
    total = fc['wind_median_mw'] + fc['solar_median_mw']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=fc['wind_median_mw'], stackgroup='r', name='Wind (P50)',
                             line=dict(width=0.5, color=WIND['line']), fillcolor=WIND['band'],
                             hovertemplate='%{x|%a %H:%M} UTC<br>wind %{y:.0f} MW<extra></extra>'))
    fig.add_trace(go.Scatter(x=t, y=fc['solar_median_mw'], stackgroup='r', name='Solar (P50)',
                             line=dict(width=0.5, color=SOLAR['line']), fillcolor=SOLAR['band'],
                             hovertemplate='%{x|%a %H:%M} UTC<br>solar %{y:.0f} MW<extra></extra>'))
    fig.add_trace(go.Scatter(x=t, y=total, name='Total renewable', line=dict(color='#334155', width=2.4),
                             hovertemplate='%{x|%a %H:%M} UTC<br>total %{y:.0f} MW<extra></extra>'))
    fig.update_layout(title=dict(text='Total renewable generation (wind + solar)', font=dict(size=17)),
                      template='plotly_white', height=360, margin=dict(l=10, r=10, t=42, b=10),
                      yaxis_title='MW', hovermode='x unified',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.update_yaxes(rangemode='tozero')
    return fig


def map_figure(dist_totals):
    """Choropleth of 24 h renewable energy (MWh) across the 18 Kreise."""
    gj = json.loads(get_boundaries().to_json())
    # ponytail: Choroplethmapbox is deprecated (works fine, no token needed via carto-positron);
    # swap to go.Choroplethmap if a plotly upgrade ever drops the mapbox traces.
    fig = go.Figure(go.Choroplethmapbox(
        geojson=gj, locations=dist_totals.index, z=dist_totals.values,
        featureidkey='properties.name', colorscale='YlGnBu', marker_opacity=0.78,
        marker_line_width=0.4, marker_line_color='#64748b', colorbar_title='MWh',
        hovertemplate='%{location}<br>%{z:,.0f} MWh (24 h)<extra></extra>'))
    fig.update_layout(mapbox_style='carto-positron', mapbox_zoom=6.3,
                      mapbox_center={'lat': 52.45, 'lon': 13.4}, height=520,
                      margin=dict(l=0, r=0, t=10, b=0))
    return fig


def render_skill():
    """Objectives 1 + 3: macro-CV skill vs persistence, and interval-coverage honesty."""
    g, muni = load_skill(), load_muni_picp()
    st.subheader("Model skill — macro cross-validation")
    cols = st.columns(4)
    for i, tech in enumerate(('wind', 'solar')):
        lgb, per = g.loc[('lightgbm', tech), 'nmae'], g.loc[('persistence', tech), 'nmae']
        cols[2 * i].metric(f"{tech.title()} nMAE (LightGBM)", f"{lgb:.1%}",
                           f"{1 - lgb / per:.0%} better than persistence")
        cols[2 * i + 1].metric(f"{tech.title()} persistence nMAE", f"{per:.1%}", "naive baseline",
                               delta_color="off")
    st.caption(
        "Skill score = 1 − nMAE(LightGBM) / nMAE(persistence), from `results/macro_cv_metrics.csv` "
        "(purged expanding-window CV on the 50Hertz zonal signal). All 18 districts run this same "
        "macro model via identity downscaling, so it is their shared skill estimate — per-district "
        "nMAE needs DSO telemetry, which only the KW/Nauen pilots have."
    )
    st.caption(
        f"**Interval coverage (PICP) — nominal 80%.** Macro-CV realized: "
        f"wind {g.loc[('lightgbm', 'wind'), 'picp']:.0%}, solar {g.loc[('lightgbm', 'solar'), 'picp']:.0%}. "
        f"Municipal realized (KW+Nauen pilots): wind {muni['wind']:.0%}, solar {muni['solar']:.0%}. "
        "The 80% band under-covers locally — quantiles are trained on the smoother zonal signal, so "
        "treat the interval as indicative, not calibrated, at Gemeinde scale."
    )


def render(fc, subtitle, caps=None, aggregate=False):
    fc = fc.copy()
    fc.index = fc.index.tz_convert(LOCAL)  # plot + table in local time; sums are tz-agnostic
    st.caption(subtitle)
    k1, k2, k3, k4 = st.columns(4)
    total = fc['wind_median_mw'] + fc['solar_median_mw']
    k1.metric("Peak renewable (P50)", f"{total.max():,.1f} MW")
    k2.metric("Renewable energy (24 h)", f"{total.sum():,.0f} MWh")
    k3.metric("Peak wind / solar (P50)", f"{fc['wind_median_mw'].max():,.0f} / {fc['solar_median_mw'].max():,.0f} MW")
    k4.metric("Wind / solar energy (24 h)", f"{fc['wind_median_mw'].sum():,.0f} / {fc['solar_median_mw'].sum():,.0f} MWh")
    st.plotly_chart(stacked_figure(fc), width='stretch')
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
        show.index = show.index.strftime('%Y-%m-%d %H:%M %Z')
        st.dataframe(show[MW_COLS].round(2), width='stretch')


# ---- Sidebar ----------------------------------------------------------------
names = [n for n, _, _ in district_registry()]
st.sidebar.title("⚡ Forecast controls")
choice = st.sidebar.selectbox("Region", [AGG] + sorted(names))

parquet = load_state_parquet(os.path.getmtime(PARQUET) if os.path.exists(PARQUET) else 0)
issue_hour = pd.Timestamp.now(tz='UTC').floor('h')
issue_local = issue_hour.tz_convert(LOCAL)
window = local_day_window(LOCAL)
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
            dist_tot = parquet.groupby("district")[["wind_median_mw", "solar_median_mw"]].sum().sum(axis=1)
        else:
            frames, per_dist = [], {}
            prog = st.progress(0.0, "Running live state-wide forecast…")
            fleets = get_fleets()
            for i, (name, fleet) in enumerate(fleets.items(), 1):
                f, _, _ = forecast_district(name, fleet, pause=0.5, window=window)
                frames.append(f[MW_COLS])
                per_dist[name] = f["wind_median_mw"].sum() + f["solar_median_mw"].sum()
                prog.progress(i / len(fleets), f"{name} ({i}/18)")
            prog.empty()
            fc = pd.concat(frames).groupby(level=0).sum()
            dist_tot = pd.Series(per_dist)
        render(fc, f"All 18 Kreise summed · issued {issue_local:%Y-%m-%d %H:%M %Z} · {src_note}",
               aggregate=True)
        st.subheader("Renewable generation by Kreis — 24 h energy")
        st.plotly_chart(map_figure(dist_tot), width='stretch')
    else:
        if parquet is not None and choice in parquet.index.get_level_values("district"):
            fc = parquet.xs(choice, level="district")
            caps = {"wind": fc["wind_nameplate_mw"].iloc[0], "solar": fc["solar_nameplate_mw"].iloc[0]}
        else:
            fc, caps = live_district(choice, window)
        st.sidebar.markdown("**Installed capacity**")
        st.sidebar.metric("Wind", f"{caps['wind']:,.1f} MW")
        st.sidebar.metric("Solar", f"{caps['solar']:,.1f} MW")
        win = fc.index.tz_convert(LOCAL)
        render(fc, f"{choice} · issued {issue_local:%Y-%m-%d %H:%M %Z} · "
                   f"window {win.min():%a %H:%M} → {win.max():%a %H:%M %Z}", caps)
except Exception as e:
    st.error(f"Could not build forecast for {choice}: {e}")
    st.stop()

st.divider()
render_skill()

st.caption(
    "Median (P50) is the point forecast; the shaded band is the 80% prediction interval from the "
    "LightGBM quantile models. Intervals are indicative — quantiles are trained on the smoother "
    "50Hertz zonal signal, so local coverage runs below nominal (see PICP in the validation report)."
)
