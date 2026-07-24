"""Static PNG export of the state-wide choropleth (interactive version lives in
dashboard.py's map_figure). Same data source and colorscale, rendered with
matplotlib/geopandas instead of plotly so it needs no browser/kaleido and can
be dropped straight into the thesis.

Reads the pre-computed batch forecast (uv run jupyter nbconvert ... notebook 11)
and the cached district boundaries (src/data/districts.py) -- no network, no
model inference.

Output:
    results/choropleth_state_forecast.png
"""

import os

import pandas as pd

from src.data.districts import load_district_boundaries

FORECAST_PARQUET = "results/forecasts/brandenburg_state_forecast.parquet"
PNG_OUT = "results/choropleth_state_forecast.png"


def _district_totals() -> pd.Series:
    """24h total renewable energy (MWh) per district, wind + solar median."""
    fc = pd.read_parquet(FORECAST_PARQUET)
    mwh = fc["wind_median_mw"] + fc["solar_median_mw"]
    return mwh.groupby(level="district").sum()


def run() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.titleweight": "bold",
    })

    totals = _district_totals()
    gdf = load_district_boundaries().merge(totals.rename("mwh"), left_on="name", right_index=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    gdf.plot(column="mwh", cmap="YlGnBu", edgecolor="#64748b", linewidth=0.5,
             legend=True, legend_kwds={"label": "24h renewable generation (MWh)", "shrink": 0.75}, ax=ax)
    for _, row in gdf.iterrows():
        ax.annotate(row["name"], (row.geometry.representative_point().x, row.geometry.representative_point().y),
                   fontsize=6.5, ha="center", va="center", color="#1e293b")
    ax.set_title("Brandenburg — 24h Renewable Generation Forecast by District")
    ax.set_axis_off()
    fig.tight_layout()

    os.makedirs("results", exist_ok=True)
    fig.savefig(PNG_OUT)
    plt.close(fig)
    print(f"Wrote {PNG_OUT}")


def demo() -> None:
    """Self-check: every district has a positive forecast total and the boundary
    join drops nothing (18 districts in, 18 rows out)."""
    totals = _district_totals()
    assert len(totals) == 18, f"expected 18 districts, got {len(totals)}"
    assert (totals > 0).all(), totals[totals <= 0]
    gdf = load_district_boundaries().merge(totals.rename("mwh"), left_on="name", right_index=True)
    assert len(gdf) == 18, f"boundary join dropped districts: {len(gdf)}"
    print("export_choropleth self-check passed: 18/18 districts joined, all totals positive")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    args = p.parse_args()
    if args.selfcheck:
        demo()
    else:
        run()
