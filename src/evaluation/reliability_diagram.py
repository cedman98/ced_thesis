"""LightGBM 80% prediction-interval reliability diagram: nominal coverage (80%)
vs realized PICP, macro (purged CV) vs municipal (KW/Nauen telemetry).

Reads the already-computed metrics (no retraining):
    results/macro_cv_metrics.csv        -> per-fold PICP, model==lightgbm
    results/municipal_validation_metrics.csv -> per-municipality PICP, model==lightgbm

Output:
    results/reliability_diagram.png
"""

import os

import pandas as pd

MACRO_CSV = "results/macro_cv_metrics.csv"
MUNI_CSV = "results/municipal_validation_metrics.csv"
PNG_OUT = "results/reliability_diagram.png"
NOMINAL = 0.80


def _macro_picp(tech: str) -> pd.Series:
    m = pd.read_csv(MACRO_CSV)
    m = m[(m.model == "lightgbm") & (m.scale == "cf") & (m.technology == tech)]
    return m.set_index("fold")["picp"].sort_index()


def _muni_picp(tech: str) -> pd.Series:
    v = pd.read_csv(MUNI_CSV)
    v = v[(v.model == "lightgbm") & (v.technology == tech)]
    return v.set_index("municipality")["picp"]


def run() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.3,
    })
    macro_c, muni_c = sns.color_palette("colorblind", 2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    for ax, tech in zip(axes, ("wind", "solar")):
        macro = _macro_picp(tech)
        muni = _muni_picp(tech)

        labels = ["Macro CV\n(mean of folds)"] + list(muni.index)
        vals = [macro.mean()] + list(muni.values)
        colors = [macro_c] + [muni_c] * len(muni)
        yerr = [[macro.mean() - macro.min()], [macro.max() - macro.mean()]]
        yerr = [yerr[0] + [0] * len(muni), yerr[1] + [0] * len(muni)]

        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6,
                      yerr=yerr, capsize=4, error_kw={"linewidth": 1})
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)
        ax.axhline(NOMINAL, color="black", linestyle="--", linewidth=1.2,
                  label=f"Nominal {NOMINAL:.0%}")
        ax.set_title(f"{tech.title()} — 80% interval coverage")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(loc="lower left", fontsize=9)
    axes[0].set_ylabel("PICP (realized coverage)")
    fig.suptitle("PICP Coverage Bars (Realised vs 80% Nominal, Macro vs Municipal)", y=1.02)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig(PNG_OUT)
    plt.close(fig)
    print(f"Wrote {PNG_OUT}")


def demo() -> None:
    """Self-check: macro CV must cover both technologies' folds (data present),
    and municipal PICP must be strictly below nominal (the known undercoverage)."""
    for tech in ("wind", "solar"):
        macro = _macro_picp(tech)
        muni = _muni_picp(tech)
        assert len(macro) > 0 and len(muni) > 0, tech
        assert (muni < NOMINAL).all(), f"{tech}: expected municipal undercoverage, got {muni.to_dict()}"
    print("reliability_diagram self-check passed: municipal PICP < nominal 80% for wind and solar")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    args = p.parse_args()
    if args.selfcheck:
        demo()
    else:
        run()
