"""Aligned frequency and query-type panels for the paper."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

OUTPUT = Path(__file__).resolve().parent / "output"


def main():
    data = pd.read_csv(OUTPUT / "monthly_queries.csv", usecols=[
        "query_id", "searches", "is_structured", "is_boolean", "has_doi",
    ])
    frequency = data.groupby("query_id")["searches"].sum()
    distinct = data.drop_duplicates("query_id")
    types = {
        "Unstructured": ("is_structured", 0),
        "Boolean": ("is_boolean", 1),
        "DOI": ("has_doi", 1),
    }
    shares = pd.DataFrame({
        name: {
            "Distinct queries": distinct[column].eq(value).mean(),
            "Searches": data.loc[data[column].eq(value), "searches"].sum()
                        / data["searches"].sum(),
        }
        for name, (column, value) in types.items()
    }).T

    plt.rcParams.update({"font.size": 18, "pdf.fonttype": 42})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 6))
    # Fixed margins give both panels identical plotting areas and baselines.
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.23, top=0.9, wspace=0.3)
    edges = np.unique(np.geomspace(1, frequency.max() + 1, 40).astype(int))
    left.hist(frequency, bins=edges, color="#4477AA", edgecolor="white", linewidth=0.5)
    left.set(xscale="log", yscale="log", xlabel="Searches per query",
             ylabel="Queries per bin", title="(a) Query search frequency")
    left.text(0.97, 0.97,
              f"{len(frequency) / 1e6:.2f} million queries\n"
              f"{frequency.eq(1).mean():.1%} searched once",
              transform=left.transAxes, ha="right", va="top", fontsize=16)

    shares.plot.bar(ax=right, width=0.75, rot=25, legend=False)
    for index, bars in enumerate(right.containers):
        for bar, color in zip(bars, ["#4477AA", "#228833", "#AA3377"]):
            if index == 0:
                color = tuple(0.5 * channel + 0.5 for channel in to_rgb(color))
            bar.set_facecolor(color)
        right.bar_label(bars, labels=[f"{bar.get_height():.1%}" for bar in bars],
                        padding=3, fontsize=14)
    right.set(ylim=(0, 1), xlabel="", ylabel="Share of total", title="(b) Query types")
    right.set_xticklabels(shares.index, rotation=25, ha="right")
    right.yaxis.set_major_formatter(PercentFormatter(1))
    right.legend(handles=[
        Patch(facecolor="#BBBBBB", label="Distinct queries"),
        Patch(facecolor="#666666", label="Searches"),
    ], frameon=False, fontsize=14, loc="upper right")
    fig.savefig(OUTPUT / "query_distribution.pdf")
    plt.close(fig)
    print(f"Saved {OUTPUT / 'query_distribution.pdf'}")


if __name__ == "__main__":
    main()
