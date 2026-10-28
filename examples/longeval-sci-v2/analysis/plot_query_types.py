"""Query-type flags across all collected months."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
import pandas as pd
from matplotlib.ticker import PercentFormatter

OUTPUT = Path(__file__).resolve().parent / "output"


def main():
    data = pd.read_csv(
        OUTPUT / "monthly_queries.csv",
        usecols=["query_id", "searches", "is_structured", "is_boolean", "has_doi"],
    )
    distinct = data.drop_duplicates("query_id")
    types = {
        "Unstructured": ("is_structured", 0),
        "Structured": ("is_structured", 1),
        "Boolean": ("is_boolean", 1),
        "DOI": ("has_doi", 1),
    }
    counts = pd.DataFrame({
        name: {
            "Distinct queries": distinct[column].eq(value).sum(),
            "Searches": data.loc[data[column].eq(value), "searches"].sum(),
        }
        for name, (column, value) in types.items()
    }).T
    shares = counts.div(pd.Series({
        "Distinct queries": len(distinct),
        "Searches": data["searches"].sum(),
    }))
    print("Queries with missing labels:")
    print(distinct[["is_structured", "is_boolean", "has_doi"]].isna().sum())

    plt.rcParams.update({"font.size": 18, "pdf.fonttype": 42})
    ax = shares.plot.bar(
        figsize=(6, 6), color=["#4477AA", "#EEAA33"],
        width=0.75, rot=0,
    )
    colors = ["#4477AA", "#EEAA33", "#228833", "#AA3377"]
    for column_index, bars in enumerate(ax.containers):
        for bar, color in zip(bars, colors):
            if column_index == 0:
                color = tuple(0.5 * channel + 0.5 for channel in to_rgb(color))
            bar.set_facecolor(color)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_xticklabels(counts.index, rotation=25, ha="right")
    ax.set_ylabel("Share of total")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Query types")
    for bars in ax.containers:
        ax.bar_label(bars, labels=[f"{bar.get_height():.1%}" for bar in bars],
                     padding=3, fontsize=16)
    ax.legend(handles=[
        Patch(facecolor="#BBBBBB", label="Distinct queries"),
        Patch(facecolor="#666666", label="Searches"),
    ], frameon=False, fontsize=16)
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(OUTPUT / "query_types.pdf")
    plt.close(fig)
    print(counts.to_string())
    print(f"Saved {OUTPUT / 'query_types.pdf'}")


if __name__ == "__main__":
    main()
