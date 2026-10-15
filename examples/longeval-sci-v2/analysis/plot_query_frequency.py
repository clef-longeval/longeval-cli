"""Log-log histogram of search frequency per distinct query, across all months."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT = Path(__file__).resolve().parent / "output"


def main():
    data = pd.read_csv(
        OUTPUT / "monthly_queries.csv", usecols=["query_id", "searches"]
    )
    frequency = data.groupby("query_id")["searches"].sum()
    # Integer-aligned logarithmic bins keep queries searched once separate.
    edges = np.unique(np.geomspace(1, frequency.max() + 1, 40).astype(int))
    if len(edges) < 2:
        edges = np.array([1, 2])
    plt.rcParams.update({"font.size": 18, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    ax.hist(frequency, bins=edges, color="#4477AA", edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Searches per query (log scale)")
    ax.set_ylabel("Queries per bin (log scale)")
    ax.set_title("Query search frequency")
    ax.text(
        0.97, 0.97,
        f"{len(frequency) / 1e6:.2f} million queries\n"
        f"{frequency.eq(1).mean():.1%} searched once",
        transform=ax.transAxes, ha="right", va="top",
    )
    fig.savefig(OUTPUT / "query_frequency.pdf")
    plt.close(fig)
    print(f"Saved {OUTPUT / 'query_frequency.pdf'}")


if __name__ == "__main__":
    main()
