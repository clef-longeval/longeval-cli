"""Number of observed months per query, restricted to recurring queries."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT = Path(__file__).resolve().parent / "output"


def main():
    data = pd.read_csv(
        OUTPUT / "monthly_queries.csv", usecols=["month", "query_id"]
    ).dropna()
    months_per_query = data.groupby("query_id")["month"].nunique()
    counts = months_per_query[months_per_query > 1].value_counts().sort_index()
    counts = counts.reindex(range(2, data["month"].nunique() + 1), fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.bar(counts.index, counts, color="#4477AA")
    ax.set_yscale("log")
    ax.set_xticks(counts.index)
    ax.set_xlabel("Number of observed months per query")
    ax.set_ylabel("Distinct queries (log scale)")
    ax.set_title("Temporal recurrence of queries")
    ax.text(
        0.97, 0.97,
        f"{counts.sum():,} queries observed in at least two months\n"
        f"{months_per_query.eq(1).mean():.1%} observed in only one month (excluded)",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
    )
    fig.savefig(OUTPUT / "query_months.pdf")
    plt.close(fig)
    print(counts.to_string())
    print(f"Saved {OUTPUT / 'query_months.pdf'}")


if __name__ == "__main__":
    main()
