"""Plot distinct-query overlap between months from the collected CSV."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT = Path(__file__).resolve().parent / "output"


def main():
    data = pd.read_csv(OUTPUT / "monthly_queries.csv", usecols=["month", "query_id", "is_structured"])
    data = data[data["is_structured"].eq(0)]
    data = data.dropna(subset=["month", "query_id"]).drop_duplicates(["month", "query_id"])
    months = sorted(data["month"].unique())
    # Keep queries observed in at least three months, regardless of search frequency.
    recurring = data.groupby("query_id")["month"].transform("count") > 2
    data = data[recurring]
    queries = data.groupby("month")["query_id"].agg(set).reindex(months)
    queries = queries.map(lambda value: value if isinstance(value, set) else set())
    shared = pd.DataFrame(
        [[len(a & b) for b in queries] for a in queries],
        index=queries.index, columns=queries.index,
    )
    sizes = queries.map(len)
    union = shared.mul(-1).add(sizes, axis=0).add(sizes, axis=1)
    jaccard = shared / union.replace(0, float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, matrix, title in zip(
        axes, [shared, jaccard], ["Shared recurring queries", "Jaccard overlap"]
    ):
        image = ax.imshow(matrix, cmap="Blues", vmin=0,
                          vmax=1 if title == "Jaccard overlap" else None)
        ax.set_xticks(range(len(matrix)), matrix.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(matrix)), matrix.index)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle("Unstructured queries observed in at least three months")
    fig.savefig(OUTPUT / "monthly_query_overlap.pdf")
    plt.close(fig)
    print(f"Saved PDF plot to {OUTPUT}")


if __name__ == "__main__":
    main()
