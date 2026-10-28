"""Plot overlap of judged queries across snapshots."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ir_datasets_longeval import load
from upsetplot import UpSet, from_memberships

OUTPUT = Path(__file__).resolve().parent / "output" / "query_upset.pdf"
SNAPSHOTS = ("snapshot-1", "snapshot-2", "snapshot-3")


def plot_query():
    query_sets = {}
    for snapshot in SNAPSHOTS:
        dataset = load(f"longeval-sci-2026/{snapshot}/dctr")
        query_sets[snapshot] = set(
            pd.DataFrame(dataset.qrels)["query_id"].dropna().astype(str)
        )

    memberships = []
    for query_id in set.union(*query_sets.values()):
        memberships.append(
            [snapshot for snapshot in SNAPSHOTS if query_id in query_sets[snapshot]]
        )
    plot = UpSet(
        from_memberships(memberships, data=pd.Series(1, index=range(len(memberships)))),
        subset_size="sum",
        sort_by="cardinality",
    )
    axes = plot.plot()
    intersections = axes["intersections"]
    intersections.bar_label(
        intersections.containers[0],
        labels=[f"{int(value):,}" for value in plot.intersections],
        padding=3,
    )
    plt.suptitle("Judged query overlap across snapshots")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT}")

def plot_docs():
    doc_sets = {}
    for snapshot in SNAPSHOTS:
        dataset = load(f"longeval-sci-2026/{snapshot}/dctr")
        doc_sets[snapshot] = set()
        for doc in dataset.docs_iter():
            doc_sets[snapshot].add(doc.doc_id)

    memberships = []
    for doc_id in set.union(*doc_sets.values()):
        memberships.append(
            [snapshot for snapshot in SNAPSHOTS if doc_id in doc_sets[snapshot]]
        )
    UpSet(
        from_memberships(memberships, data=pd.Series(1, index=range(len(memberships)))),
        subset_size="sum",
        sort_by="cardinality",
    ).plot()
    plt.suptitle("Judged document overlap across snapshots")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT.parent / "doc_upset.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT.parent / 'doc_upset.pdf'}")

if __name__ == "__main__":
    plot_query()
    plot_docs()
