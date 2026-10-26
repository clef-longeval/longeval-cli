"""RAW statistics for LongEval Sci 2026, loaded through the ir_datasets_longeval library.


Usage:
    python qrels_stats_raw_from_library.py
    python qrels_stats_raw_from_library.py --no-queries --out-dir results/
"""
import argparse
import os

import pandas as pd
from ir_datasets_longeval import load
from ir_datasets_longeval.datasets import longeval_sci_2026

SNAPSHOTS = {1: "03-05", 2: "06-08", 3: "09-11"}
TAG = "longeval-sci-2026/snapshot-{}/raw"


def patch_unreleased_topics():
    cls = longeval_sci_2026.LongEvalSciDataset
    original = cls.__init__

    def init(self, *a, **kw):
        try:
            original(self, *a, **kw)
        except FileNotFoundError:
            kw["queries_path"] = None
            original(self, *a, **kw)

    cls.__init__ = init


def read_qrels(ds):
    return pd.DataFrame([(q.query_id, q.doc_id, q.relevance) for q in ds.qrels_iter()],
                        columns=["query_id", "doc_id", "relevance"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-queries", action="store_true", help="skip test queries (coverage and query text)")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    patch_unreleased_topics()
    datasets = {s: load(TAG.format(s)) for s in SNAPSHOTS}
    qrels = {s: read_qrels(ds) for s, ds in datasets.items()}

    all_queries, texts = None, {}
    if not args.no_queries:
        # all three raw tags share the same test queries file
        texts = {q.query_id: q.text for q in datasets[1].queries_iter()}
        all_queries = set(texts)


    rows = []
    for s, df in qrels.items():
        per_q = df.groupby("query_id").size()
        row = {
            "snapshot": s,
            "period": SNAPSHOTS[s],
            "tag": TAG.format(s),
            "judged_queries": df["query_id"].nunique(),
            "qrels_rows": len(df),
            "unique_docs": df["doc_id"].nunique(),
            "duplicate_pairs": int(df.duplicated(["query_id", "doc_id"]).sum()),
            "qrels_per_query_mean": round(per_q.mean(), 2),
            "qrels_per_query_median": float(per_q.median()),
            "qrels_per_query_max": int(per_q.max()),
        }
        for lab, n in df["relevance"].value_counts().sort_index().items():
            row[f"label_{lab}"] = int(n)
        if all_queries is not None:
            row["total_queries"] = len(all_queries)
            row["judged_share_%"] = round(100 * row["judged_queries"] / len(all_queries), 1)
            row["judged_not_in_queries_file"] = len(set(df["query_id"]) - all_queries)
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("snapshot")

  
    sets = {s: set(df["query_id"]) for s, df in qrels.items()}
    union = set().union(*sets.values())
    membership = pd.DataFrame({f"S{s}": [qid in sets[s] for qid in sorted(union)] for s in sets}, index=sorted(union))
    n_snap = membership.sum(axis=1)
    overlap_rows = [
        ("S1 only", int((membership["S1"] & ~membership["S2"] & ~membership["S3"]).sum())),
        ("S2 only", int((~membership["S1"] & membership["S2"] & ~membership["S3"]).sum())),
        ("S3 only", int((~membership["S1"] & ~membership["S2"] & membership["S3"]).sum())),
        ("S1 & S2 only", int((membership["S1"] & membership["S2"] & ~membership["S3"]).sum())),
        ("S1 & S3 only", int((membership["S1"] & ~membership["S2"] & membership["S3"]).sum())),
        ("S2 & S3 only", int((~membership["S1"] & membership["S2"] & membership["S3"]).sum())),
        ("all three", int((n_snap == 3).sum())),
        ("exactly one snapshot", int((n_snap == 1).sum())),
        ("exactly two snapshots", int((n_snap == 2).sum())),
        ("union (distinct judged queries)", len(union)),
        ("S1 total / S1&S2 total", f"{len(sets[1])} / {len(sets[1] & sets[2])}"),
        ("S1&S3 total / S2&S3 total", f"{len(sets[1] & sets[3])} / {len(sets[2] & sets[3])}"),
    ]
    if all_queries is not None:
        overlap_rows.append(("test queries never judged", len(all_queries - union)))
    overlap = pd.DataFrame(overlap_rows, columns=["region", "queries"]).set_index("region")

 
    counts = pd.DataFrame({f"qrels_S{s}": df.groupby("query_id").size() for s, df in qrels.items()}).reindex(sorted(union)).fillna(0).astype(int)
    counts["n_snapshots_judged"] = (counts[[f"qrels_S{s}" for s in sets]] > 0).sum(axis=1)
    if texts:
        counts.insert(0, "text", [texts.get(qid, "") for qid in counts.index])
    counts.index.name = "query_id"
    per_query = counts.sort_values(["n_snapshots_judged", "qrels_S1", "qrels_S2", "qrels_S3"], ascending=False)

    os.makedirs(args.out_dir, exist_ok=True)
    summary.to_csv(os.path.join(args.out_dir, "summary_raw.csv"))
    overlap.to_csv(os.path.join(args.out_dir, "overlap_raw.csv"))
    per_query.to_csv(os.path.join(args.out_dir, "per_query_raw.csv"))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n== Summary per snapshot (raw, via ir_datasets_longeval) ==")
    print(summary.T)
    print("\n== Overlap of judged queries ==")
    print(overlap)
    print("\n== Per-query table (top 10) ==")
    print(per_query.head(10))
    print(f"\nCSVs written to {args.out_dir}/")


if __name__ == "__main__":
    main()
