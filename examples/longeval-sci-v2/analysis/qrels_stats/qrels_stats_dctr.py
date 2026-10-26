"""DCTR-specific statistics for LongEval Sci 2026, loaded through the ir_datasets_longeval library.


DCTR labels are graded (0, 1, 2) and label 0 means "judged, not relevant". So two query notions exist:
  - judged query   : has at least one row, with any label
  - relevant query : has at least one row with label > 0   (the ones that can actually be scored)

Builds three pandas DataFrames and saves them as CSV:
  1. summary     one row per snapshot
  2. overlap     overlap across snapshots, for both notions (judged / relevant)
  3. per_query   one row per judged query: rows and relevant rows in each snapshot

Usage:
    python qrels_stats_dctr_from_library.py
    python qrels_stats_dctr_from_library.py --no-queries --out-dir results/
"""
import argparse
import os

import pandas as pd
from ir_datasets_longeval import load
from ir_datasets_longeval.datasets import longeval_sci_2026

SNAPSHOTS = {1: "03-05", 2: "06-08", 3: "09-11"}
TAG = "longeval-sci-2026/snapshot-{}/dctr"


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


def regions(sets):
    """Overlap regions for three sets of query IDs."""
    a, b, c = sets[1], sets[2], sets[3]
    union = a | b | c
    one = {q for q in union if (q in a) + (q in b) + (q in c) == 1}
    two = {q for q in union if (q in a) + (q in b) + (q in c) == 2}
    return {
        "snapshot 1": len(a), "snapshot 2": len(b), "snapshot 3": len(c),
        "S1 & S2 (any overlap)": len(a & b), "S1 & S3 (any overlap)": len(a & c), "S2 & S3 (any overlap)": len(b & c),
        "all three": len(a & b & c),
        "exactly two snapshots": len(two),
        "exactly one snapshot": len(one),
        "union (distinct queries)": len(union),
    }


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
        # all three dctr tags share the same test queries file
        texts = {q.query_id: q.text for q in datasets[1].queries_iter()}
        all_queries = set(texts)

    # 1. summary
    rows = []
    for s, df in qrels.items():
        rel = df[df["relevance"] > 0]
        per_q_all = df.groupby("query_id").size()
        per_q_rel = rel.groupby("query_id").size()
        row = {
            "snapshot": s,
            "period": SNAPSHOTS[s],
            "tag": TAG.format(s),
            "judged_queries": df["query_id"].nunique(),
            "queries_with_relevant": rel["query_id"].nunique(),
            "queries_all_zero": df["query_id"].nunique() - rel["query_id"].nunique(),
            "qrels_rows": len(df),
            "label_0": int((df["relevance"] == 0).sum()),
            "label_1": int((df["relevance"] == 1).sum()),
            "label_2": int((df["relevance"] == 2).sum()),
            "relevant_rows (label>0)": len(rel),
            "relevant_share_%": round(100 * len(rel) / len(df), 1),
            "unique_docs": df["doc_id"].nunique(),
            "duplicate_pairs": int(df.duplicated(["query_id", "doc_id"]).sum()),
            "rows_per_query_mean": round(per_q_all.mean(), 2),
            "rows_per_query_median": float(per_q_all.median()),
            "rows_per_query_max": int(per_q_all.max()),
            "relevant_per_query_mean": round(per_q_rel.mean(), 2),
            "relevant_per_query_median": float(per_q_rel.median()),
            "relevant_per_query_max": int(per_q_rel.max()),
        }
        other = sorted(set(df["relevance"].unique()) - {0, 1, 2})
        if other:
            row["other_labels"] = str(other)
        if all_queries is not None:
            n = len(all_queries)
            row["total_queries"] = n
            row["judged_share_%"] = round(100 * row["judged_queries"] / n, 1)
            row["relevant_query_share_%"] = round(100 * row["queries_with_relevant"] / n, 1)
            row["judged_not_in_queries_file"] = len(set(df["query_id"]) - all_queries)
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("snapshot")

    # 2. overlap, two definitions
    judged = {s: set(df["query_id"]) for s, df in qrels.items()}
    relevant = {s: set(df.loc[df["relevance"] > 0, "query_id"]) for s, df in qrels.items()}
    ov_j, ov_r = regions(judged), regions(relevant)
    overlap = pd.DataFrame({"judged (any label)": ov_j, "with >=1 relevant (label>0)": ov_r})
    overlap.index.name = "region"
    if all_queries is not None:
        overlap.loc["test queries never judged / without relevant"] = [
            len(all_queries - set().union(*judged.values())),
            len(all_queries - set().union(*relevant.values())),
        ]

    # 3. per-query table
    union = sorted(set().union(*judged.values()))
    per_query = pd.DataFrame(index=pd.Index(union, name="query_id"))
    for s, df in qrels.items():
        per_query[f"rows_S{s}"] = df.groupby("query_id").size().reindex(union).fillna(0).astype(int)
        per_query[f"relevant_S{s}"] = (df[df["relevance"] > 0].groupby("query_id").size()
                                       .reindex(union).fillna(0).astype(int))
    per_query["n_snapshots_judged"] = (per_query[[f"rows_S{s}" for s in qrels]] > 0).sum(axis=1)
    per_query["n_snapshots_relevant"] = (per_query[[f"relevant_S{s}" for s in qrels]] > 0).sum(axis=1)
    if texts:
        per_query.insert(0, "text", [texts.get(qid, "") for qid in per_query.index])
    per_query = per_query.sort_values(["n_snapshots_relevant", "n_snapshots_judged"], ascending=False)

    os.makedirs(args.out_dir, exist_ok=True)
    summary.to_csv(os.path.join(args.out_dir, "summary_dctr.csv"))
    overlap.to_csv(os.path.join(args.out_dir, "overlap_dctr.csv"))
    per_query.to_csv(os.path.join(args.out_dir, "per_query_dctr.csv"))

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print("\n== Summary per snapshot (DCTR, via ir_datasets_longeval) ==")
    print(summary.T)
    print("\n== Overlap across snapshots ==")
    print(overlap)
    print("\n== Per-query table (top 10) ==")
    print(per_query.head(10))
    print(f"\nCSVs written to {args.out_dir}/")


if __name__ == "__main__":
    main()
