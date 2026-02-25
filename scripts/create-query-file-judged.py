import pandas as pd


def load_qrels(qrels_path):
    qrels = pd.read_csv(
        qrels_path, sep=" ", names=["query_id", "q0", "doc_id", "label"]
    )
    return qrels


def main():
    queries = pd.read_csv("data/processed/queries-test.tsv", sep="\t")

    qrels_1_dctr = load_qrels("data/processed/qrels-snapshot-1-dctr.txt")
    qrels_2_dctr = load_qrels("data/processed/qrels-snapshot-2-dctr.txt")
    qrels_3_dctr = load_qrels("data/processed/qrels-snapshot-3-dctr.txt")
    
    judged_queries = pd.concat([qrels_1_dctr, qrels_2_dctr, qrels_3_dctr], ignore_index=True)
    judged_queries = queries[queries["query_id"].isin(judged_queries["query_id"].unique())]
    judged_queries[["query_id", "query"]].to_csv("data/processed/queries-judged.tsv", sep="\t", index=False)

if __name__ == "__main__":
    main()
