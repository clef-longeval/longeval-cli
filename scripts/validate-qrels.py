import pandas as pd

def main():
    queries_task_1 = pd.read_csv(
        "data/interim/candidate_queries_task_1_filtered.tsv", sep="\t"
    )
    
    qrels_1 = pd.read_csv("data/interim/qrels-1-raw.tsv", sep=" ", names=["query_id", "q0", "doc_id", "label"])
    qrels_2 = pd.read_csv("data/interim/qrels-2-raw.tsv", sep=" ", names=["query_id", "q0", "doc_id", "label"])
    qrels_3 = pd.read_csv("data/interim/qrels-3-raw.tsv", sep=" ", names=["query_id", "q0", "doc_id", "label"])
    
    qrels_all = pd.concat([qrels_1, qrels_2, qrels_3], ignore_index=True)
    
    print(f"missing query ids in qrels: {set(queries_task_1['query_id'].unique()) - set(qrels_all['query_id'].unique())}")


    # each doc relevant for DCTR should be also in the raw qrels
    for snapshot in [1, 2, 3]:
        qrels_dctr = pd.read_csv(f"data/interim/qrels-{snapshot}-dctr.tsv", sep=" ", names=["query_id", "q0", "doc_id", "label"])
        qrels_dctr_unique = qrels_dctr[qrels_dctr["label"] > 0]["doc_id"].to_list()
        
        qrels_raw = pd.read_csv(f"data/interim/qrels-{snapshot}-raw.tsv", sep=" ", names=["query_id", "q0", "doc_id", "label"])
        
        missing_docs = set(qrels_dctr_unique) - set(qrels_raw["doc_id"].unique())
        assert missing_docs == set(), f"missing doc ids in raw qrels for snapshot {snapshot}: {missing_docs}"

if __name__ == "__main__":
    main()
