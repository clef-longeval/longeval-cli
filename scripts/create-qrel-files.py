import hashlib

import pandas as pd


def hash_query_id(query_id):
    return hashlib.md5(str(query_id).encode()).hexdigest()


def load_qrels(qrels_path):
    qrels = pd.read_csv(qrels_path, sep=" ", names=["query_id", "q0", "doc_id", "label"])
    return qrels

def save_qrels(qrels, output_path):
    qrels["query_id"] = qrels["query_id"].apply(hash_query_id)
    qrels.to_csv(output_path, sep=" ", index=False, header=False)

def main():
    # queries
    train_queries = pd.read_csv("data/processed/queries-train.tsv", sep="\t")
    test_queries = pd.read_csv("data/processed/queries-test.tsv", sep="\t")

    # qrels
    qrels_1_dctr = load_qrels("data/interim/qrels-1-dctr.tsv")
    qrels_2_dctr = load_qrels("data/interim/qrels-2-dctr.tsv")
    qrels_3_dctr = load_qrels("data/interim/qrels-3-dctr.tsv")
    
    qrels_1_raw = load_qrels("data/interim/qrels-1-raw.tsv")
    qrels_2_raw = load_qrels("data/interim/qrels-2-raw.tsv")
    qrels_3_raw = load_qrels("data/interim/qrels-3-raw.tsv")



    qrels_1_dctr_train = qrels_1_dctr[qrels_1_dctr["query_id"].isin(train_queries["query_id"])]
    qrels_1_raw_train = qrels_1_raw[qrels_1_raw["query_id"].isin(train_queries["query_id"])]
    save_qrels(qrels_1_dctr_train, "data/processed/qrels-snapshot-train-dctr.txt")
    save_qrels(qrels_1_raw_train, "data/processed/qrels-snapshot-train-raw.txt")
    
    qrels_1_dctr_test = qrels_1_dctr[qrels_1_dctr["query_id"].isin(test_queries["query_id"])]
    qrels_1_raw_test = qrels_1_raw[qrels_1_raw["query_id"].isin(test_queries["query_id"])]
    save_qrels(qrels_1_dctr_test, "data/processed/qrels-snapshot-1-dctr.txt")
    save_qrels(qrels_1_raw_test, "data/processed/qrels-snapshot-1-raw.txt")

    save_qrels(qrels_2_dctr, "data/processed/qrels-snapshot-2-dctr.txt")
    save_qrels(qrels_3_dctr, "data/processed/qrels-snapshot-3-dctr.txt")
    
    save_qrels(qrels_2_raw, "data/processed/qrels-snapshot-2-raw.txt")
    save_qrels(qrels_3_raw, "data/processed/qrels-snapshot-3-raw.txt")
    
    
if __name__ == "__main__":
    main()
