import hashlib

import pandas as pd


def hash_query_id(query_id):
    return hashlib.md5(str(query_id).encode()).hexdigest()


def sample_train_queries():
    queries_task_1 = pd.read_csv(
        "data/interim/candidate_queries_task_1_filtered.tsv", sep="\t"
    )

    qrels_1 = pd.read_csv(
        "data/interim/qrels-1-dctr.tsv",
        sep=" ",
        names=["query_id", "q0", "doc_id", "label"],
    )
    qrels_1 = qrels_1.groupby("query_id").size()

    qrels_2 = pd.read_csv(
        "data/interim/qrels-2-dctr.tsv",
        sep=" ",
        names=["query_id", "q0", "doc_id", "label"],
    )
    qrels_2 = qrels_2.groupby("query_id").size()

    qrels_3 = pd.read_csv(
        "data/interim/qrels-3-dctr.tsv",
        sep=" ",
        names=["query_id", "q0", "doc_id", "label"],
    )
    qrels_3 = qrels_3.groupby("query_id").size()

    qrels_all = pd.merge(
        qrels_1.to_frame(name="qrels_1"),
        qrels_2.to_frame(name="qrels_2"),
        left_index=True,
        right_index=True,
        how="outer",
    )
    qrels_all = pd.merge(
        qrels_all,
        qrels_3.to_frame(name="qrels_3"),
        left_index=True,
        right_index=True,
        how="outer",
    )

    train_queries = qrels_all[
        (qrels_all["qrels_2"].isna()) & (qrels_all["qrels_3"].isna())
    ]

    task_3_queries = pd.read_csv("data/interim/candidate_queries_task_3.tsv", sep="\t")
    train_queries = train_queries[~train_queries.index.isin(task_3_queries["query_id"])]

    train_queries = train_queries.sample(n=100, random_state=42).index.tolist()

    return train_queries


def load_qrels(qrels_path):
    qrels = pd.read_csv(
        qrels_path, sep=" ", names=["query_id", "q0", "doc_id", "label"]
    )
    return qrels


def save_qrels(qrels, output_path):
    qrels["query_id"] = qrels["query_id"].apply(hash_query_id)
    qrels.to_csv(output_path, sep=" ", index=False, header=False)


def main():
    # split
    train_query_ids = sample_train_queries()

    queries_task_1 = pd.read_csv(
        "data/interim/candidate_queries_task_1_filtered.tsv", sep="\t"
    )
    queries_task_3 = pd.read_csv("data/interim/candidate_queries_task_3.tsv", sep="\t")

    queries = pd.concat([queries_task_1, queries_task_3], ignore_index=True)
    queries = queries.drop_duplicates(subset=["query_id"])

    train_queries = queries[queries["query_id"].isin(train_query_ids)]

    test_queries = queries[~queries["query_id"].isin(train_query_ids)]
    queries_task_3_new = pd.read_csv(
        "data/interim/candidate_queries_task_3_new.tsv", sep="\t"
    )
    test_queries = pd.concat([test_queries, queries_task_3_new], ignore_index=True)
    test_queries = test_queries.drop_duplicates(subset=["query_id"])
    test_queries = test_queries.sample(frac=1, random_state=42).reset_index(drop=True)

    # qrels
    qrels_1_dctr = load_qrels("data/interim/qrels-1-dctr.tsv")
    qrels_2_dctr = load_qrels("data/interim/qrels-2-dctr.tsv")
    qrels_3_dctr = load_qrels("data/interim/qrels-3-dctr.tsv")

    qrels_1_raw = load_qrels("data/interim/qrels-1-raw.tsv")
    qrels_2_raw = load_qrels("data/interim/qrels-2-raw.tsv")
    qrels_3_raw = load_qrels("data/interim/qrels-3-raw.tsv")

    qrels_1_dctr_train = qrels_1_dctr[
        qrels_1_dctr["query_id"].isin(train_queries["query_id"])
    ]
    qrels_1_raw_train = qrels_1_raw[
        qrels_1_raw["query_id"].isin(train_queries["query_id"])
    ]
    save_qrels(qrels_1_dctr_train, "data/processed/qrels-snapshot-train-dctr.txt")
    save_qrels(qrels_1_raw_train, "data/processed/qrels-snapshot-train-raw.txt")

    qrels_1_dctr_test = qrels_1_dctr[
        qrels_1_dctr["query_id"].isin(test_queries["query_id"])
    ]
    qrels_1_raw_test = qrels_1_raw[
        qrels_1_raw["query_id"].isin(test_queries["query_id"])
    ]
    save_qrels(qrels_1_dctr_test, "data/processed/qrels-snapshot-1-dctr.txt")
    save_qrels(qrels_1_raw_test, "data/processed/qrels-snapshot-1-raw.txt")

    save_qrels(qrels_2_dctr, "data/processed/qrels-snapshot-2-dctr.txt")
    save_qrels(qrels_3_dctr, "data/processed/qrels-snapshot-3-dctr.txt")

    save_qrels(qrels_2_raw, "data/processed/qrels-snapshot-2-raw.txt")
    save_qrels(qrels_3_raw, "data/processed/qrels-snapshot-3-raw.txt")

    # queries
    train_queries["query_id"] = train_queries["query_id"].apply(hash_query_id)
    train_queries[["query_id", "query"]].to_csv("data/processed/queries-train.tsv", sep="\t", index=False, header=False)

    test_queries["query_id"] = test_queries["query_id"].apply(hash_query_id)
    test_queries[["query_id", "query"]].to_csv("data/processed/queries-test.tsv", sep="\t", index=False, header=False)


if __name__ == "__main__":
    main()
