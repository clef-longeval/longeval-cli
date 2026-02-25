import omegaconf
import pandas as pd
import psycopg2

cfg = omegaconf.OmegaConf.load("config.yaml")
conn = psycopg2.connect(**cfg.db)


def load_searches(snapshot, query_ids):
    query = f"""
    SELECT search_id, serp, queries.query_id, original_id, log_date
    FROM search_logs_normalized
    JOIN queries ON search_logs_normalized.query_id = queries.query_id
    WHERE log_date >= '{cfg["snapshots"][snapshot]["start_date"]}'
        AND log_date <= '{cfg["snapshots"][snapshot]["end_date"]}'
        AND is_structured is FALSE AND is_boolean is FALSE and has_doi is FALSE and was_clicked is TRUE
        AND queries.query_id IN {tuple(query_ids)};
    """
    searches = pd.read_sql_query(query, con=conn)
    searches = searches.drop_duplicates(subset=["search_id"])
    searches = searches.rename(columns={"log_date": "search_date"})
    return searches


def load_clicks(snapshot, searches):
    search_ids = list(searches["search_id"].unique())

    query = f"""
    SELECT *
    FROM tracker_logs
    WHERE search_id in {tuple(search_ids)}
        AND log_date >= '{cfg["snapshots"][snapshot]["start_date"]}'
        AND log_date <=  '{cfg["snapshots"][snapshot]["end_date"]}'
        AND (entry_type = 'get-pdf' OR entry_type = 'works')
    """
    clicks = pd.read_sql_query(query, con=conn)
    clicks = clicks.rename(columns={"log_date": "click_date"})

    searches_query_ids = (
        searches[["query_id", "search_id"]].set_index("search_id").to_dict()["query_id"]
    )
    clicks["query_id"] = clicks["search_id"].map(searches_query_ids)
    return clicks


def filter_clicks(clicks, k=30):
    counts = clicks.groupby("query_id")["click_date"].count()
    valid_queries = counts[counts >= k].index.tolist()
    return clicks[clicks["query_id"].isin(valid_queries)]


def create_impressions_table(searches):
    impressions = (
        searches[["query_id", "serp", "search_id"]]
        .explode("serp")
        .groupby(["query_id", "serp"])
        .count()
        .reset_index()
        .copy()
    )
    impressions.rename(
        columns={"search_id": "impressions", "serp": "doc_id"}, inplace=True
    )
    impressions["doc_id"] = impressions["doc_id"].astype(int)

    print(
        f"Loaded {len(impressions)} impressions for {len(impressions['query_id'].unique())} queries."
    )
    return impressions


def create_dctr_table(impressions, clicks):
    dctr = impressions.merge(clicks, on=["query_id", "doc_id"], how="left").copy()
    dctr["dctr"] = dctr["clicks"] / dctr["impressions"]
    dctr.fillna(0, inplace=True)
    return dctr


def create_clicks_table(clicks, valid_queries):
    clicks = clicks[clicks["query_id"].isin(valid_queries)]
    clicks = (
        clicks[["query_id", "doc_id", "search_id"]]
        .groupby(["query_id", "doc_id"])
        .count()
        .reset_index()
        .copy()
    )
    clicks = clicks.rename(columns={"search_id": "clicks"})
    return clicks


def discretize_labels(dctr):
    def _discretize(score):
        if score < 0.1:
            return 0
        elif score < 0.9:
            return 1
        else:
            return 2

    dctr["label"] = dctr["dctr"].apply(_discretize)
    return dctr


def create_qrels(dctr):
    valid_queries = dctr[dctr["label"] > 0]["query_id"].unique()
    dctr = dctr[dctr["query_id"].isin(valid_queries)]
    dctr["q0"] = "0"
    dctr = dctr[["query_id", "q0", "doc_id", "label"]]
    return dctr


def get_qrels(queries, snapshot):
    query_ids = queries["query_id"].unique().tolist()
    searches = load_searches(snapshot, query_ids)
    impressions = create_impressions_table(searches)

    clicks = load_clicks(snapshot, searches)
    filtered_clicks = filter_clicks(clicks)

    clicks_table = create_clicks_table(
        clicks, filtered_clicks["query_id"].unique().tolist()
    )
    dctr = create_dctr_table(impressions, clicks_table)
    dctr = discretize_labels(dctr)
    return dctr


def main():
    queries_task_1 = pd.read_csv(
        "data/interim/candidate_queries_task_1_filtered.tsv", sep="\t"
    )
    queries_task_3 = pd.read_csv("data/interim/candidate_queries_task_3.tsv", sep="\t")

    for snapshot in [1, 2, 3]:
        dctr_qrels_task_1 = get_qrels(queries_task_1, snapshot)
        dctr_qrels_task_3 = get_qrels(queries_task_3, snapshot)

        dctr_qrels = pd.concat(
            [dctr_qrels_task_1, dctr_qrels_task_3], ignore_index=True
        )
        dctr_qrels.drop_duplicates(inplace=True)

        valid_queries = dctr_qrels[dctr_qrels["label"] > 0]["query_id"].unique()
        dctr_qrels = dctr_qrels[dctr_qrels["query_id"].isin(valid_queries)]
        dctr_qrels["q0"] = "0"
        dctr_qrels = dctr_qrels[["query_id", "q0", "doc_id", "label"]]
        dctr_qrels.to_csv(
            f"data/interim/qrels-{snapshot}-dctr.tsv",
            sep=" ",
            index=False,
            header=False,
        )


if __name__ == "__main__":
    main()
