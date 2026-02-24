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

def get_qrels(queries, snapshot):
    query_ids = queries["query_id"].unique().tolist()
    searches = load_searches(snapshot, query_ids)
    clicks = load_clicks(snapshot, searches)
    raw_qrels = clicks[["query_id", "doc_id"]].copy()
    raw_qrels["q0"] = "0"
    raw_qrels["label"] = "1"
    raw_qrels.drop_duplicates(inplace=True)
    
    raw_qrels = raw_qrels[["query_id", "q0", "doc_id", "label"]]
    return raw_qrels



def main():
    queries_task_1 = pd.read_csv("data/interim/candidate_queries_task_1_filtered.tsv", sep="\t")
    queries_task_3 = pd.read_csv("data/interim/candidate_queries_task_3.tsv", sep="\t")

    for snapshot in [1, 2, 3]:
        raw_qrels_task_1 = get_qrels(queries_task_1, snapshot)
        raw_qrels_task_3 = get_qrels(queries_task_3, snapshot)
        raw_qrels = pd.concat([raw_qrels_task_1, raw_qrels_task_3], ignore_index=True)
        raw_qrels.drop_duplicates(inplace=True)
        raw_qrels.to_csv(f"data/interim/qrels-{snapshot}-raw.tsv", sep=" ", index=False, header=False)

if __name__ == "__main__":
    main()
