import pandas as pd
from omegaconf import DictConfig
import omegaconf
import psycopg2

cfg = omegaconf.OmegaConf.load("config.yaml")
conn = psycopg2.connect(**cfg.db)

# Get searches
def load_searches(snapshot):
    if snapshot == "test":
        snapshot = 1
    query = f"""
    SELECT search_id, serp, queries.query_id, original_id, log_date
    FROM search_logs_normalized
    JOIN queries ON search_logs_normalized.query_id = queries.query_id
    WHERE log_date >= '{cfg["snapshots"][snapshot]["start_date"]}'
        AND log_date <= '{cfg["snapshots"][snapshot]["end_date"]}'
        AND is_structured is FALSE AND is_boolean is FALSE and has_doi is FALSE and was_clicked is TRUE;
    """
    searches = pd.read_sql_query(query, con=conn)
    searches = searches.drop_duplicates(subset=["search_id"])
    searches = searches.rename(columns={"log_date": "search_date"})
    return searches

def load_clicks(snapshot, searches):
    if snapshot == "test":
        snapshot = 1
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

def filter_candidate_queries(clicks, searches):
    clicks = clicks.merge(
        searches[["search_id", "search_date"]],
        left_on="search_id",
        right_on="search_id",
        how="left",
    )
    print("Clicks before 24h filter:", len(clicks))
    # Only keep clicks that happened within 1 day of the search
    mask = (clicks["click_date"] >= clicks["search_date"]) & (clicks["click_date"] <= clicks["search_date"] + pd.Timedelta(days=1))
    # mask = clicks["click_date"] <= clicks["search_date"] + pd.Timedelta(days=1)
    clicks = clicks[mask]

    print("Clicks after 24h filter:", len(clicks))

    click_stats = (
        clicks.groupby("query_id").count().reset_index()[["query_id", "search_id"]]
    )
    candidate_queries = click_stats[click_stats["search_id"] > 30]
    print("Candidate queries with at least 30 clicks:", len(candidate_queries))
    return candidate_queries

def load_queries(query_ids):
    query = f"""
    SELECT query_text AS query, query_id
    FROM queries
    WHERE query_id IN {tuple(query_ids)}
    """
    return pd.read_sql(query, conn)


def main():
    candidates = []
    for snapshot in [1, 2, 3]:
        searches = load_searches(snapshot)
        
        clicks = load_clicks(snapshot, searches)
        candidate_queries = filter_candidate_queries(clicks, searches)
        candidate_queries.rename(
            columns={"search_id": f"clicks-snapshot-{snapshot}"}, inplace=True
        )
        candidates.append(candidate_queries)
    
    # merge
    candidates = (
        candidates[0]
        .merge(candidates[1], on="query_id", how="outer")
        .merge(candidates[2], on="query_id", how="outer")
    )
    candidates[["query_id"]].to_csv("data/interim/candidate_queries_task_1.csv", index=False)
    
    queries = load_queries(candidates["query_id"].dropna().astype(int))
    
    candidates = candidates.merge(queries, on="query_id", how="left")
    
    candidates["session_id"] = ""
    candidates["turn"] = 1
    candidates[["query_id", "query", "session_id", "turn"]].to_csv("data/interim/candidate_queries_task_1.tsv", sep="\t", index=False)
    

if __name__ == "__main__":
    main()
