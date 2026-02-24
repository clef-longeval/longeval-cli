import hashlib

import omegaconf
import pandas as pd
import psycopg2

cfg = omegaconf.OmegaConf.load("config.yaml")
conn = psycopg2.connect(**cfg.db)


def hash_query_id(query_id):
    return hashlib.md5(str(query_id).encode()).hexdigest()


def main():
    task_3_queries = pd.read_csv("data/raw/task-3/public_logs.csv")
    
    query = f"""
    SELECT search_id, serp, queries.query_id, original_id, log_date, query_text
    FROM search_logs_normalized
    JOIN queries ON search_logs_normalized.query_id = queries.query_id
    WHERE search_id IN {tuple(task_3_queries["sid"].to_list())};
    """
    searches = pd.read_sql_query(query, con=conn)
    searches.drop_duplicates(subset=["search_id"], inplace=True)
    
    print(f"Total queries: {len(task_3_queries)}")
    
    print(f"Total searches in DB: {len(searches)}")

    # filter search_ids not in DB
    task_3_queries = task_3_queries[~task_3_queries["sid"].isin(searches["search_id"])]
    print(f"Queries after filtering search_ids not in DB: {len(task_3_queries)}")
    
    
    task_3_queries["query_id"] = task_3_queries["session_id"].apply(hash_query_id)

    task_3_queries[["x", "session_id", "turn"]] = task_3_queries[
        "session_id"
    ].str.split("_", expand=True)

    task_3_queries = task_3_queries[["query_id", "query", "session_id", "turn"]]
    
    task_3_queries["session_id"] = task_3_queries["session_id"].str.strip("session")
    task_3_queries["turn"] = task_3_queries["turn"].str.strip("query")
    task_3_queries["query"] = task_3_queries["query"].str.lower()
    
    
    task_3_queries.to_csv("data/interim/candidate_queries_task_3_new.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
