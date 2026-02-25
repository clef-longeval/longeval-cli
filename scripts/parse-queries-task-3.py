import omegaconf
import pandas as pd
import psycopg2

cfg = omegaconf.OmegaConf.load("config.yaml")
conn = psycopg2.connect(**cfg.db)


def main():
    task_3_queries = pd.read_csv("data/raw/task-3/public_logs.csv")
    unique_sids = list(set(task_3_queries["sid"].to_list()))
    print(f"Total queries: {len(task_3_queries)}")
    print(f"Unique search ids: {len(unique_sids)}")
    
    query = f"""
    SELECT search_id, queries.query_id, query_text as query
    FROM search_logs_normalized
    JOIN queries ON search_logs_normalized.query_id = queries.query_id
    WHERE search_id IN {tuple(unique_sids)};
    """
    searches = pd.read_sql_query(query, con=conn)
    searches.drop_duplicates(subset=["search_id"], inplace=True)

    print(f"Total searches: {len(searches)}")

    searches["query_id"] = searches["query_id"].astype(str).str.strip(".0")
    
    task_3_queries.drop(columns=["query"], inplace=True)
    
    task_3_queries = task_3_queries.merge(
        searches[["search_id", "query", "query_id"]],
        right_on="search_id",
        left_on="sid",
    )
    

    task_3_queries[["x", "session_id", "turn"]] = task_3_queries["session_id"].str.split(
        "_", expand=True
    )

    task_3_queries["session_id"] = task_3_queries["session_id"].str.strip("session")
    task_3_queries["turn"] = task_3_queries["turn"].str.strip("query")

    task_3_queries[["query_id", "query", "session_id", "turn"]].to_csv(
        "data/interim/candidate_queries_task_3.tsv", index=False, sep="\t"
    )


if __name__ == "__main__":
    main()
