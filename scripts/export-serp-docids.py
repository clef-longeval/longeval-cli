import omegaconf
import pandas as pd
import psycopg2

cfg = omegaconf.OmegaConf.load("config.yaml")
conn = psycopg2.connect(**cfg.db)


def main():
    train_queries = pd.read_csv(
        "data/interim/candidate_queries_task_1_filtered.tsv", sep="\t"
    )
    test_queries = pd.read_csv("data/interim/candidate_queries_task_3.tsv", sep="\t")

    queries = pd.concat([train_queries, test_queries], ignore_index=True)
    query_ids = queries["query_id"].unique().tolist()

    query = f"""
    SELECT serp
    FROM search_logs_normalized
    WHERE query_id IN {tuple(query_ids)}
    """
    serps = pd.read_sql_query(query, con=conn)
    serps = serps["serp"].explode()
    
    serps = serps.drop_duplicates()
    
    
    serps.to_csv("data/processed/serp-docids.tsv", index=False, header=False)


if __name__ == "__main__":
    main()
