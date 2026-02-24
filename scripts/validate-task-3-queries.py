import pandas as pd

def main():
    new_queries = pd.read_csv("data/interim/candidate_queries_task_3_new.tsv", sep="\t")
    queries = pd.read_csv("data/interim/candidate_queries_task_3.tsv", sep="\t")
    
    task_3_queries = pd.read_csv("data/raw/task-3/public_logs.csv")
    
    print(f"New queries: {len(new_queries)}")
    print(f"Original queries: {len(queries)}")
    print(f"Sum of new and original queries: {len(new_queries) + len(queries)}")
    
    print(f"Sessions: {len(task_3_queries)}")
    

    assert len(task_3_queries) == len(queries) + len(new_queries), "Number of queries should be the same"
    
    print("All queries are valid and accounted for.")


if __name__ == "__main__":
    main()
