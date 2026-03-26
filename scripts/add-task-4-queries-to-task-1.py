import pandas as pd
import json

def main():
    with open("data/processed/task-4-cleaned/query_docids.jsonl", "r") as f:
        task_4_queries = [json.loads(line) for line in f]
    
    task_1_df = pd.read_csv("data/processed/task-1-cleaned/queries-snapshot-test.tsv", sep="\t", names=["query_id", "query_text"])
    task_1_df["query_id"] = task_1_df["query_id"].astype(str)
    
    task4_df = []
    for query in task_4_queries:
        query_id = query["query_id"]
        query_texts = query["query"]
        task4_df.append((query_id, query_texts))
    
    task4_df = pd.DataFrame(task4_df, columns=["query_id", "query_text"])
    merged_df = pd.concat([task_1_df, task4_df], axis=0)
    
    print("Merged DataFrame:")
    print(len(merged_df))
    print(len(task_1_df))
    print(merged_df)
    
    merged_df.to_csv("data/processed/task-1-queries-with-task-4-queries/queries-snapshot-test.tsv", sep="\t", index=False, header=False)
        
if __name__ == "__main__":
    main()
