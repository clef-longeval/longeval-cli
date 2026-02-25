import pandas as pd


def main():
    df = pd.read_csv(
        "data/interim/LongEval-Sci Query Candidates - candidate_queries.tsv", sep="\t"
    )
    removed_queries = df[df["REMOVE (Jüri)"] == "x"]["query_id"].tolist()

    df = pd.read_csv("data/interim/new_candidate_queries_to_check.csv")
    removed_queries_2 = df[df["remove"] == "x"]["query_id"].tolist()

    removed_queries.extend(removed_queries_2)

    candidates = pd.read_csv("data/interim/candidate_queries_task_1.tsv", sep="\t")
    
    print("Number of candidates before filtering:", len(candidates))
    candidates = candidates[~candidates["query_id"].isin(removed_queries)]
    print("Number of candidates after filtering:", len(candidates))
    
    candidates.to_csv("data/interim/candidate_queries_task_1_filtered.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
