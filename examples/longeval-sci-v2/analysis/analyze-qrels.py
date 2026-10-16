from pathlib import Path
from ir_datasets_longeval import load
import pandas as pd

HERE = Path(__file__).resolve().parent

def main():
    results = pd.read_csv(HERE / "../data/processed/results.csv")
    tab = []
    for tag in ["human", "gpt-oss-120b", "dctr"]:
        ds = load(f"longeval-sci-2026/snapshot-3/{tag}")
        df = pd.DataFrame(ds.qrels)
        
        mean_judgments_query = df.groupby("query_id").size().mean()
        cov_10 = results[(results["qrels_set"] == tag) & (results["measure"] == "judged@10") & (results["query"] == "all")]["value"].mean()
        cov_100 = results[(results["qrels_set"] == tag) & (results["measure"] == "judged@100") & (results["query"] == "all")]["value"].mean()   
        tab.append({
            "qrels": tag,
            "num_queries": df["query_id"].nunique(),
            "num_labels": len(df),
            "mean_judgments_query": mean_judgments_query,
            "judged@10": cov_10,
            "judged@100": cov_100
        })
        

    print(pd.DataFrame(tab).to_latex(index=False, float_format="%.2f", caption="Summary of qrels statistics for the LongEval-Sci dataset."))
    

if __name__ == "__main__":
    main()