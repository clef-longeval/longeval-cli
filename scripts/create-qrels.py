import hashlib

import omegaconf
import pandas as pd
import psycopg2
from omegaconf import DictConfig


class DCTR:
    def __init__(self, cfg: DictConfig, conn):
        self.cfg = cfg
        self.conn = conn

    def load_candidate_query_ids(self, snapshot, min_click_threshold):
        test = False
        if snapshot == "test":
            snapshot = 1
            test = True

        df = pd.read_csv(
            "data/interim/LongEval-Sci Query Candidates - candidate_queries.tsv",
            sep="\t",
            decimal=",",
        )
        df = df[df["REMOVE (Jüri)"] != "x"]
        query_ids = set(
            df[df[f"snapshot-{snapshot}"] >= min_click_threshold]["query_id"].tolist()
        )

        # Remove further queries from snapshot 3 that were rejected later to keep the snapshots a similar size
        df = pd.read_csv(
            "data/interim/candidate-queries-snapshot-3-only.tsv", sep="\t"
        )
        rejected = set(df[df["REMOVE (Jüri)"] == "x"]["query_id"].to_list())
        query_ids = query_ids.difference(rejected)

        if test:
            test_query_ids = set(
                pd.read_csv("data/interim/test-queries.tsv", sep="\t")[
                    "query_id"
                ].to_list()
            )
            query_ids = query_ids.intersection(test_query_ids)
        if not test and snapshot == 1:
            test_query_ids = set(
                pd.read_csv("data/interim/test-queries.tsv", sep="\t")[
                    "query_id"
                ].to_list()
            )
            query_ids = query_ids.difference(test_query_ids)

        print(
            f"Loaded {len(query_ids)} candidate query IDs with at least {min_click_threshold} clicks."
        )
        return query_ids

    def load_searches(self, snapshot, query_ids):
        if snapshot == "test":
            snapshot = 1
        query = f"""
        SELECT search_id, serp, query_id, original_id
        FROM search_logs_normalized
        WHERE query_id in {tuple(query_ids)}
            AND log_date >= '{self.cfg["snapshots"][snapshot]["start_date"]}'
            AND log_date <= '{self.cfg["snapshots"][snapshot]["end_date"]}'
        """
        searches = pd.read_sql_query(query, con=self.conn)
        searches = searches.drop_duplicates(subset=["search_id"])
        return searches

    def create_impressions_table(self, searches):
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

    def load_clicks(self, snapshot, searches):
        if snapshot == "test":
            snapshot = 1
        search_ids = list(searches["search_id"].unique())

        query = f"""
        SELECT *
        FROM tracker_logs
        WHERE search_id in {tuple(search_ids)}
            AND log_date >= '{self.cfg["snapshots"][snapshot]["start_date"]}'
            AND log_date <=  '{self.cfg["snapshots"][snapshot]["end_date"]}'
            AND (entry_type = 'get-pdf' OR entry_type = 'works')
        """
        clicks = pd.read_sql_query(query, con=self.conn)

        searches_query_ids = (
            searches[["query_id", "search_id"]]
            .set_index("search_id")
            .to_dict()["query_id"]
        )

        clicks["query_id"] = clicks["search_id"].map(searches_query_ids)
        clicks = (
            clicks[["query_id", "doc_id", "search_id"]]
            .groupby(["query_id", "doc_id"])
            .count()
            .reset_index()
            .copy()
        )
        clicks = clicks.rename(columns={"search_id": "clicks"})
        return clicks

    def create_dctr_table(self, impressions, clicks):
        dctr = impressions.merge(clicks, on=["query_id", "doc_id"], how="left").copy()
        dctr["dctr"] = dctr["clicks"] / dctr["impressions"]
        dctr.fillna(0, inplace=True)
        return dctr

    def discretize_labels(self, dctr):
        def _discretize(score):
            if score < 0.1:
                return 0
            elif score < 0.6:
                return 1
            else:
                return 2

        dctr["label"] = dctr["dctr"].apply(_discretize)
        return dctr

    def hash_query_id(self, query_id):
        return hashlib.md5(str(query_id).encode()).hexdigest()

    def create_qrels(self, dctr):
        valid_queries = dctr[dctr["label"] > 0]["query_id"].unique()
        dctr = dctr[dctr["query_id"].isin(valid_queries)]
        dctr["q0"] = "0"
        dctr = dctr[["query_id", "q0", "doc_id", "label"]]
        return dctr

    def create_queries(self, qrels):
        query_ids = set(qrels["query_id"].to_list())
        query = f"""
        SELECT query_text, query_id
        FROM queries
        WHERE query_id in {tuple(query_ids)}
        """
        queries = pd.read_sql_query(query, con=self.conn)
        return queries

    def create_dctr_dataset(self, snapshot, min_click_threshold):
        query_ids = self.load_candidate_query_ids(snapshot, min_click_threshold)
        searches = self.load_searches(snapshot, query_ids)

        impressions = self.create_impressions_table(searches)
        clicks = self.load_clicks(snapshot, searches)
        dctr = self.create_dctr_table(impressions, clicks)
        dctr = self.discretize_labels(dctr)
        qrels = self.create_qrels(dctr)
        queries = self.create_queries(qrels)
        queries["query_id"] = queries["query_id"].apply(self.hash_query_id)
        qrels["query_id"] = qrels["query_id"].apply(self.hash_query_id)

        queries.to_csv(
            f"data/interim/queries-snapshot-{snapshot}.tsv",
            sep="\t",
            index=False,
            header=False,
        )
        qrels.to_csv(
            f"data/interim/qrels-snapshot-{snapshot}",
            sep=" ",
            index=False,
            header=False,
        )


def main():
    cfg = omegaconf.OmegaConf.load("config.yaml")
    conn = psycopg2.connect(**cfg.db)

    min_click_threshold = 30
    snapshot = "test"
    snapshot = 1

    dctr_creator = DCTR(cfg, conn)
    dctr_creator.create_dctr_dataset(snapshot, min_click_threshold)


if __name__ == "__main__":
    main()
