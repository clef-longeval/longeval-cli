import os
import xml.etree.ElementTree as ET
from typing import List

import pandas as pd
from tqdm import tqdm


def traverse_logs(logs_path, k=None):
    file_paths = [logs_path / f for f in logs_path.iterdir() if f.suffix == ".log"]

    c = 0
    for file_path in file_paths:
        with open(file_path, "r") as f:
            for line in tqdm(f, desc=f"Processing {file_path.name}"):
                c += 1
                if k is not None and c > k:
                    return
                yield line


def bin_logged_queries(
    query: pd.DataFrame, bins: List[float] = [0, 0.3, 0.6, 1.0]
) -> pd.DataFrame:
    """Split the queries into head, torso and tail based on the frequency the queries were issued.

    Args:
        query (pd.DataFrame): Dataframe with query logs from `parse_logs()`
        bins (List[float], optional): Bins for the splits. Defaults to [0, 0.3, 0.6, 1.0].

    Returns:
        pd.DataFrame: Dataframe with the queries binned into head, torso and tail.
    """
    value_counts = query["query"].value_counts()
    cumulative_percentage = value_counts.cumsum() / value_counts.sum()
    bins = pd.cut(
        cumulative_percentage,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["head", "torso", "tail"],
        include_lowest=True,
    )
    binned_queries = pd.DataFrame(
        {
            "query": value_counts.index,
            "count": value_counts.values,
            "cumulative_percentage": cumulative_percentage.values,
            "bin": bins.values,
        }
    )
    return binned_queries


def save_topics(
    df: pd.DataFrame,
    format: str,
    path: str = "",
    task: str = "",
    batch: str = "",
    split: str = "",
):
    """Write topics to file"""
    file_name = f"topics-{task}-{batch}-{split}".strip("-")
    if format == "trecxml":
        root = ET.Element("topics", task=task, batch=batch)

        for index, row in df.iterrows():
            topic = ET.SubElement(root, "topic", number=str(row["qid"]))

            query = ET.SubElement(topic, "query")
            query.text = row["query"]

        # Save to file
        xml_output = ET.tostring(root, encoding="unicode")
        with open(os.path.join(path, f"{file_name}.xml"), "w", encoding="utf-8") as f:
            f.write(xml_output)
    elif format == "tsv":
        df.to_csv(os.path.join(path, file_name), sep="\t", index=False, header=False)
    else:
        raise ValueError(f"Format {format} not supported")


def save_qrels(
    df: pd.DataFrame, path: str = "", task: str = "", batch: str = "", split: str = ""
):
    """Write qrels to file"""
    file_name = f"qrels-{task}-{batch}-{split}".strip("-")
    # write qrels
    df["iter"] = batch
    df = df[["qid", "iter", "docid", "label"]].copy()
    df["docid"] = df["docid"].astype(str)
    df["qid"] = df["qid"].astype(str)
    df["label"] = df["label"].astype(int)

    df.to_csv(os.path.join(path, file_name), index=False, header=False, sep=" ")
