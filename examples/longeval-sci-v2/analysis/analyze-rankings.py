from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from repro_eval.measure.document_order import RBO
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
TAGS = ["human", "gpt-oss-120b", "dctr"]
NAMES = ["Human", "LLM", "DCTR"]
MEASURE = "nDCG@10"
RBO_P = 0.9


def main():
    results = pd.read_csv(HERE / "../data/processed/results.csv")
    results = results[
        (results["snapshot"] == "snapshot-3")
        & (results["meta_measure"] == "ARP")
        & (results["measure"] == MEASURE)
        & results["qrels_set"].isin(TAGS)
    ].copy()
    results["system"] = results["team"] + "/" + results["run"]
    # Each source uses its own full query set; retain the same systems.
    scores = results.pivot(index="system", columns="qrels_set", values="value")[
        TAGS
    ].dropna()
    tau, rbo = np.eye(3), np.eye(3)
    for i, j in combinations(range(3), 2):
        a, b = scores[TAGS[i]], scores[TAGS[j]]
        tau[i, j] = tau[j, i] = kendalltau(a, b).statistic
        # repro_eval expects topic -> ranked item -> score. Here items are systems.
        # Its RBO breaks score ties by reverse alphabetical system ID.
        rbo[i, j] = rbo[j, i] = RBO(
            {"systems": a.sort_values(ascending=False).to_dict()},
            {"systems": b.sort_values(ascending=False).to_dict()},
            p=RBO_P,
            depth=len(scores),
            pbar=False,
            per_topic=False,
        )

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.6), layout="constrained")
    for ax, matrix, title, cmap, low in zip(
        axes,
        [tau, rbo],
        [r"Kendall's $\tau_b$", f"RBO (p={RBO_P})"],
        ["coolwarm", "Blues"],
        [-1, 0],
    ):
        ax.imshow(matrix, cmap=cmap, vmin=low, vmax=1)
        ax.set_xticks(range(3), NAMES)
        ax.set_yticks(range(3), NAMES)
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if abs(matrix[i, j]) > 0.7 else "black",
                )
    fig.suptitle(f"{MEASURE}, snapshot 3, {len(scores)} runs")
    (HERE / "output").mkdir(exist_ok=True)
    fig.savefig(HERE / "output/ranking-agreement.pdf")
    plt.close(fig)
    print("Kendall tau-b:\n", pd.DataFrame(tau, index=NAMES, columns=NAMES))
    print("RBO:\n", pd.DataFrame(rbo, index=NAMES, columns=NAMES))


if __name__ == "__main__":
    main()
