from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ir_datasets_longeval import load

TAGS = ["human", "gpt-oss-120b", "dctr"]
NAMES = {"human": "Human", "gpt-oss-120b": "LLM", "dctr": "DCTR"}
LABELS = [0, 1, 2]

qrels = []
for tag in TAGS:
    ds = load(f"longeval-sci-2026/snapshot-3/{tag}")
    df = pd.DataFrame(ds.qrels)
    df = df.rename(columns={"relevance": tag})
    df.drop(columns=["iteration"], inplace=True)
    qrels.append(df)

df = pd.merge(qrels[0], qrels[1], on=["query_id", "doc_id"], how="outer")
df = pd.merge(df, qrels[2], on=["query_id", "doc_id"], how="outer")

# Labels outside 0-2 (gpt-oss-120b has a few 3s) are treated as missing.
df[TAGS] = df[TAGS].where(df[TAGS].isin(LABELS))


def kappa(confusion):
    """Quadratically weighted Cohen's kappa of a label confusion matrix."""
    k = len(confusion)
    i, j = np.indices((k, k))
    disagreement = ((i - j) / (k - 1)) ** 2
    observed = confusion / confusion.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    return 1 - (disagreement * observed).sum() / (disagreement * expected).sum()


def binarize(confusion):
    """Collapse a graded confusion matrix to not relevant (0) vs relevant (>0)."""
    return np.array(
        [
            [confusion[0, 0], confusion[0, 1:].sum()],
            [confusion[1:, 0].sum(), confusion[1:, 1:].sum()],
        ]
    )


# Agreement for each pair of qrels types, on the (query, doc) pairs both judged.
pairs = list(combinations(TAGS, 2))
rows = []
confusions = {}
for a, b in pairs:
    both = df[[a, b]].dropna().astype(int)
    confusion = pd.crosstab(both[a], both[b]).reindex(
        index=LABELS, columns=LABELS, fill_value=0
    )
    confusions[a, b] = confusion
    graded = confusion.to_numpy()
    binary = binarize(graded)
    rows.append(
        {
            ("", "Pair"): f"{NAMES[a]} vs {NAMES[b]}",
            ("", "Overlap"): len(both),
            ("Graded", "Agreement"): np.trace(graded) / graded.sum(),
            ("Graded", r"$\kappa_w$"): kappa(graded),
            ("Binary", "Agreement"): np.trace(binary) / binary.sum(),
            ("Binary", r"$\kappa$"): kappa(binary),
        }
    )

summary = pd.DataFrame(rows)
summary.columns = pd.MultiIndex.from_tuples(summary.columns)
print(
    summary.to_latex(
        index=False,
        float_format="%.3f",
        multicolumn_format="c",
        caption="Pairwise label agreement on the jointly judged "
        "(query, document) pairs, graded (0/1/2) and binary (0 vs 1/2). "
        r"$\kappa_w$ is quadratically weighted Cohen's kappa; for two labels "
        "the weighting makes no difference.",
        label="tab:label-agreement",
    )
)
print(f"\nJudged by all three: {len(df[TAGS].dropna())}")

# One confusion matrix per pair, counts in each cell. Sized for the Springer
# LNCS text width (122 mm), so the fonts print at their nominal size.
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, len(pairs), figsize=(4.8, 1.9), layout="constrained")
for ax, (a, b) in zip(axes, pairs):
    confusion = confusions[a, b]
    ax.imshow(confusion, cmap="Reds")
    # Dark text on the light cells, white on the dark ones.
    half = confusion.to_numpy().max() / 2
    for (i, j), n in np.ndenumerate(confusion.to_numpy()):
        color = "w" if n > half else "k"
        ax.text(j, i, n, ha="center", va="center", color=color)
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel(NAMES[b])
    ax.set_ylabel(NAMES[a])
    ax.set_title(f"n={confusion.to_numpy().sum()}")

output = Path(__file__).with_name("label-agreement.pdf")
fig.savefig(output)
print(f"Saved {output}")
