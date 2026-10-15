from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ir_datasets_longeval import load

TAGS = ["human", "gpt-oss-120b", "dctr"]

qrels = []
for tag in TAGS:
    ds = load(f"longeval-sci-2026/snapshot-3/{tag}")
    df = pd.DataFrame(ds.qrels)
    df = df.rename(columns={"relevance": tag})
    df.drop(columns=["iteration"], inplace=True)
    qrels.append(df)

df = pd.merge(qrels[0], qrels[1], on=["query_id", "doc_id"], how="outer")
df = pd.merge(df, qrels[2], on=["query_id", "doc_id"], how="outer")

print(len(df))


# Relative label distribution per qrels type, one stacked bar each, with the
# total number of judgments under its name. Labels outside 0-2 (gpt-oss-120b
# has a few 3s) are dropped.
LABELS = {0: "Not relevant", 1: "Relevant", 2: "Highly relevant"}
NAMES = {"human": "Human", "gpt-oss-120b": "LLM", "dctr": "DCTR"}
counts = pd.DataFrame({tag: df[tag].value_counts() for tag in TAGS})
counts = counts.reindex(LABELS).astype(int)
dist = (counts / counts.sum()).T.rename(columns=LABELS)
dist.index = [f"{NAMES[tag]}\n(n={n})" for tag, n in counts.sum().items()]

# Sized for the Springer LNCS text width (122 mm), so the fonts print at their
# nominal size.
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.2, 3), layout="constrained")
dist.plot.bar(ax=ax, stacked=True, width=0.8, rot=0, legend=False)
# bar_label can only center or end-align, so place each share at its start.
for container in ax.containers:
    for bar in container:
        x, y = bar.get_x() + bar.get_width() / 2, bar.get_y()
        ax.text(x, y + 0.01, f"{bar.get_height():.0%}", ha="center", va="bottom")
ax.set_ylim(0, 1.08)  # room for the labels of thin top segments
fig.legend(loc="outside lower center", ncol=2)

output = Path(__file__).with_name("label-distribution.pdf")
fig.savefig(output)
print(f"Saved {output}")
