import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kendalltau

TAGS = ["human", "gpt-oss-120b", "dctr"]
NAMES = {"human": "Human", "gpt-oss-120b": "LLM", "dctr": "DCTR"}
MEASURES = ["nDCG@10", "nDCG", "AP", "P@10"]
SNAPSHOT = "snapshot-3"
# The measure the per-run scatter plots show.
PLOT_MEASURE = "nDCG@10"

default = Path(__file__).parents[1] / "data" / "processed" / "results.csv"
results = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else default)
results = results[results["snapshot"] == SNAPSHOT]

# Mean score of each run (ARP over all queries), one column per qrels type.
arp = results[(results["meta_measure"] == "ARP") & results["measure"].isin(MEASURES)]
arp = arp.assign(system=arp["team"] + "/" + arp["run"])
scores = arp.pivot_table(
    index=["measure", "system"], columns="qrels_set", values="value"
)
tags = [tag for tag in TAGS if tag in scores.columns]
missing = sorted(set(TAGS) - set(tags))
if missing:
    print(f"Missing qrels types in the results: {', '.join(missing)}")
if len(tags) < 2:
    sys.exit("Need at least two qrels types to compare system rankings.")

# Kendall's tau-b between the system rankings of each pair of qrels types, per
# measure, on the runs scored with both. A pair without common runs is left
# empty.
pairs = list(combinations(tags, 2))
rows = []
for measure, group in scores.groupby(level="measure"):
    row = {("", "Measure"): measure}
    for a, b in pairs:
        both = group[[a, b]].dropna()
        tau = kendalltau(both[a], both[b])[0] if len(both) > 1 else None
        row[(f"{NAMES[a]} vs {NAMES[b]}", r"$\tau$")] = tau
        row[(f"{NAMES[a]} vs {NAMES[b]}", "Runs")] = len(both)
    rows.append(row)

summary = pd.DataFrame(rows)
summary.columns = pd.MultiIndex.from_tuples(summary.columns)
print(
    summary.to_latex(
        index=False,
        float_format="%.3f",
        na_rep="--",
        multicolumn_format="c",
        caption="Kendall's $\\tau$ between the system rankings induced by each "
        f"pair of qrels types on {SNAPSHOT}, per measure (mean over queries).",
        label="tab:ranking-correlation",
    )
)

# One scatter plot of the run ranks per pair of qrels types, one marker per
# run. Sized for the Springer LNCS text width (122 mm), so the fonts
# print at their nominal size.
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
fig, axes = plt.subplots(
    1,
    len(pairs),
    figsize=(4.8, 4.8 / len(pairs) + 0.4),
    layout="constrained",
    squeeze=False,
)
plotted = scores.xs(PLOT_MEASURE, level="measure")
for ax, (a, b) in zip(axes[0], pairs):
    ranks = plotted[[a, b]].dropna().rank(ascending=False)
    ax.scatter(ranks[a], ranks[b], s=10)
    n = len(ranks)
    ax.plot([1, n], [1, n], color="grey", linewidth=0.5, zorder=0)
    ax.set_xlabel(f"Rank ({NAMES[a]})")
    ax.set_ylabel(f"Rank ({NAMES[b]})")
    ax.set_aspect("equal")
fig.suptitle(f"{PLOT_MEASURE}, {SNAPSHOT}")

output = Path(__file__).with_name("ranking-correlation.pdf")
fig.savefig(output)
print(f"Saved {output}")
