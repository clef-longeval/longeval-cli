"""Monthly click activity by type, split into valid and late clicks."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

OUTPUT = Path(__file__).resolve().parent / "output"


def main():
    data = pd.read_csv(OUTPUT / "monthly_clicks.csv")
    data = data[data["month"].str.startswith("2025-")].copy()
    data["type"] = data["type"].fillna("Unknown")
    data["late_clicks"] = data["clicks"] - data["valid_clicks"]
    months = pd.period_range(data["month"].min(), data["month"].max(), freq="M").astype(str)
    types = ["works", "author", "data-provider"]
    colors = ["#AA3377", "#4477AA", "#228833"]
    titles = ["Works", "Authors", "Data providers"]
    plt.rcParams.update({"font.size": 16, "pdf.fonttype": 42})
    fig, axes = plt.subplots(
        1, 3, figsize=(12, 4.5),
        layout="constrained", squeeze=False,
    )
    for ax, click_type, color, title in zip(axes.flat, types, colors, titles):
        monthly = data[data["type"].eq(click_type)].set_index("month")[["valid_clicks", "late_clicks"]].reindex(months, fill_value=0)
        lighter = tuple(0.65 + 0.35 * channel for channel in to_rgb(color))
        ax.bar(months, monthly["valid_clicks"], color=color)
        ax.bar(months, monthly["late_clicks"], bottom=monthly["valid_clicks"], color=lighter)
        ax.set_title(title)
        ax.set_xticks(range(len(months)), pd.to_datetime(months).month)
        ax.tick_params(axis="x", labelsize=13)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e6:g}"))
    axes[0, 0].set_ylabel("Clicks (millions)")
    fig.legend(handles=[
        Patch(facecolor="#666666", label="Valid"),
        Patch(facecolor="#DDDDDD", label="Late"),
    ], loc="outside lower center", ncols=2, frameon=False)
    fig.savefig(OUTPUT / "click_activity.pdf")
    plt.close(fig)
    print(data.groupby("type")[["clicks", "valid_clicks", "late_clicks"]].sum())
    print(f"Saved {OUTPUT / 'click_activity.pdf'}")


if __name__ == "__main__":
    main()
