"""Check a trained click model against the ground truth the logs were simulated from.

    uv run python examples/minimal-example/check_scores.py \
        /tmp/minimal-scores.parquet --qrels /tmp/minimal-qrels.txt

``--qrels`` may also be gzipped, e.g. a snapshot's ``qrels.trec.gz``.

``generate_logs.py`` knows every document's true grade (``ground_truth.tsv``),
so this prints, and with ``--strict`` asserts, how well the learned ``score``
recovers it: the mean ``score`` per grade must rise with the grade, and the
qrels label must equal the grade for most pairs. Exact ``score`` values vary
between runs and machines; these checks do not.
"""

import csv
import gzip
from collections import defaultdict
from pathlib import Path
from statistics import mean

import click
import pyarrow.parquet as pq

from src.importing.loader import load_parser

HERE = Path(__file__).parent


@click.command()
@click.argument("scores", type=click.Path(exists=True, dir_okay=False))
@click.option("--qrels", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--min-agreement", default=0.85, show_default=True)
@click.option("--strict", is_flag=True, help="Exit non-zero if a check fails.")
def main(scores: str, qrels: str | None, min_agreement: float, strict: bool) -> None:
    parser = load_parser(HERE / "parser.py")  # the IDs the import assigned
    with open(HERE / "ground_truth.tsv", encoding="utf-8", newline="") as file:
        grade = {
            (parser.query_id(row["query"]), row["doc_id"]): int(row["grade"])
            for row in csv.DictReader(file, delimiter="\t")
        }

    table = pq.read_table(scores).to_pydict()
    by_grade = defaultdict(list)
    for query_id, doc_id, score in zip(
        table["query_id"], table["doc_id"], table["score"]
    ):
        by_grade[grade[query_id, doc_id]].append(score)
    means = [mean(by_grade[g]) for g in sorted(by_grade)]
    for g, value in zip(sorted(by_grade), means):
        click.echo(f"grade {g}: {len(by_grade[g]):>4} pairs, mean score {value:.3f}")
    failures = []
    if means != sorted(means):
        failures.append("mean score does not increase with the grade")

    if qrels:
        with (gzip.open if qrels.endswith(".gz") else open)(qrels, "rt") as file:
            pairs = [line.split() for line in file]
        agree = mean(int(label) == grade[q, d] for q, _, d, label in pairs)
        click.echo(f"qrels label == grade for {agree:.0%} of {len(pairs)} pairs")
        if agree < min_agreement:
            failures.append(f"qrels agree with the grades below {min_agreement:.0%}")

    for failure in failures:
        click.echo(f"FAILED: {failure}", err=True)
    if strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
