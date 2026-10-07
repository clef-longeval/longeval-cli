# LongEval-Sci-v2

This is an example on how to recreate the LongEval-Sci-v2 dynamic test collection with the `longeval-cli`. Run every command from this directory.

```bash
export DB=sqlite:///longeval.sqlite3   # or set LONGEVAL_DATABASE_URL
```

> A database created before classifications gained `snapshot`/`value` columns has to be recreated (delete `longeval.sqlite3` and re-import); `create_schema` does not alter existing tables.

## 1. Log Ingestion
Import the logs from `data/logs` with the `parser.py` defined here:

```bash
uv run longeval-cli import-queries data/logs --parser parser.py --database-url $DB
uv run longeval-cli import-clicks data/logs --parser parser.py --database-url $DB
```

## 2. Query Classification

Every `--classifier` is a Python file in `classifiers/` defining a classifier or a list of them. `text.py` bundles the four text classifiers: the generic `is-boolean` and `word-count` built-ins and the CORE-specific `has-doi` and `is-structured`. Each classifier writes its own row per query. They don't depend on time, so they run once (global):

```bash
uv run longeval-cli classify --classifier classifiers/text.py --database-url $DB
```

Click counts depend on the snapshot's date range, so they run once per snapshot. The range lives in the snapshot's filter file (`data/raw/filters/snapshot-N.yaml`: `snapshot` (its name), `from` inclusive, `to` exclusive, UTC), the one file you keep editing while building the snapshot, so it is stated only once. `clicks.py` (kept separate because it runs per snapshot) counts the clicks the filter defines (its `click_types`, minus excluded sessions/interactions), stored under the filter's `snapshot` name:

```bash
for n in 1 2 3; do
  uv run longeval-cli classify --classifier classifiers/clicks.py --filter data/raw/filters/snapshot-$n.yaml --database-url $DB
done
```

## 3. Manual Review

Export the queries with their classifier columns (click counts show up as `clicks@snapshot-N`), fill the `manual_label` column elsewhere (e.g. `remove`), save as `review/annotated.csv`, and import it as the `manual` classifier:

```bash
uv run longeval-cli export-queries review/queries.csv --database-url $DB
# ... annotate review/queries.csv -> review/annotated.csv ...
uv run longeval-cli classify --classifier classifiers/manual.py --database-url $DB
```

## 4. Query Filtering

One filter file per snapshot (`filters/snapshot-N.yaml`) keeps the queries that pass the automatic filters, have more than 30 qualifying clicks in that snapshot, and weren't removed by review. The command prints how many queries survive each rule. The files exclude on `manual`, so run step 3 first (or delete the `exclude` block to preview the funnel before review):

```bash
uv run longeval-cli filter-queries filters/snapshot-$n.yaml --output work/queries-snapshot-$n.tsv --database-url $DB

```

## 6. Relevance Labels (qrels)

`export qrels` writes TREC qrels from that file, with one of the built-in discretizers (no wrapper files): `threshold` (default 0.1/0.9 -> labels 0/1/2, as in the earlier pipeline), `chapelle` (the DCG-discount inversion `g = log2(P * 4 + 1)`, rounded, giving 0/1/2). Leave `--discretizer` out to get the raw probabilities.

```bash
uv run longeval-cli export qrels work/dctr-snapshot-1.parquet --discretizer chapelle --output work/qrels-snapshot-1.txt
uv run longeval-cli export qrels work/dctr-snapshot-1.parquet --discretizer threshold --output work/qrels-snapshot-1-dctr.txt
uv run longeval-cli export qrels work/dctr-snapshot-1.parquet --output work/qrels-snapshot-1-prob.txt
```

Note that the 0.1/0.9 thresholds were chosen for plain clicks-per-query session ratios; a CLAX DCTR model (as trained in `examples/minimal-example/train_dctr_from_*.py`) is gradient-trained, whose probabilities (and therefore the label distribution) can differ. Check the label counts before using them.
