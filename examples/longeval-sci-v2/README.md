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

## 7. Document Collection

Documents come from CORE through the document source `core_source.py` (set `CORE_API_KEY`; it searches 100 ids at a time and tries each request twice, as LongEval-Sci-v2 was collected). Each snapshot's filter has its publication window, the documents published in it being that snapshot's `published` candidates, and nothing published on or after `published_to` being released. The windows are the snapshots' search windows:

| snapshot   | `published_from` | `published_to` |
| ---------- | ---------------- | -------------- |
| snapshot-1 | 2025-03-01       | 2025-06-01     |
| snapshot-2 | 2025-06-01       | 2025-09-01     |
| snapshot-3 | 2025-09-01       | 2025-12-01     |

```yaml
published_from: 2025-03-01   # inclusive, UTC
published_to: 2025-06-01     # exclusive, UTC
documents:
  exclude: {abstract-length: short}
```

Collect the candidates (the `published` window listing, and the `negative` ids: the next CORE id of each qrels document), then select the documents. Each snapshot extends the previous one, so the collection only grows; a fixed `--seed` makes the samples reproducible:

```bash
export CORE_API_KEY=...
F=data/raw/filters
uv run longeval-cli collect-document-candidates $F/snapshot-1.yaml --source core_source.py --qrels work/qrels-snapshot-1.txt --database-url $DB
uv run longeval-cli select-documents $F/snapshot-1.yaml --qrels work/qrels-snapshot-1.txt --seed 42 --database-url $DB
for n in 2 3; do
  uv run longeval-cli collect-document-candidates $F/snapshot-$n.yaml --source core_source.py --qrels work/qrels-snapshot-$n.txt --database-url $DB
  uv run longeval-cli select-documents $F/snapshot-$n.yaml --qrels work/qrels-snapshot-$n.txt --extends snapshot-$((n - 1)) --seed 42 --database-url $DB
done
```

Re-selecting a snapshot replaces its selection, so re-run the snapshots that extend it too.

Fetch the selected documents. Documents already downloaded by the earlier scripts are read from their `core_works.sqlite` first (`legacy_sqlite_source.py`), then the rest come from the API; `--retry-not-found` lets CORE try the ids the old file did not have:

```bash
export LEGACY_CORE_DB=data/interim/core_works.sqlite
uv run longeval-cli fetch-documents --source legacy_sqlite_source.py --database-url $DB
uv run longeval-cli fetch-documents --source core_source.py --retry-not-found --max-workers 5 --full-text-max-chars 400000 --database-url $DB
```

Both runs can be interrupted and started again. Then flag the short abstracts (once, globally) and check what each snapshot would release:

```bash
uv run longeval-cli classify --classifier document_classifiers.py --database-url $DB
uv run longeval-cli export documents $F/snapshot-1.yaml work/documents-snapshot-1 --shard-size 100000 --database-url $DB
```

## 8. Packaging

`export snapshot --documents` writes the release: queries, qrels, the documents and a manifest. Every qrels document has to be in the collection; if some were dropped (a short abstract, published too late, never found), either fix the selection or pass `--drop-missing-documents` to leave their qrels pairs out (counted in the manifest).

```bash
uv run longeval-cli export snapshot $F/snapshot-1.yaml release \
  --scores work/dctr-snapshot-1.parquet --discretizer threshold \
  --documents --document-shard-size 100000 --database-url $DB
```
