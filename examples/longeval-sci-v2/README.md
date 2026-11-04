# Example: a LongEval Sci-style collection from CORE logs

How to build a collection like LongEval Sci 2026 (paper §4) with
`longeval-cli`: three snapshots of the CORE search logs from 2025, following the
paper's method. The released LongEval Sci 2026 collection was built with
earlier scripts, before this toolkit existed, so the numbers you get here can
differ from the paper's.

The CORE logs can't be published, so this is a realistic setup to copy and
adapt, not something you can run as is. For a runnable walkthrough, see the
[tutorial](../minimal-example/README.md).

Run every command from this directory:

```bash
cd examples/longeval-sci-v2
export DB=sqlite:///longeval.sqlite3   # or postgresql+psycopg://...
F=filters
```

## The choices at a glance

| Decision | Setting | Where |
| --- | --- | --- |
| Snapshots | `snapshot-1` 2025-03–05, `snapshot-2` 06–08, `snapshot-3` 09–11 (`[from, to)`, UTC) | `filters/snapshot-N.yaml` |
| Query normalization | Unicode normalization, case folding, whitespace cleanup | `parser.py` |
| Clicks | only publication clicks (`works`); clicks more than 24 h after their search are excluded | filter `click_types`, `interactions:` |
| Query selection | no field names, Boolean operators or DOIs; at least 31 clicks in the snapshot; not removed by review | filter `queries:` |
| Click model | CLAX `DocumentCTRModel` (DCTR) | `train_dctr.py` |
| Documents | all qrels documents + seeded samples of documents published in the snapshot and of `id + 1` negatives; each snapshot extends the previous one; abstracts shorter than 100 characters are dropped | `core_source.py`, filter `documents:` |

| File | What it is |
| --- | --- |
| `parser.py` | the CORE log parser (searches have `search_id`, interactions `trackId`) |
| `classifiers.py` | `is-boolean`, `word-count`, `has-doi`, `is-structured`, `click-delay`; global |
| `clicks.py` | click counts; once per snapshot |
| `manual.py` | the reviewer's labels from the checked query sheet |
| `document_classifiers.py` | `abstract-length`; global |
| `train_dctr.py` | fits DCTR for one snapshot and writes the scores Parquet |
| `core_source.py` | document source for the CORE API (needs `CORE_API_KEY`) |
| `filters/snapshot-N.yaml` | one filter per snapshot: date range, click types, rules |
| `analysis/` | the scripts behind the paper's figures (§4.1); they read the full database and are not part of the example |

## 1. Ingest Logs

```bash
uv run longeval-cli init-db --database-url "$DB"
uv run longeval-cli import-queries data/raw/logs --parser parser.py --database-url "$DB"
uv run longeval-cli import-clicks  data/raw/logs --parser parser.py --database-url "$DB"
```

On the full year of logs this takes about 10 h for the searches and 1.5 h for
the clicks on a laptop (paper §4.1).

## 2. Select Queries

Classify once globally, then count the clicks per snapshot. `clicks` counts
only the clicks the filter accepts, so it runs after `click-delay`:

```bash
uv run longeval-cli classify --classifier classifiers.py --database-url "$DB"
for n in 1 2 3; do
  uv run longeval-cli classify --classifier clicks.py --filter $F/snapshot-$n.yaml --database-url "$DB"
done
```

Export the candidates for review. The command prints how many queries remain
after each rule. The sheets hold raw user queries, so `data/` is gitignored:

```bash
uv run longeval-cli export queries $F/snapshot-3.yaml \
  --output data/interim/queries-to-check/candidate-queries-snapshot-3.tsv --database-url "$DB"
```

A reviewer fills the `manual_label` column and saves the sheet to
`data/interim/queries-checked/`. Load the labels with `manual.py`; the filters
then drop the queries labelled `remove` with `exclude: {manual: remove}` (other
labels are kept):

```bash
uv run longeval-cli classify --classifier manual.py --database-url "$DB"
```

## 3. Create Labels

Train DCTR per snapshot (needs `uv sync --extra clax`), then turn the click
probabilities into qrels:

```bash
for n in 1 2 3; do
  uv run python train_dctr.py --filter $F/snapshot-$n.yaml --database-url "$DB" \
    --output work/dctr-snapshot-$n.parquet
  uv run longeval-cli export qrels work/dctr-snapshot-$n.parquet \
    --discretizer threshold --output work/qrels-snapshot-$n.txt
done
```

`threshold` defaults to 0.1/0.9 (labels 0/1/2), as in the earlier LongEval
collections; `chapelle` is the alternative. Check the label distribution before
choosing: DCTR's probabilities can be distributed differently from the plain
click ratios these thresholds were chosen for.

## 4. Fetch Docs

Each filter's `published_from`/`published_to` equals its search window. The
documents published in it are the `published` candidates, and nothing published
on or after `published_to` is released. `core_source.py` derives the `negative`
candidates as the next CORE id of each qrels document. `--extends` carries every
document of the previous snapshot over, so the collection only grows:

```bash
export CORE_API_KEY=...
for n in 1 2 3; do
  uv run longeval-cli collect-document-candidates $F/snapshot-$n.yaml --source core_source.py \
    --qrels work/qrels-snapshot-$n.txt --database-url "$DB"
  EXT=$([ $n -gt 1 ] && echo "--extends snapshot-$((n - 1))")
  uv run longeval-cli select-documents $F/snapshot-$n.yaml --qrels work/qrels-snapshot-$n.txt \
    $EXT --seed 42 --database-url "$DB"
done
```

Re-selecting a snapshot replaces its selection, so re-run the snapshots that
extend it as well.

Fetch the selected documents from the CORE API, then flag short abstracts. An
interrupted fetch can simply be restarted; documents already fetched are
skipped:

```bash
uv run longeval-cli fetch-documents --source core_source.py \
  --max-workers 5 --full-text-max-chars 400000 --database-url "$DB"
uv run longeval-cli classify --classifier document_classifiers.py --database-url "$DB"
```

## 5. Package

```bash
for n in 1 2 3; do
  uv run longeval-cli export snapshot $F/snapshot-$n.yaml release \
    --scores work/dctr-snapshot-$n.parquet --discretizer threshold \
    --documents --document-shard-size 100000 --database-url "$DB"
done
```

Every qrels document has to be in the collection. If some were dropped (short
abstract, published too late, never found), fix the selection or pass
`--drop-missing-documents`, which leaves those qrels pairs out and counts them
in the manifest.

## 6. Retrieve

```bash
uv run longeval-cli retrieve release/snapshot-1 work/retrieval
```

Runs the default PyTerrier baselines through TIRA (needs `uv sync --extra tira`
and Docker); see the [command reference](../../docs/reference.md#6-retrieve-baselines).
