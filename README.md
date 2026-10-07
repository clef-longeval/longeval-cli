# 📟 longeval-cli

Create dynamic test collections from search logs. The toolkit imports searches
and interactions, classifies and filters them, prepares click-model data, and
exports queries and relevance judgments.

## Installation

```bash
uv sync
uv run longeval-cli init-db --database-url sqlite:///work/longeval.sqlite3
```

Install the optional click-model dependencies with `uv sync --extra clax` when
training or scoring a CLAX model.

## Usage

The workflow is `import-queries` → `import-clicks` → `classify` → train and
score a click model → `export snapshot`. The [minimal example](examples/minimal-example/README.md)
runs every step with sample data.

### 1. Log Ingestion

Imports raw search logs into the database in two passes: first the search events (queries + their result rankings), then the click/interaction events, each linked back to the search it belongs to via the log's own search id. Run `import-queries` before `import-clicks`.

```bash
uv run longeval-cli import-queries PATHS --parser FILE --database-url URL
uv run longeval-cli import-clicks PATHS --parser FILE --database-url URL
```

`--parser` points to a Python file that reads your log format; see the
[minimal parser](examples/minimal-example/parser.py).


### 2. Classify and filter

Classifiers label queries, sessions, results, or interactions. Their verdicts
are stored separately by level. `--classifier` points to a Python file defining
one or more classifiers; see the [minimal classifiers](examples/minimal-example/classifiers.py).

A filter YAML names the snapshot, selects a half-open UTC date range `[from,
to)`, lists the interaction types that count as clicks, and optionally applies
`require`, `min`, or `exclude` rules at each level. See the
[minimal filter](examples/minimal-example/filter.yaml).

**The date range selects searches by their search time.** `ClicksClassifier`
counts accepted clicks on those searches, even if the clicks happened after
`to`. A click on a search before `from` does not count. `export clicks` uses
the same search-time range.

```bash
uv run longeval-cli classify --classifier FILE [--filter FILTER.yaml] --database-url URL
```



### 3. Train and score a click model

CLAX can train from query sessions streamed from the database
(`loaders_from_db`) or from a frozen [`export clicks`](#export-clicks) directory
(`loaders_from_file`). Each query session contains the displayed documents and
their binary click flags. The loaders split sessions into training and
validation sets and return a `QueryDocIdMap` with stable IDs for each
query–document pair. This map must fit in memory and be saved with the model.

See [training from the database](examples/minimal-example/train_dctr_from_db.py)
or [training from an export](examples/minimal-example/train_dctr_from_export.py)
for complete code. `src/clicks/scoring.py` scores the selected pairs and writes
a Parquet file with `query_id`, `doc_id`, `score`, and `query_sessions` columns.
That file feeds `export qrels` and `export snapshot`.


### 4. TODO: Document Collection

Sample, fetch and save the full documents referenced by the selected queries' results, for inclusion in the exported snapshot.

### 5. Packaging

Four exporters save intermediate and final results. Text outputs are gzipped
when their name ends in `.gz`.

| command           | input                         | output                                      | used for                                    |
| ----------------- | ----------------------------- | ------------------------------------------- | ------------------------------------------- |
| `export queries`  | filter YAML + db              | `query_id<TAB>query` TSV                    | manual review                               |
| `export clicks`   | filter YAML + db              | query-session shards + `manifest.json`      | training a click model without the database |
| `export qrels`    | scores file                   | TREC qrels                                  | trying discretizers on one scores file      |
| `export snapshot` | filter YAML + db + scores file | `<snapshot>/` with queries, qrels, manifest | the release                                 |

#### `export queries`

```bash
uv run longeval-cli export queries filter.yaml --output queries.tsv --database-url URL
```

Writes the queries passing the filter's `queries:` rules and prints how many remain after each rule. The TSV doubles as the review sheet: add a `manual_label` column, then feed it back with `ManualLabelsClassifier(path)` via `classify` and drop the labelled queries with `queries: {exclude: {manual: remove}}`.

#### `export clicks`

```bash
uv run longeval-cli export clicks exports/snapshot-1 --filter filter.yaml \
  [--shard-size 100000] [--batch-size 500] --database-url URL
```

Writes a frozen copy of the snapshot's query sessions (every displayed item with a binary click flag; the filter's `click_types` and rules decide what counts as a click, `from`/`to` bound the snapshot) as `query-sessions-NNNNN.jsonl.gz` shards plus a `manifest.json` (format version, selection, click types, counts, skipped records, per-shard SHA-256; no database URL). The manifest is written last; `loaders_from_file` checks its version, record counts and checksums. A directory that already holds an export is refused.

#### `export qrels`

```bash
uv run longeval-cli export qrels scores.parquet --output qrels.trec.gz \
  [--discretizer threshold|chapelle|any-click] [--score-column score]
```

Writes TREC qrels (`query_id 0 doc_id label`) from a scores file. Without `--discretizer` the scores themselves are written as labels. The built-in discretizers:

| discretizer  | label                                                                 | options                              |
| ------------ | --------------------------------------------------------------------- | ------------------------------------ |
| `threshold`  | number of thresholds the score reaches                                | `--thresholds 0.1,0.9` (default)     |
| `chapelle`   | grade `g = log2(P * 2**gmax + 1)`, rounded and capped at `gmax` (Chapelle et al., 2009) | `--gmax 2`, `--rounding nearest\|floor` |
| `any-click`  | 1 if the score (a click count, e.g. `--score-column clicks`) is > 0, else the pair is left out | –                |

When discretizing, queries without any label above 0 are dropped unless `--keep-all-queries` is set.

#### `export snapshot`

```bash
uv run longeval-cli export snapshot filter.yaml release/ \
  --scores scores.parquet --discretizer chapelle --database-url URL
```

Packages one snapshot for release into `release/<snapshot>/` (named by the filter's `snapshot`). It takes the same discretizer options as `export qrels`, but `--discretizer` is required. Only the queries with qrels are kept, and qrels for queries the filter drops are refused. The exported snapshots can be loaded directly with `ir_datasets_longeval`:
```
|- <snapshot>
|    |- qrels.trec.gz      # as `export qrels` writes it
|    |- queries.tsv.gz     # as `export queries`, only the queries with qrels
|    |- manifest.json      # selection, scores file + discretizer, counts, SHA-256 per file
|    L  documents          # TODO: once documents are scraped (step 4)
|          L documents_0001.jsonl.gz
|          L ...
|- <snapshot>
|    |- ...
```
