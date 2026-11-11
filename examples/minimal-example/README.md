# Minimal Example: Data Formats and Pipeline Walkthrough

A small, self-contained, synthetic dataset that goes through every stage of the `longeval-cli` pipeline: raw logs, the database, classifiers, a snapshot filter, every export, a CLAX click model, qrels and the packaged snapshot. It is large enough for the click model to learn something (about 1000 searches, 1650 interactions). Because it is simulated, the true relevance of every document is known, so the result can
be checked. Every number below is what the commands produce on exactly this data, and CI checks them (`.github/workflows/minimal-example.yml`).

| file                    | what it is                                                        |
| ----------------------- | ----------------------------------------------------------------- |
| `logs/2024-11-0{4,5,6}.jsonl` | the raw search logs, one file per day                       |
| `classifiers.py`        | query, session (bot), interaction (late click) and click-count classifiers |
| `filter.yaml`           | snapshot 1: its date range, click types and rules per level       |
| `train_dctr_from_db.py` | fits CLAX's `DocumentCTRModel` from the database                  |
| `train_dctr_from_export.py` | the same from a frozen `export clicks` directory              |
| `document_source.py`    | an offline document source: made-up records for the judged documents |

## 1. Raw log input format

The logs are plain JSONL files. A line is either a **search event** or an **interaction event**, distinguished by which key it has:

**Search event**: has `search_id`.

```json
{"date": 1730707332, "search_id": "d57b41c9730cc5c42dc5f7cc5b786d16", "serp": [87701757, 43053556, 10108819, ...], "query": "social media use and adolescent mental health", "uid": "e0b4341c457b"}
```

| field       | meaning                                                          |
| ----------- | ---------------------------------------------------------------- |
| `search_id` | the log's own id for this search, later resolved to a result row |
| `query`     | the raw query text                                               |
| `serp`      | ranked list of document ids returned for the query               |
| `date`      | Unix-epoch timestamp                                             |
| `uid`       | session/user id (falls back to `anonymous:<search_id>` if absent) |

**Interaction event**: has `trackId`.

```json
{"date": 1730708184, "trackId": "d6e3411047e87c8d0a1e95359a9be3b8-76955905", "type": "data-provider"}
```

| field      | meaning                                                            |
| ---------- | ------------------------------------------------------------------ |
| `trackId`  | `"<search_id>-<document_id>"`, links the event back to its search  |
| `type`     | what was opened: `works` (record page), `get-pdf`, `author`, `data-provider` |
| `date`     | Unix-epoch timestamp                                               |

These are the real LongEval-Sci interaction types. Only `works` and `get-pdf` count as a *click* here (`click_types` in `filter.yaml`); `author` and `data-provider` events are imported but ignored by the click model.

Both event kinds live in the same files. `import-queries` and `import-clicks` each skip the lines that don't belong to their pass.


## 2. Step-by-step pipeline

```bash
export DB=sqlite:////tmp/minimal-example.sqlite3
export E=examples/minimal-example
export PARSER=examples/minimal-example/parser.py

# 1. Create the schema
uv run longeval-cli init-db --database-url "$DB"

# 2. Import search events (pass 1), then interaction events (pass 2)
uv run longeval-cli import-queries $E/logs --parser $PARSER --database-url "$DB"
uv run longeval-cli import-clicks $E/logs --parser $PARSER --database-url "$DB"

# 3. Classify: queries, sessions and interactions once
uv run longeval-cli classify --classifier $E/classifiers.py --filter $E/filter.yaml --database-url "$DB"

# 4. Check row counts
uv run longeval-cli database-status --database-url "$DB"

# 5. Export the queries passing the filter (also the manual-review sheet)
uv run longeval-cli export queries $E/filter.yaml --output /tmp/minimal-queries.tsv --database-url "$DB"

# 6. Fit a click model (needs `uv sync --extra clax`), turn its per-pair click
#    probabilities into qrels, and compare both with the ground truth
uv run python $E/train_dctr_from_db.py --database-url "$DB" --output /tmp/minimal-scores.parquet

uv run longeval-cli export qrels /tmp/minimal-scores.parquet \
  --output /tmp/minimal-qrels.txt --discretizer threshold --thresholds 0.1,0.3
uv run python $E/check_scores.py /tmp/minimal-scores.parquet --qrels /tmp/minimal-qrels.txt

# 7. Or save the query sessions once and train from that copy, with no database
uv run longeval-cli export clicks /tmp/minimal-clicks --filter $E/filter.yaml --database-url "$DB"
uv run python $E/train_dctr_from_export.py /tmp/minimal-clicks --output /tmp/minimal-scores-export.parquet

# 8. Collect the documents: candidates, selection, download
uv run longeval-cli collect-document-candidates $E/filter.yaml --source $E/document_source.py \
  --qrels /tmp/minimal-qrels.txt --only negative --database-url "$DB"
uv run longeval-cli select-documents $E/filter.yaml --qrels /tmp/minimal-qrels.txt \
  --seed 42 --database-url "$DB"
uv run longeval-cli fetch-documents --source $E/document_source.py --snapshot snapshot-1 \
  --database-url "$DB"
uv run longeval-cli export documents $E/filter.yaml /tmp/minimal-documents --database-url "$DB"

# 9. Package the snapshot for release: queries + qrels + documents + manifest
uv run longeval-cli export snapshot $E/filter.yaml /tmp/minimal-release \
  --scores /tmp/minimal-scores.parquet --discretizer threshold --thresholds 0.1,0.3 \
  --database-url "$DB"

# 10. Index the released documents and retrieve with two models (Docker, via TIRA)
uv run longeval-cli retrieve /tmp/minimal-release/snapshot-1 /tmp/minimal-retrieval \
  --model BM25 --model PL2
```

### 2.1 Log Ingestion
The imports report the imperfections:

```
import-queries:  searches seen: 1032, conflicts: 1, known mappings: 1031
import-clicks:   interactions seen: 1653, rejected records: 2, missing documents: 1
```

The conflict is the duplicate `search_id`; the first occurrence wins. The two rejected records are the `undefined` trackIds, and the missing document is the click on a document that wasn't shown.

### 2.2 Query Selection and Interaction Filtering
Every classifier labels the rows of one table, its *level*: `queries`, `sessions`, `results` or `interactions`. Its verdicts go to that level's `*_classifications` table, and `filter.yaml` applies them in the section of the same name. One file may mix levels. `classify` reports each one:

```
is-boolean: 18 verdicts (query_classifications, snapshot 'snapshot-1')
word-count: 18 verdicts (query_classifications, snapshot 'snapshot-1')
bot: 1 verdicts (session_classifications, snapshot 'snapshot-1')
click-delay: 1 verdicts (interaction_classifications, snapshot 'snapshot-1')
clicks: 14 verdicts (query_classifications, snapshot 'snapshot-1')
```

With `--filter`, every verdict is stored under the filter's snapshot. A filter also accepts global verdicts (snapshot empty), so with several snapshots you could run the snapshot-independent classifiers (`is-boolean`, `word-count`, `click-delay`) once without `--filter` instead.

`bot` and `click-delay` store only the exceptions: the one bot session and the one late click. Everything else has no verdict, which an `exclude` rule keeps. Classifiers run in list order, and `clicks` comes last because it counts what the filter calls a click, so the bot's clicks and the late click don't count. The 14 queries with counted clicks are the 12 topical and the 2 boolean ones; the bot's `free pdf download` gets none.

Expected `database-status` output:

```
  sessions: 42
  queries: 18
  results: 1031
  interactions: 1652
  query classifications: 50
  session classifications: 1
  result classifications: 0
  interaction classifications: 1
```

That is 40 users, the bot and one anonymous session; 12 + 2 + 2 queries, the bot's query and the empty search; 2 × 18 text verdicts plus 14 click counts.

`filter.yaml` is the config of snapshot 1: its name, its half-open UTC date range `[2024-11-04, 2024-11-06)` (day 3 is outside), which interaction types are clicks, and one section of rules per level.

```yaml
snapshot: snapshot-1
from: 2024-11-04
to: 2024-11-06
click_types: [works, get-pdf]
queries:
  require: {is-boolean: 0}
  min: {clicks: 1}
sessions:
  exclude: {bot: bot}
interactions:
  exclude: {click-delay: late}
```

Every level takes the same `require`/`min`/`exclude` rules. A query, session or result that fails them is left out together with its searches; an interaction that fails them is not a click.

### 2.3 Relevance Label Creation

[`train_dctr_from_db.py`](train_dctr_from_db.py) streams the snapshot's query sessions from the database (`loaders_from_db`, every 5th one held out for validation) and fits `DocumentCTRModel`: one click probability per query–document pair, written as Parquet (`query_id`, `doc_id`, `score`). Training stops early when the validation loss stops improving (after about 20 epochs here, a few seconds on a CPU). [`train_dctr_from_export.py`](train_dctr_from_export.py) does the same from the `export clicks` directory and gives the same result.

`export qrels --discretizer threshold --thresholds 0.1,0.3` labels a `score` below 0.1 as 0, below 0.3 as 1, and from 0.3 up as 2. That gives 180 qrels (12 queries × 15 documents): 135 × 0, 28 × 1, 17 × 2.

`check_scores.py` compares the result with `ground_truth.tsv`:

```
grade 0:  133 pairs, mean score 0.028
grade 1:   27 pairs, mean score 0.172
grade 2:   20 pairs, mean score 0.395
qrels label == grade for 96% of 180 pairs
```

Without the `sessions` rule, the bot's clicks on ranks 1–5 make unrelated documents look relevant: agreement drops to 54% and the check fails.

A DCTR estimate is the click rate averaged over the positions a document was shown at, so it sits below the attractiveness (0.85 for grade 2). Early stopping shrinks it a little further. The order of the grades is what matters. With `--strict` (as in CI) the script fails if the mean `score` does not rise with the grade or if fewer than 85% of qrels labels match. Exact `score` values vary slightly between machines; these checks do not.


### 2.4 Document Collection
The logs only hold document ids. [`document_source.py`](document_source.py) stands in for a provider such as CORE: a `BaseDocumentSource` whose `fetch_batch` returns a made-up record (title, abstract, publication date) for each document of `ground_truth.tsv`, and nothing for any other id. It is offline, so the example needs no API key.

Collection runs in three steps, all per snapshot:

```
collect-document-candidates ... --only negative
  0  negative candidates
select-documents ...
       180  relevant
         0  published (of 90 wanted: too few candidates)
         0  negative (of 90 wanted: too few candidates)
Selected 180 documents for snapshot snapshot-1
fetch-documents ...
       180  ids requested
       180  fetched
         0  not found
```

- `collect-document-candidates` stores the pools to sample from. The example source does not list documents by publication date, so `--only negative` skips the `published` pool (it also needs `published_from`/`published_to` in the filter). The `negative` pool is every document the snapshot's searches displayed that has no qrels, and here there is none: all 180 displayed documents are judged.
- `select-documents` takes every qrels document (180) plus seeded samples of the pools, which are empty here.
- `fetch-documents` downloads the selected documents through the source. Ids already fetched are skipped, so running it again fetches nothing (`180 skipped`).

`export documents` writes the collection as gzipped JSONL shards (`documents_000000.jsonl.gz`, 180 lines) with a `manifest.json`, and `export snapshot` adds it to the release, where every qrels document has to be in the collection.

### 2.5 Packaging
The exports and what they are for:

| command           | output                                   | used for                                  |
| ----------------- | ---------------------------------------- | ----------------------------------------- |
| `export queries`  | `query_id<TAB>query` TSV                 | manual reviewing                          |
| `export clicks`   | query-session shards + `manifest.json`   | training a click model without the database |
| `export qrels`    | TREC qrels from a scores file            | trying discretizers on one scores file    |
| `export documents` | the snapshot's fetched documents, as shards + `manifest.json` | checking the collection, building indices |
| `export snapshot` | `<snapshot>/` with queries, qrels, (documents), manifest | the release                  |

Scoring stays outside the CLI (`train_dctr_from_*.py` write the scores Parquet), so one trained model can be discretized many ways without retraining or rescoring.

`export snapshot` runs the qrels, queries and documents writers into `/tmp/minimal-release/snapshot-1/` (the directory is named after the filter's `snapshot`):

```
       180  pairs read
       180  pairs written
        12  queries
        12  queries passing the filter
       180  documents
Wrote snapshot snapshot-1 to /tmp/minimal-release/snapshot-1
```

```
snapshot-1/
|- manifest.json     # selection, scores file (SHA-256), discretizer, counts, file SHA-256s
|- qrels.trec.gz     # the same 180 qrels as /tmp/minimal-qrels.txt
|- queries.tsv.gz    # the 12 queries, as `export queries` writes them
L  documents/        # the 180 documents, as `export documents` writes them
```

```
$ gunzip -c /tmp/minimal-release/snapshot-1/qrels.trec.gz | head -3
5c5942c5e7fd21bede75a46e225541707a4f6d92e83acc4d2f840628b2b59bff 0 87701757 0
5c5942c5e7fd21bede75a46e225541707a4f6d92e83acc4d2f840628b2b59bff 0 43053556 0
5c5942c5e7fd21bede75a46e225541707a4f6d92e83acc4d2f840628b2b59bff 0 10108819 2
```

Only queries with qrels are packaged. Here all 12 have some, but a query whose documents all get label 0 would be dropped from both files and reported as left out. If the scores held a query the filter drops (scores trained for another selection), nothing would be written. Running the command again refuses the existing `snapshot-1/`. Equal inputs give byte-identical files.

### 2.6 Retrieval
`retrieve` builds a PyTerrier index of the 180 released documents and retrieves the 12 released queries with BM25 and PL2, using the `tira-ir-starter` approaches through TIRA (Docker has to be usable, the first call pulls the images). It writes one TREC run per model, `/tmp/minimal-retrieval/runs/BM25/run.txt` and `.../PL2/run.txt`. CI checks that every query has results.
