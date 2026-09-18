# Minimal Example: Data Formats and Pipeline Walkthrough

A tiny, self-contained dataset that shows what data looks like at every
stage of the `longeval-cli` pipeline: the raw log input format, the
database schema it's imported into, and the labels a classifier attaches
to it. Every command below was run against exactly this data to produce
the shown output.

## 1. Raw log input format

`logs/day1.jsonl` and `logs/day2.jsonl` are plain JSONL files (one JSON
object per line) in the format understood by
[`scripts/longeval-sci-v2/parser.py`](../../scripts/longeval-sci-v2/parser.py)
— see `src/importing/parser.py`/`src/importing/records.py` for the
format-agnostic contract any custom `--parser` file implements.

A line is either a **search event** or an **interaction event**,
distinguished by which key it has:

**Search event** — has `search_id`:

```json
{"search_id": "1001", "query": "climate change adaptation", "serp": ["D100", "D101", "D102"], "date": 1700000000, "uid": "user-a"}
```

| field       | meaning                                                          |
| ----------- | ----------------------------------------------------------------|
| `search_id` | the log's own id for this search, later resolved to a result row |
| `query`     | the raw query text                                               |
| `serp`      | ranked list of document ids returned for the query               |
| `date`      | Unix-epoch timestamp                                             |
| `uid`       | session/user id (falls back to `anonymous:<search_id>` if empty) |

**Interaction (click) event** — has `trackId`:

```json
{"trackId": "1001-D101", "type": "click", "date": 1700000042, "uid": "user-a"}
```

| field      | meaning                                                              |
| ---------- | --------------------------------------------------------------------|
| `trackId`  | `"<search_id>-<document_id>"`, links the click back to its search   |
| `type`     | interaction type, e.g. `click`, `dwell`                             |
| `date`     | Unix-epoch timestamp                                                |
| `uid`      | observed session/user id                                            |

Both event kinds live in the same files here; `import-queries` and
`import-clicks` each silently skip the lines that don't belong to their
pass (see `BaseParser.iter_searches`/`iter_interactions`).

## 2. Step-by-step pipeline

```bash
export DB=sqlite:////tmp/minimal-example.sqlite3

# 1. Create the schema
uv run longeval-cli init-db --database-url "$DB"

# 2. Import search events (pass 1)
uv run longeval-cli import-queries data/minimal-example/logs \
  --parser scripts/longeval-sci-v2/parser.py --database-url "$DB"

# 3. Import click/interaction events (pass 2) — run after import-queries
uv run longeval-cli import-clicks data/minimal-example/logs \
  --parser scripts/longeval-sci-v2/parser.py --database-url "$DB"

# 4. Classify every stored query with a couple of built-in classifiers
uv run longeval-cli classify-queries \
  --classifier "src.classifying.builtins:IsBooleanClassifier" --database-url "$DB"
uv run longeval-cli classify-queries \
  --classifier "src.classifying.builtins:WordCountClassifier" --database-url "$DB"

# 5. Check row counts
uv run longeval-cli database-status --database-url "$DB"
```

Expected `database-status` output:

```
Database: sqlite:////tmp/minimal-example.sqlite3
  sessions: 2
  queries: 3
  results: 3
  interactions: 4
  query classifications: 6
  interaction classifications: 0
```

## 3. Resulting database format

Querying the SQLite file directly (`sqlite3 /tmp/minimal-example.sqlite3`)
shows what each table (`src/models.py`) looks like after the import:

**`queries`** — one row per unique, deduplicated query text (`id` is a
stable hash of the normalized query):

```
id                                                                query
bcfc7306fa90718a41035524f34c3dee970e1022e05bf400f0ea67637a7d5bae climate change adaptation
e677e4c7d54a7a0560db5679c70686b5066f693121d1fca5ff936ff397a09c0c "machine learning" AND "healthcare"
35d1f9e2b7ed0f4d79215648c40816a9f1212419cc278126dd36262a83a1a5fa renewable energy storage
```

**`sessions`** — one row per observed `uid`:

```
id      start                      end                        system_ranking
user-a  2023-11-14 22:13:20.000000 2023-11-15 22:14:35.000000 NULL
user-b  2023-11-14 22:15:00.000000 2023-11-14 22:15:30.000000 NULL
```

**`results`** — one row per search event; `items` is a JSON map of result
rank to document id:

```
id  session_id  query_id                                                           q_date                      items
1   user-a      bcfc7306fa90718a41035524f34c3dee970e1022e05bf400f0ea67637a7d5bae  2023-11-14 22:13:20.000000  {"1": {"docid": "D100"}, "2": {"docid": "D101"}, "3": {"docid": "D102"}}
2   user-b      e677e4c7d54a7a0560db5679c70686b5066f693121d1fca5ff936ff397a09c0c  2023-11-14 22:15:00.000000  {"1": {"docid": "D200"}, "2": {"docid": "D201"}}
3   user-a      35d1f9e2b7ed0f4d79215648c40816a9f1212419cc278126dd36262a83a1a5fa  2023-11-15 22:13:20.000000  {"1": {"docid": "D300"}, "2": {"docid": "D301"}, "3": {"docid": "D302"}, "4": {"docid": "D303"}}
```

**`interactions`** — one row per click/interaction event, linked to the
`results` row its document appeared in:

```
id  session_id  result_id  document_id  type
1   user-a      1          D101         click
2   user-b      2          D200         click
3   user-a      3          D302         click
4   user-a      3          D300         dwell
```

**`query_classifications`** — one row per `(query_id, classifier)`:

```
query_id (truncated)  classifier   label
bcfc7306...            is-boolean   not-boolean
e677e4c7...            is-boolean   boolean
35d1f9e2...            is-boolean   not-boolean
bcfc7306...            word-count   3
e677e4c7...            word-count   4
35d1f9e2...            word-count   3
```

`search_references`/`interaction_references` are internal bookkeeping
tables (search id → result row, and "have I already imported this exact
line") — see the docstrings in `src/models.py` for why they exist.

## 4. What's not covered here yet

Relevance judgment (qrels) creation, document scraping, and snapshot
packaging (`qrels.trec.gz` / `queries.tsv.gz` / `documents/*.jsonl.gz`)
are still `TODO` in the main [README](../../README.md) and have no
concrete format to show yet.
