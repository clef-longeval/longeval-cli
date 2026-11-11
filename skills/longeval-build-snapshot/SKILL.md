---
name: longeval-build-snapshot
description: Guide a user through building and releasing a new snapshot for a dynamic test collection with longeval-cli, end to end or from wherever they are init-db, import-queries/import-clicks, classify, filter YAML, export clicks, training and scoring a CLAX click model, export qrels with a discretizer (threshold/chapelle/any-click), collect-document-candidates/select-documents/fetch-documents, export snapshot, and retrieve baselines. Use this whenever someone wants to create, rebuild, extend or release a snapshot or qrels from search logs, asks "what's the next step" or "how do I get from logs to a release", picks a discretizer, adds a new snapshot to an existing database, or hits an error in one of these commands.
---

# Building a snapshot with longeval-cli

The pipeline turns search logs into a new snapshot for a dynamic test collection: 
queries, qrels inferred from clicks, and a document collection. Each stage 
persists its results (in the database or in files), so the user can enter anywhere.
**First find out where they are**: run `database-status`, look for existing
filter, classifier, parser and scores files, and ask what they already have.
Then continue from there instead of restarting.

`examples/minimal-example/README.md` runs every step on a small synthetic
dataset and lists the expected numbers. When the user is unsure what a step
should print, point them to it, or run it in a scratch database to compare.

Run commands from the repository root. Pass `--database-url "$DB"`
explicitly every time. The fallback is `./longeval.sqlite3` if not specified in `LONGEVAL_DATABASE_URL`.

```bash
export DB=sqlite:///work/longeval.sqlite3      # or postgresql+psycopg://...
```

## The stages

```
logs ─import─▶ db ─classify─▶ verdicts ─filter.yaml─▶ click sessions ─CLAX─▶ scores.parquet
                                                                               │
                         release/<snapshot>/ ◀─export snapshot─ qrels ◀─discretize
                                    ▲
           documents: collect candidates ─▶ select ─▶ fetch
```

### 0. Setup
Outside a clone of the repository, install from PyPI instead
(`pip install "longeval-cli[clax]"`, extras `clax`/`postgres`/`tira`, and
drop the `uv run` prefix below if not using uv). In a clone:
```bash
uv sync                      # base
uv sync --extra clax         # click-model training/scoring (JAX, CLAX)
uv sync --extra postgres     # if $DB is PostgreSQL
uv sync --extra tira         # only for `retrieve`
uv run longeval-cli init-db --database-url "$DB"
```

### 1. Import logs (needs a parser file)
```bash
uv run longeval-cli import-queries data/raw/logs --parser PARSER.py --database-url "$DB"
uv run longeval-cli import-clicks  data/raw/logs --parser PARSER.py --database-url "$DB"
```
Queries come first. Both commands are idempotent, so new log files can be
added later. Report `conflicts`, `rejected records` and `missing documents`
to the user. For a new log format, use the `longeval-log-parser` skill. The
LongEval-Sci format is `examples/longeval-sci-v2/parser.py`. Make sure queries 
are normalized to the liking of the user and matching the production system. 


### 2. Snapshot filter + classifiers
Write `filter.yaml` (snapshot name, `[from, to)` search window,
`click_types`, rules). Then run the classifiers its rules name.
Snapshot-independent ones can run globally. Range-dependent ones need
`--filter`, and aggregates (`clicks`) come last:

```bash
uv run longeval-cli classify --classifier text_classifiers.py --database-url "$DB"
uv run longeval-cli classify --classifier snapshot_classifiers.py --filter filter.yaml --database-url "$DB"
uv run longeval-cli export queries filter.yaml --output review.tsv --database-url "$DB"
```

`export queries` prints the query funnel rule by rule. It is the place to
tune the thresholds and to run a manual review (add `manual_label`, feed it
back with `ManualLabelsClassifier`, and `exclude: {manual: remove}`). Details:
the `longeval-snapshot-filter` and `longeval-classifier` skills. Before
training, run the filter checker
(`skills/longeval-snapshot-filter/scripts/check_filter.py`): a rule whose
classifier never ran fails silently.

### 3. Click data, model, scores
The model is fitted in user code. The CLI only provides the data and later
consumes the scores. Two equivalent inputs:

- **From the database**: `loaders_from_db(session, config, ...)`. See
  `examples/minimal-example/train_dctr_from_db.py`.
- **From a frozen export** (reproducible, no database needed):
  ```bash
  uv run longeval-cli export clicks exports/snapshot-2-clicks --filter filter.yaml --database-url "$DB"
  ```
  Then `loaders_from_file`. See `train_dctr_from_export.py`. The directory
  must not already hold an export.

The example scripts load `filter.yaml` **from their own folder**. Copy one
and point it at the user's filter (and parameters) rather than running it
unchanged on real data. The output must be a Parquet file with `query_id`,
`doc_id` and `score` (`src/clicks/scoring.py:score_candidates` scores a
trained model over the sessions). Keep the scores file. It can be
discretized many ways without retraining.

No model is needed for raw click-based qrels: a scores file with a click
count column plus `--discretizer any-click --score-column clicks` gives
binary "was clicked" qrels.

### 4. Qrels: choose a discretizer
```bash
uv run longeval-cli export qrels scores.parquet --output qrels.trec.gz \
    --discretizer threshold --thresholds 0.1,0.3
```
- `threshold`: the label is the number of thresholds reached
  (`0.1,0.3` gives labels 0/1/2).
- `chapelle`: `round(log2(P·2^gmax + 1))` capped at `--gmax` (default 2),
  following Chapelle et al. 2009.
- `any-click`: label 1 if the score is > 0, otherwise the pair is omitted.

When discretizing, queries with no label above 0 are dropped
(`--keep-all-queries` keeps them). Try a few settings and show the user the
label distribution, e.g.
`zcat qrels.trec.gz | awk '{print $4}' | sort | uniq -c`. Stage 5 needs
qrels; stage 6 re-derives them from the same scores and settings.

### 5. Documents (optional, needed for a full release)
Needs a document source file: a top-level `source`, a subclass of
`src.documents.source.BaseDocumentSource` with `fetch_batch`. Examples:
`examples/minimal-example/document_source.py` and
`examples/longeval-sci-v2/core_source.py`.
```bash
uv run longeval-cli collect-document-candidates filter.yaml --source SOURCE.py --qrels qrels.trec.gz --database-url "$DB"
uv run longeval-cli select-documents filter.yaml --qrels qrels.trec.gz [--extends snapshot-1] --seed 42 --database-url "$DB"
uv run longeval-cli fetch-documents --source SOURCE.py --snapshot snapshot-2 --database-url "$DB"
uv run longeval-cli export documents filter.yaml exports/snapshot-2-docs --database-url "$DB"   # optional check
```
- The `published` pool needs `published_from`/`published_to` in the filter
  and `iter_published_ids` on the source. Without them, use
  `--only negative`.
- `--extends <previous snapshot>` keeps the collection growing across
  snapshots. Use it for every snapshot after the first.
- `fetch-documents` can be resumed. Rerun it after an interruption.
  `--max-workers` parallelizes the requests, and `--retry-not-found` is for
  after switching sources.
- Document rules (e.g. `exclude: {abstract-length: short}`) need a
  `classify` run *after* fetching, because documents have to exist to be
  classified.

### 6. Export
```bash
uv run longeval-cli export snapshot filter.yaml release/ \
    --scores scores.parquet --discretizer threshold --thresholds 0.1,0.3 \
    --database-url "$DB"
```
This writes `release/<snapshot>/` with `qrels.trec.gz`, `queries.tsv.gz`
(only the queries with qrels), `manifest.json`, and `documents/`, in the
LongEval snapshot format v2. The filter needs `from`, `to` and `period`
(`to` = `from` + `period`), and the snapshot has to fit the release's time grid
in `release/` (same `period`, no overlap). Use the **same discretizer settings** as stage 4. Otherwise
the released qrels differ from the ones the documents were selected for.

Expected refusals and what they mean:
- **Existing `release/<snapshot>/`**: it never overwrites. Pick a new root
  or remove the old one deliberately (ask the user first).
- **Qrels for queries the filter drops**: the scores were trained for a
  different selection. Retrain on this filter, or fix the filter.
- **Qrels documents missing from the collection**:
  fetch them, or use `--drop-missing-documents` to omit those pairs (they
  are counted in the manifest).

### 7. Retrieval baselines (optional)
```bash
uv run longeval-cli retrieve release/snapshot-2 work/retrieval --model BM25 --model PL2
```
This needs Docker and `tira-cli verify-installation --local-only`. Existing
runs are reused, and `--recompute` rebuilds them.

## Adding a later snapshot to an existing database

1. Import the new logs (both passes) into the same database.
2. Copy the previous filter: give it a new `snapshot`, move `from`/`to` (the
   new `from` = the old `to`) and `published_*`, and keep the rules
   consistent unless the user wants changes.
3. Global verdicts carry over, but newly imported queries need the global
   classifiers re-run. Re-run the snapshot-scoped ones with the new
   `--filter`.
4. Train and score on the new snapshot's sessions. Select documents with
   `--extends <previous snapshot>`.

## When reporting back

After each stage, tell the user the key counts the command printed (queries
kept, clicks, qrels pairs and label distribution, documents fetched, not
found) and what they imply. A sudden zero almost always means a
filter/classifier mismatch, not a bug. Check it with the filter checker
before suspecting the code.
