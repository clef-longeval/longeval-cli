# 📟 longeval-sci

Create dynamic test collections from search logs.

An example instantiation for the LongEval Sci Test Collection from 2026 is provided in `scripts/longeval-sci-v2`!

## Installation

```bash
uv sync
uv run longeval-cli init-db --database-url sqlite:///work/longeval.sqlite3
```

## Usage

### 1. Log Ingestion

Imports raw search logs into the database in two passes: first the search events (queries + their result rankings), then the click/interaction events, each linked back to the search it belongs to via the log's own search id. Run `import-queries` before `import-clicks`.

```bash
uv run longeval-cli import-queries PATHS... --parser FILE --database-url URL
uv run longeval-cli import-clicks PATHS... --parser FILE --database-url URL
```


### 2. Query Selection

Runs classifiers over every stored query (or interaction), labeling it for suitability. The classifiers provided detect boolean syntax, embedded DOIs, structured field queries, and so on. Each classifier is a separate run and writes its own label per query. This step builds up the signal a later filtering pass will use to pick the final query set.

```bash
uv run longeval-cli classify-queries --classifier SPEC --database-url URL
```

`--parser`/`--classifier` are dynamically loaded extension points, not built-in registries — see `src/importing/` and `src/classifying/`.

### 3. TODO: Relevance Label Creation

Derive relevance judgments (qrels) for the selected queries from their click data.

### 4. TODO: Document Scraping

Sample, fetch and save the full documents referenced by the selected queries' results, for inclusion in the exported snapshot.

### 5. TODO: Packaging

Export the test collection snapshots and intermediate components. The exported snapshot has this format:

```
|- <snapshot>
|    |- qrels.trec.gz
|    |- queries.tsv.gz
|    L  documents
|          L documents_0001.jsonl.gz
|          L documents_0002.jsonl.gz
|          L ...
|- <snapshot>
|    |- ...
```
