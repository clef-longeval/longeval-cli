# 📟 longeval-cli
[![Python](https://img.shields.io/pypi/pyversions/longeval-cli?style=flat-square&cacheSeconds=3600)](https://pypi.org/project/longeval-cli/)
[![PyPi](https://img.shields.io/pypi/v/longeval-cli?style=flat-square)](https://pypi.org/project/longeval-cli/)
[![CI status](https://img.shields.io/github/actions/workflow/status/clef-longeval/longeval-cli/ci.yml?branch=main&style=flat-square)](https://github.com/clef-longeval/longeval-cli/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/clef-longeval/longeval-cli?style=flat-square)](LICENSE)

Build dynamic test collections from search logs. `longeval-cli` turns raw search logs into a sequence of  self-contained snapshots, each with its own queries, relevance labels (qrels), and documents. It is introduced in our paper *Building Dynamic Test Collections: LongEval CLI and Scientific Retrieval Dataset* (under review) to which we refer for the methodological background.

## The pipeline

![The longeval-cli pipeline: 1. Ingest Logs → parsed logs, 2. Select Queries → queries, 3. Create Labels → qrels, 4. Fetch Docs. → docs, 5. Package → snapshot, 6. Retrieve → pool](docs/pipeline.png)

Each stage stores its results in the database or in files, so you can stop, inspect, and continue at any stage, and add a new snapshot without reprocessing the earlier ones.

| Stage | What you provide | Commands | Paper |
| --- | --- | --- | --- |
| [1. Ingest Logs](docs/reference.md#1-ingest-logs) | a **parser** for your log format | `import-queries`, `import-clicks` | §3.1 Log Ingestion |
| [2. Select Queries](docs/reference.md#2-select-queries-classify-and-filter) | **classifiers** and a **filter YAML** per snapshot | `classify`, `export queries` | §3.2 Query Selection and Interaction Filtering |
| [3. Create Labels](docs/reference.md#3-create-labels-train-and-score-a-click-model) | a CLAX click model (Python script) | `export clicks`, `export qrels` | §3.3 Pseudo Relevance Label Creation |
| [4. Fetch Docs](docs/reference.md#4-fetch-documents) | a **document source** | `collect-document-candidates`, `select-documents`, `fetch-documents` | §3.4 Document Collection |
| [5. Package](docs/reference.md#5-package) | – | `export snapshot` (and `export documents`) | §3.5 Packaging |
| [6. Retrieve](docs/reference.md#6-retrieve-baselines) | TIRA + Docker | `retrieve` | §3.6 Retrieval |

Everything that depends on your search engine (parser, classifiers, document
source) is a small Python file you pass on the command line, so the toolkit
itself stays format-independent. The exported snapshots can be loaded with
`ir_datasets_longeval`.

## Installation

```bash
# from a clone of this repository
uv sync                      # base install
uv sync --extra clax         # click-model training and scoring (JAX, CLAX)
uv sync --extra postgres     # if your database is PostgreSQL
uv sync --extra tira         # for `retrieve`
```

The database defaults to a local SQLite file. Pass `--database-url` (or set
`LONGEVAL_DATABASE_URL`) to use another file or PostgreSQL, e.g.
`postgresql+psycopg://user:password@host/database`.

## Quickstart

Run the first stages on the synthetic logs that ship with the repository:

```bash
export DB=sqlite:////tmp/minimal-example.sqlite3 E=examples/minimal-example

uv run longeval-cli init-db --database-url "$DB"
uv run longeval-cli import-queries $E/logs --parser $E/parser.py --database-url "$DB"
uv run longeval-cli import-clicks  $E/logs --parser $E/parser.py --database-url "$DB"
uv run longeval-cli classify --classifier $E/classifiers.py --filter $E/filter.yaml --database-url "$DB"
uv run longeval-cli database-status --database-url "$DB"
```

The [tutorial](examples/minimal-example/README.md) continues from here through
click-model training, qrels, documents, and the packaged snapshot, with the
expected output of every step.

## Documentation

| | |
| --- | --- |
| **[Tutorial](examples/minimal-example/README.md)** | the whole pipeline on a small synthetic dataset with known ground truth |
| **[Command reference](docs/reference.md)** | every command, its options and output formats, the filter YAML |
| **[LongEval Sci 2026 setup](examples/longeval-sci-v2/README.md)** | the configuration used for the collection in the paper (CORE logs) |
| **[Agent skills](skills/)** | step-by-step guides that let an AI agent (e.g. Claude Code) run the toolkit for you |

The agent skills double as how-to guides for the four tasks you will do most:

- [`longeval-log-parser`](skills/longeval-log-parser/SKILL.md): write a parser for a new log format
- [`longeval-classifier`](skills/longeval-classifier/SKILL.md): write or configure classifiers
- [`longeval-snapshot-filter`](skills/longeval-snapshot-filter/SKILL.md): write a snapshot's filter YAML
- [`longeval-build-snapshot`](skills/longeval-build-snapshot/SKILL.md): build and release a snapshot end to end

## Roadmap

- Query intent classification
- Query recency classification
- Topic generation
- Sampling instead of pooling
- Generalize filters.yml into a snapshot config file to reduce options needed

## License

MIT, see [LICENSE](LICENSE).
