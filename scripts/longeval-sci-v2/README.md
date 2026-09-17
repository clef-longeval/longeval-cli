# LongEval-Sci-v2

This is an example on how to recreate the LongEval-Sci-v2 dynamic test collection with the `longeval-cli`.

## 1. Log Ingestion
We import the logs from `data/logs` with this command (run from this directory, using the `parser.py` defined here):

```bash
uv run longeval-cli import-queries data/logs --parser parser.py --database-url sqlite:///longeval.sqlite3
uv run longeval-cli import-clicks data/logs --parser parser.py --database-url sqlite:///longeval.sqlite3
```

## 2. Query Classification

Each classifier is a separate run, and each writes its own row per query rather than overwriting the others. `is_boolean`/`word_count` are generic, so they're loaded directly by import spec rather than through a file; `has_doi`/`is_structured` are specific to CORE's query syntax and live alongside this parser:

```bash
uv run longeval-cli classify-queries --classifier src.classifying.builtins:IsBooleanClassifier --database-url sqlite:///longeval.sqlite3
uv run longeval-cli classify-queries --classifier src.classifying.builtins:WordCountClassifier --database-url sqlite:///longeval.sqlite3
uv run longeval-cli classify-queries --classifier classifiers/has_doi.py --database-url sqlite:///longeval.sqlite3
uv run longeval-cli classify-queries --classifier classifiers/is_structured.py --database-url sqlite:///longeval.sqlite3
```
