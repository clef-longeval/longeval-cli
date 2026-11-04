---
name: longeval-log-parser
description: Write, adapt or debug a `--parser` file for longeval-cli's `import-queries`/`import-clicks`, so a new search-log format (JSONL, CSV/TSV, other keys, other id schemes) can be imported into the LongEval database. Use this whenever someone has raw search or click logs that longeval-cli should ingest, asks how to support a different log format, sees many rejected records or "missing documents"/"unresolved" counts after an import, wants different query normalization (query_id/dedup), or mentions BaseParser, LogParser, SearchLogEntry or InteractionLogEntry, even if they never say "parser".
---

# Writing a longeval-cli log parser

`longeval-cli import-queries` and `import-clicks` know nothing about any log
format. Each call takes `--parser FILE`: a Python file with a top-level
`parser` object that turns raw lines into normalized entries. Your job is to
write that file for the user's logs, prove it on real lines *before* touching
the database, and then run the import.

Run all commands from the repository root (the parser file imports `src`).

## 1. Look at the logs first

Get real lines before writing code; the format usually has surprises
(placeholder ids like `undefined`, missing keys, NUL bytes, milliseconds vs.
seconds). Inputs can be plain files, `.gz`, or multi-member `.tar.gz`
archives, or directories of them. `zcat`/`tar -xzOf ... | head` or:

```bash
uv run python -c "
from pathlib import Path; import itertools
from src.importing.sources import discover_inputs, stream_source
for p in discover_inputs([Path('data/raw/logs')])[:1]:
    for l in itertools.islice(stream_source(p), 5): print(l.text)"
```

Work out, together with the user if unclear:

- How a **search** line (query session) and an **interaction** (click) line 
  are told apart.
- The **search id** both kinds share. Interactions are linked to searches
  only through it, so both passes must extract it identically.
- **Session/user id** (fallback when absent: something unique per search such
  as `anonymous:<search_id>`, so anonymous searches aren't merged).
- **Timestamp** encoding (epoch s/ms, ISO string, timezone).
- The **ranked document ids** shown for a search, and the **document id** and
  **interaction type** of a click.

## 2. What the parser must produce

From `src/importing/records.py`:

```python
SearchLogEntry(external_search_id: str, session_id: str, query: str,
               document_ids: tuple[str, ...],  # ranked, best first
               timestamp: datetime)            # timezone-aware (UTC)
InteractionLogEntry(external_search_id: str, document_id: str,
                    timestamp: datetime,
                    interaction_type: str | None = None,
                    observed_session_id: str | None = None)
```

Every line ends up as `SourceRecord(line, entry)` or
`RejectedRecord(line, "short-reason")`. **Never raise on a bad line.** Reject
it with a specific kebab-case reason: the importer counts rejections, and
`--strict`/`--max-errors` use them. A raised exception would abort a
multi-hour import on one bad line. Specific reasons such as
`undefined-search-id` or `invalid-timestamp` tell the user what is wrong with
their data. A catch-all reason doesn't.

Ids are strings. Convert numeric document ids with `str()` so search
rankings and clicks match. A click whose document id is not in the search's
`document_ids` is counted as a "missing document" and not stored.

## 3. Write it: subclass `BaseParser` (JSON lines)

`src/importing/parser.py:BaseParser` already does JSON decoding, dispatch
between the two passes, and `query_id`. For JSON-per-line logs, implement
four hooks. The reference implementation is
`examples/minimal-example/parser.py` (the real LongEval-Sci format). Read it
and copy its structure:

```python
from src.importing.parser import BaseParser, parse_timestamp
from src.importing.records import (InteractionLogEntry, ParseResult,
    RejectedRecord, SearchLogEntry, SourceRecord)
from src.importing.sources import SourceLine

class MyParser(BaseParser):
    def _is_search(self, raw: dict) -> bool: ...        # e.g. "search_id" in raw
    def _is_interaction(self, raw: dict) -> bool: ...   # e.g. "trackId" in raw
    def _parse_search(self, line: SourceLine, raw: dict) -> ParseResult: ...
    def _parse_interaction(self, line: SourceLine, raw: dict) -> ParseResult: ...

parser = MyParser()   # the loader looks for exactly this name
```

Things that are easy to get wrong:

- `parse_timestamp` only reads **epoch seconds** and returns UTC. For
  milliseconds, divide by 1000. For ISO strings, use
  `datetime.fromisoformat` and attach/convert to UTC. A naive datetime breaks
  date-range filtering later.
- `iter_searches` rejects lines matching neither `_is_search` nor
  `_is_interaction` as `unknown-record-shape`. Make the two predicates cover
  every line kind the logs really contain, or accept that count.
- `iter_interactions` silently skips non-interaction lines, so a broken
  `_is_interaction` shows up as **0 interactions**, not as rejections.
- Strip `\x00` from text fields (PostgreSQL rejects NUL bytes). The example
  has a `clean_text` helper for this.
- To ignore some interaction types at import time, pass
  `accepted_interaction_types=frozenset({...})` to the constructor. Usually
  it is better to import every type and choose clicks later with the filter's
  `click_types`, which keeps that decision changeable without re-importing.

## 4. Non-JSON logs (CSV/TSV, other)

`BaseParser._decode` expects one JSON object per line. Either override
`_decode` (a static method returning a `dict` or `None`) to parse the line
into a dict and keep the four hooks, or implement the `LogParser` protocol
directly (`iter_searches`, `iter_interactions`, `query_id`). Keep the
contracts: `iter_searches` yields only search results and rejections,
`iter_interactions` only interaction results and rejections, and each pass
skips the other kind's lines. Skip a CSV header line yourself (e.g. when
`line.record_number == 1`).

## 5. Query normalization

`query_id(text)` decides which searches count as the *same query*: verdicts,
click counts and qrels are all per query id. The default is
NFKC + whitespace collapse + casefold, then SHA-256 (`src/identity.py`).
Override `query_id` only if the user wants different grouping (e.g. also
stripping punctuation). Changing it after an import changes every id, so the
database needs re-importing (into a fresh database). Say so before changing
it on an existing database.

## 6. Dry-run before importing

`scripts/dry_run_parser.py` (in this skill's folder) runs both passes with
no database. It reports accepted and rejected counts per reason with example
lines, the interaction types, and how many interactions don't link to any
search id from pass 1:

```bash
uv run python skills/longeval-log-parser/scripts/dry_run_parser.py \
    --parser path/to/parser.py data/raw/logs --limit 200000
```

Iterate until the numbers make sense: rejections should be explainable and
there should be few unlinked interactions. Unlinked interactions are normal
when a click's search falls outside the sample, or when `--limit` cuts off
the end. They are a bug if almost all are unlinked. Show the user the report
and explain the remaining rejections.

If the repo has tests, add one like `tests/test_base_parser.py` for the new
format: feed a few `SourceLine`s and assert the entries and reasons.

## 7. Import

```bash
uv run longeval-cli init-db --database-url "$DB"
uv run longeval-cli import-queries LOGS... --parser FILE --database-url "$DB"
uv run longeval-cli import-clicks  LOGS... --parser FILE --database-url "$DB"
uv run longeval-cli database-status --database-url "$DB"
```

`$DB` is e.g. `sqlite:///work/longeval.sqlite3` or
`postgresql+psycopg://user:pw@host/db` (needs `uv sync --extra postgres`).
Without `--database-url`, `LONGEVAL_DATABASE_URL` or `./longeval.sqlite3` is
used, so be explicit.

- Run `import-queries` first. `import-clicks` resolves clicks via the
  searches already imported.
- Both are idempotent: re-running skips what's already there, so new log
  files can be appended later.
- Read the summary. `conflicts` means the same search id with different
  content (the first occurrence wins). `missing documents` means a click on a
  document that wasn't in that search's ranking. `rejected records` means
  your parser's rejections.
- `--limit N` gives a quick trial import. `--strict` aborts on the first
  rejection, which is useful while debugging.
