---
name: longeval-classifier
description: Write and run classifiers for longeval-cli's `classify` command, which label queries, sessions, results, interactions or documents (spam, boolean syntax, DOIs, bots, late clicks, click counts, abstract length, manual review labels, ...) so a snapshot filter can select on them. Use this whenever a LongEval user wants to flag, label, score, detect or remove some kind of query/session/click/document, asks for a `--classifier` file, wants to use a built-in like ClicksClassifier/SearchBurstClassifier/ManualLabelsClassifier, feeds a reviewed queries TSV back in, or wonders why verdicts are missing or a classify run stored 0 verdicts.
---

# Writing and running longeval-cli classifiers

A classifier stores **verdicts** about the rows of one table. A snapshot
filter YAML then keeps or drops rows by those verdicts. Classifiers don't
filter anything themselves. They only label rows, so the filtering decision
stays in the YAML, where it can change without re-running anything.

```bash
uv run longeval-cli classify --classifier FILE [--filter snapshot.yaml] --database-url "$DB"
```

`FILE` is a Python file with a top-level `classifiers`: one classifier or a
list, and the levels may be mixed. Classifiers run in list order. Run
commands from the repository root. Examples to imitate:
`examples/minimal-example/classifiers.py` and
`examples/longeval-sci-v2/document_classifiers.py`.

## 1. Pick the level and the verdict shape

| level          | a verdict is about    | view a RowClassifier gets (`src/classifying/rows.py`) |
| -------------- | --------------------- | ------------------------------------------------- |
| `queries`      | a distinct query text | `QueryView(id, text)` |
| `sessions`     | a user session        | `SessionView(id, started_at, ended_at)` |
| `results`      | one displayed search  | `ResultView(id, query_id, query, session_id, queried_at, doc_ids)` |
| `interactions` | one click/interaction | `InteractionView(id, type, date, document_id, session_id, result_id, query_id, searched_at, rank)` |
| `documents`    | a fetched document    | `DocumentView(id, title, abstract, published_at, full_text_length, details)` |

A verdict is `Classification(label: str, confidence=None, value=None)`.
Choose the shape from the rule the user will write in the filter:

- **Yes/no property** → `Classification.flag(bool)`, which gives label
  `"1"`/`"0"` and value 1.0/0.0. Name the classifier after the property
  (`has-doi`). The filter then says `require: {has-doi: 0}`.
- **Category** → a label (`"bot"`, `"late"`, `"short"`). The filter says
  `exclude: {bot: bot}`.
- **Number to threshold** → set `value` (and the same number as `label`).
  The filter says `min: {clicks: 5}`. `min` reads only `value`.

**Exceptions-only vs. every row.** Returning `None` means "no opinion": no
row is stored. An `exclude` rule keeps rows without a verdict, but a
`require`/`min` rule drops them. So:

- For something rare you will `exclude` (bots, late clicks, spam): store
  only the exceptions. Tables stay small, and everything else passes.
- For something you will `require`: label **every** row (both `"0"` and
  `"1"`), or rows without a verdict are silently dropped.

## 2. Built-ins first

From `src.classifying` (wrap them in a file and set parameters in code):

| class | level | label / value |
| --- | --- | --- |
| `IsBooleanClassifier()` | queries | flag `is-boolean` (AND/OR/NOT) |
| `WordCountClassifier()` | queries | `word-count`: count as label and value |
| `ManualLabelsClassifier(path, name="manual")` | queries | the reviewer's `manual_label` column from an `export queries` TSV |
| `ClicksClassifier(name="clicks")` | queries | click count in the snapshot (label and value) |
| `SearchBurstClassifier("bot", searches=10, within_seconds=60)` | sessions | `bot` if N searches in S seconds |
| `ClickDelayClassifier("click-delay", max_hours=24)` | interactions | `late`, value = hours |
| `AbstractLengthClassifier(min_chars=100)` | documents | `short`, value = length |

## 3. Custom classifiers

**`RowClassifier`**: decides from one row's view and never touches the
database. Use it for anything you can tell from the row itself (regexes on
query text, document fields):

```python
import re
from src.classifying import Classification, QueryView, RowClassifier

DOI = re.compile(r"\b10\.\d{4,9}/\S+", re.I)

class HasDoiClassifier(RowClassifier):
    name = "has-doi"          # the key used in filter rules
    level = "queries"

    def classify(self, row: QueryView) -> Classification | None:
        return Classification.flag(bool(DOI.search(row.text)))

classifiers = [HasDoiClassifier()]
```

A `RowClassifier` visits **every row of the level, whatever the snapshot
range**, so its verdicts don't depend on the snapshot. Run it once without
`--filter` (global verdicts), and every snapshot's filter will see them.

**`Classifier`** (aggregate): implement
`classify_all(db, config) -> iterable[(row_id, Classification)]`. Use it when
the verdict needs other rows (counts, rates, co-occurrence). Use one SQL
`GROUP BY` instead of per-row lookups. Respect the snapshot:
`config.start`/`config.end` bound `SearchResult.queried_at` (half-open), and
`config.click_types` names the click types. Use
`src.filtering.rule_clauses(config, level, entity_id_column)` to apply the
filter's other rules, the way `ClicksClassifier` does in
`src/classifying/builtins.py` (copy that pattern). If you read the database
while yielding, collect the results into a list first (as
`SearchBurstClassifier` does), because the runner writes while it consumes
them.

Models are in `src/models.py`: `Query`, `SearchSession`, `SearchResult`,
`Interaction`, `Document`. Check that file rather than guessing column names.

## 4. Running: snapshot scope and order

- `--filter snapshot.yaml` stores the verdicts **under that snapshot** and
  gives aggregates its date range, `click_types` and rules. Without it, the
  verdicts are **global** (snapshot `""`) and the range is unbounded. A filter
  sees its own snapshot's verdicts plus the global ones.
  - Snapshot-independent classifiers (query text, documents, click delay):
    run once, globally.
  - Range-dependent ones (`clicks`, `bot`): run with each snapshot's
    `--filter`.
- **Order matters for aggregates.** `ClicksClassifier` counts only clicks
  that pass the filter's `sessions`/`results`/`interactions` rules, so `bot`
  and `click-delay` verdicts must exist first. Put them earlier in the same
  list, or run `clicks` in a later `classify` call.
- A complete run **replaces** that classifier's verdicts under that snapshot
  (in one transaction). Re-running after a code change is safe and removes
  stale verdicts. `--limit N` makes a partial trial run that only upserts and
  never deletes. Use it to try a slow classifier, then do a full run.
- Renaming a classifier leaves the old name's verdicts in place. Filters
  still referencing the old name keep reading them.

## 5. Manual review round trip

1. `uv run longeval-cli export queries snapshot.yaml --output review.tsv --database-url "$DB"`
2. The reviewer adds a `manual_label` column (e.g. `remove`, leaving the
   others blank).
3. Write a classifier file containing
   `classifiers = ManualLabelsClassifier("review.tsv")`, then run `classify`
   with it (globally, or with `--filter`).
4. The filter drops those queries with `queries: {exclude: {manual: remove}}`.

Matching is on the exact query text. Blank labels give no verdict.

## 6. Check the result

The `classify` output prints `name: N verdicts (table, snapshot '...')`.
**0 verdicts** usually means the wrong level, an always-`None` branch, an
empty date range, or (for `clicks`) `click_types` that don't match the
logged types. Look at the label distribution:

```bash
sqlite3 work/longeval.sqlite3 "SELECT snapshot, label, count(*) FROM query_classifications
  WHERE classifier='has-doi' GROUP BY 1,2 ORDER BY 3 DESC LIMIT 20"
```

The tables are `query_`, `session_`, `result_`, `interaction_` and
`document_classifications`. Look at a few labelled rows to sanity-check
them, then add or adjust the filter rule. The `longeval-snapshot-filter`
skill covers writing the rule and has a checker that confirms each rule's
classifier has verdicts the filter can see.
