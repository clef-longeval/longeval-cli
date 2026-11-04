---
name: longeval-snapshot-filter
description: Write, review or debug the snapshot filter YAML that longeval-cli's classify, export queries/clicks/documents/snapshot and the document-collection commands all read: snapshot name, [from, to) search window, published_from/published_to, click_types, and require/min/exclude rules per level (queries, sessions, results, interactions, documents). Use this whenever a LongEval user defines a new snapshot or time window, decides which queries/clicks/documents count, wants to drop bots, late clicks, boolean or manually removed queries, or set a minimum click count, or asks why a filter keeps too few/too many queries (e.g. "export queries gives 0 queries").
---

# Writing a longeval-cli snapshot filter

One YAML file describes one snapshot, and every stage reads the same file.
That is why the date range, click definition and rules are stated once. Most
mistakes here are silent: the CLI doesn't fail, it just selects nothing or
drops nothing. Write the file, then **check it against the database**.

## The format (`src/filtering.py`)

```yaml
snapshot: snapshot-2          # name the verdicts are stored under; names the release dir
from: 2025-06-01              # search time, inclusive, UTC
to: 2025-09-01                # search time, exclusive, UTC
published_from: 2025-06-01    # documents published in [published_from, published_to)
published_to: 2025-09-01      #   are candidates; later-published ones are not released
retrieved_depth: 1000         # optional: depth of the document source's own ranking
click_types: [works, get-pdf] # interaction types that count as a click (empty: all)
queries:                      # one section per level, each with the same three rules
  require: {is-boolean: 0}    # a verdict must exist with one of these labels
  min: {clicks: 5}            # a verdict's numeric value must be >= n
  exclude: {manual: remove}   # dropped if its verdict has one of these labels
sessions:
  exclude: {bot: bot}
results: {}
interactions:
  exclude: {click-delay: late}
documents:
  exclude: {abstract-length: short}
```

Real examples: `examples/minimal-example/filter.yaml` and
`examples/longeval-sci-v2/filters/snapshot-*.yaml`. Unknown keys are
an error, so the file is validated.

## The semantics, carefully

- **Dates.** Both ranges are half-open UTC: `[from, to)`. To cover all of
  March 2025, write `from: 2025-03-01` and `to: 2025-04-01`, not `to:
  2025-03-31`, which would lose the last day. Naive datetimes are read as UTC.
  Consecutive snapshots should share a boundary (one's `to` = the next one's
  `from`) so no day is lost or counted twice.
- **The search window selects searches by search time.** A click on an
  in-window search counts even if it happened after `to`. A click on a search
  before `from` doesn't count. Excluding very late clicks is the job of a
  `click-delay` verdict.
- **`click_types`** must match the type strings the logs actually contain.
  A misspelled type means zero clicks, and then `min: {clicks: n}` keeps
  nothing.
- **Rules per level.**
  - `require: {c: labels}`: the row needs a verdict from `c` with one of the
    labels. **No verdict means dropped.**
  - `min: {c: n}`: the row needs a verdict from `c` with `value >= n`.
    **No verdict means dropped.**
  - `exclude: {c: labels}`: the row is dropped if its verdict has one of the
    labels. **No verdict means kept.**

  Labels are strings: `require: {is-boolean: 0}` matches label `"0"`. A list
  means any of them: `exclude: {manual: [remove, spam]}`. There is
  deliberately no OR between rules and no "missing" test. Rules are ANDed.
- **What failing a rule means.** A query, session or result that fails is
  left out together with its searches, so a bot session's searches vanish
  from the click data. An interaction that fails is just not a click. A
  document that fails is not released (nor is one published on or after
  `published_to`; undated documents are kept).
- **Which verdicts count.** Those stored under this file's `snapshot`, plus
  global ones (snapshot `""`). A classifier that never ran, or ran under a
  different snapshot name, is **invisible**: `require`/`min` on it then select
  nothing, and `exclude` on it drops nothing. Nothing warns you about this.
  It is the most common reason a filter yields 0 or "too many" rows.
- **Rule order and dependencies.** `clicks` counts only clicks that pass the
  `sessions`/`results`/`interactions` rules *as they were when `clicks` ran*.
  After adding or changing a session or interaction rule, re-run the
  `ClicksClassifier` with this filter so the counts reflect it.

## Workflow

1. **Ask or infer the decisions**: snapshot name, search window, which
   interaction types are clicks, what to drop (bots, late clicks, boolean or
   structured queries, manual removals) and the minimum click count. Base the
   rules on classifier names that exist. Look at existing filter files and
   classifier files, or ask.
2. **Write the YAML** with short comments explaining each choice. The file
   is the snapshot's documentation, and the examples are commented the same
   way.
3. **Make sure the verdicts exist.** Run the classifiers each rule names (see
   the `longeval-classifier` skill). Range-dependent ones (`clicks`, `bot`)
   need `--filter this.yaml`. Snapshot-independent ones can be global.
4. **Check the filter against the database** with this skill's script:

   ```bash
   uv run python skills/longeval-snapshot-filter/scripts/check_filter.py \
       snapshot.yaml --database-url "$DB"
   ```

   It validates the file, then for each rule shows whether the classifier
   has visible verdicts, its labels and value range, and whether the wanted
   labels occur at all. It also checks that every `click_type` was logged.
   Fix every `!!` line before going on.
5. **Look at the funnel**:

   ```bash
   uv run longeval-cli export queries snapshot.yaml --output review.tsv --database-url "$DB"
   ```

   This prints the number of queries left after each `queries` rule, in file
   order. A step that drops to 0, or drops nothing, points at the rule to
   examine. Tune thresholds (e.g. `min: {clicks: n}`) with the user from
   these numbers. Let the user pick the trade-off between fewer, better-judged
   queries and more queries.

## Debugging "0 queries" / "too many"

Check these in order:
1. Did the checker flag an unseen classifier? Is there a snapshot-name
   mismatch between `classify --filter` and this file?
2. Does the date window contain searches? It is search time, UTC, half-open.
3. Do the `click_types` match the logged types?
4. Was `clicks` run after the bot/late rules were in place, and with this
   filter?
5. Is a `require` rule on a classifier that only stores exceptions? For
   example, `require: {bot: ...}` cannot work. Use `exclude` for those.
