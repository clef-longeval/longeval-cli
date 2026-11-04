"""Dry-run a longeval-cli --parser file over raw logs, without a database.

Runs both passes exactly as import-queries / import-clicks would see them and
reports what was accepted, what was rejected (by reason, with example lines),
and whether the interactions can be linked to the searches at all.

    uv run python skills/longeval-log-parser/scripts/dry_run_parser.py \
        --parser path/to/parser.py data/raw/logs [--limit 100000] [--examples 3]

Run it from the repository root (it imports ``src``).
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.importing.loader import load_parser  # noqa: E402
from src.importing.records import (  # noqa: E402
    InteractionLogEntry,
    RejectedRecord,
    SearchLogEntry,
)
from src.importing.sources import discover_inputs, stream_source  # noqa: E402


def lines(paths, limit):
    stream = itertools.chain.from_iterable(stream_source(p) for p in paths)
    return itertools.islice(stream, limit) if limit else stream


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--parser", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None, help="lines per pass")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    parser = load_parser(args.parser)
    paths = discover_inputs(args.paths)
    print(f"{len(paths)} input file(s)")

    # Pass 1: searches
    searches, rejected, examples = 0, Counter(), defaultdict(list)
    search_ids: set[str] = set()
    query_ids: Counter = Counter()
    sessions: set[str] = set()
    first = None
    for result in parser.iter_searches(lines(paths, args.limit)):
        if isinstance(result, RejectedRecord):
            rejected[result.reason] += 1
            if len(examples[result.reason]) < args.examples:
                examples[result.reason].append(result.line.text[:200])
            continue
        entry = result.record
        if not isinstance(entry, SearchLogEntry):
            print(f"!! iter_searches yielded a {type(entry).__name__}")
            continue
        if entry.timestamp.tzinfo is None:
            print("!! naive timestamp (needs tzinfo=UTC):", entry)
            break
        searches += 1
        first = first or entry
        search_ids.add(entry.external_search_id)
        query_ids[parser.query_id(entry.query)] += 1
        sessions.add(entry.session_id)

    print("\n== import-queries pass")
    print(f"accepted searches: {searches}  (distinct search ids: {len(search_ids)})")
    print(f"distinct queries: {len(query_ids)}   sessions: {len(sessions)}")
    if first:
        print(f"first entry: {first}")
    _report(rejected, examples)

    # Pass 2: interactions
    accepted, unlinked, types = 0, 0, Counter()
    rejected, examples = Counter(), defaultdict(list)
    first = None
    for result in parser.iter_interactions(lines(paths, args.limit)):
        if isinstance(result, RejectedRecord):
            rejected[result.reason] += 1
            if len(examples[result.reason]) < args.examples:
                examples[result.reason].append(result.line.text[:200])
            continue
        entry = result.record
        if not isinstance(entry, InteractionLogEntry):
            print(f"!! iter_interactions yielded a {type(entry).__name__}")
            continue
        accepted += 1
        first = first or entry
        types[entry.interaction_type] += 1
        if entry.external_search_id not in search_ids:
            unlinked += 1

    print("\n== import-clicks pass")
    print(f"accepted interactions: {accepted}")
    if first:
        print(f"first entry: {first}")
    print(f"interaction types: {dict(types.most_common())}")
    print(f"interactions whose search id was not seen in pass 1: {unlinked}")
    if accepted and unlinked == accepted:
        print("!! no interaction links to a search: check that both passes "
              "extract the same search id (format, prefix, case)")
    _report(rejected, examples)


def _report(rejected: Counter, examples) -> None:
    if not rejected:
        print("rejected: none")
        return
    print(f"rejected: {sum(rejected.values())}")
    for reason, count in rejected.most_common():
        print(f"  {count:>8}  {reason}")
        for text in examples[reason]:
            print(f"            e.g. {text}")


if __name__ == "__main__":
    main()
