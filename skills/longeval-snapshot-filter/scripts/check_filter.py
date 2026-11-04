"""Check a snapshot filter YAML against the database before relying on it.

Validates the file the way longeval-cli does, then, for every rule, looks up
whether the named classifier has verdicts the filter can see (under the
file's snapshot or global) and which labels/values they have. A rule whose
classifier never ran fails silently in longeval-cli (``require``/``min``
select nothing, ``exclude`` drops nothing); this script makes that visible.

    uv run python skills/longeval-snapshot-filter/scripts/check_filter.py \
        filter.yaml --database-url sqlite:///work/longeval.sqlite3

Run it from the repository root (it imports ``src``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import sqlalchemy as sa  # noqa: E402

from src.database import create_database_engine, make_session_factory  # noqa: E402
from src.filtering import FilterError, load_snapshot_config  # noqa: E402
from src.models import LEVELS  # noqa: E402

MAX_LABELS = 8


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("filter", type=Path)
    ap.add_argument("--database-url", default=None)
    args = ap.parse_args()

    try:
        config = load_snapshot_config(args.filter)
    except FilterError as exc:
        sys.exit(f"invalid filter: {exc}")

    print(f"snapshot: {config.snapshot or '(none: verdicts read as global only)'}")
    print(f"searches: [{config.start}, {config.end})")
    if config.published_start or config.published_end:
        print(f"published: [{config.published_start}, {config.published_end})")
    print(f"click_types: {list(config.click_types) or '(all types count)'}")

    engine = create_database_engine(args.database_url)
    problems = 0
    with make_session_factory(engine)() as db:
        for level, (table, _) in LEVELS.items():
            rules = getattr(config, level)
            named = [
                *(("require", n, v) for n, v in rules.require.items()),
                *(("min", n, v) for n, v in rules.min.items()),
                *(("exclude", n, v) for n, v in rules.exclude.items()),
            ]
            if not named:
                continue
            print(f"\n[{level}]")
            for kind, name, wanted in named:
                problems += _check(db, table, config.snapshot, kind, name, wanted)
        if config.click_types:
            problems += _check_click_types(db, config.click_types)

    print(f"\n{problems} problem(s)" if problems else "\nno problems found")


def _check(db, table, snapshot, kind, name, wanted) -> int:
    visible = sa.and_(
        table.classifier == name, table.snapshot.in_(("", snapshot))
    )
    rows = db.execute(
        sa.select(table.snapshot, table.label, sa.func.count(),
                  sa.func.min(table.value), sa.func.max(table.value))
        .where(visible)
        .group_by(table.snapshot, table.label)
        .order_by(sa.func.count().desc())
    ).all()
    shown = f"{kind} {name}: {list(wanted) if kind != 'min' else f'>= {wanted:g}'}"
    if not rows:
        other = db.scalars(
            sa.select(table.snapshot).where(table.classifier == name).distinct()
        ).all()
        hint = (f" (it has verdicts under {other}, not '{snapshot}' or global)"
                if other else " (no verdicts at all: run classify first)")
        effect = "drops nothing" if kind == "exclude" else "selects NOTHING"
        print(f"  !! {shown} -> classifier unseen, rule {effect}{hint}")
        return 1
    snapshots = sorted({r[0] or "global" for r in rows})
    total = sum(r[2] for r in rows)
    print(f"  ok {shown}  ({total} verdicts, {', '.join(snapshots)})")
    labels = {}
    for _, label, count, _, _ in rows:
        labels[label] = labels.get(label, 0) + count
    top = sorted(labels.items(), key=lambda kv: -kv[1])[:MAX_LABELS]
    print("     labels: " + ", ".join(f"{lbl!r}={c}" for lbl, c in top)
          + (" ..." if len(labels) > MAX_LABELS else ""))
    if kind == "min":
        values = [(r[3], r[4]) for r in rows if r[3] is not None]
        if not values:
            print("  !! no numeric `value` on these verdicts: min selects nothing")
            return 1
        print(f"     value range: {min(v[0] for v in values):g} .. "
              f"{max(v[1] for v in values):g}")
    elif not set(wanted) & set(labels):
        effect = "drops nothing" if kind == "exclude" else "selects NOTHING"
        print(f"  !! none of {list(wanted)} occurs as a label: rule {effect}")
        return 1
    return 0


def _check_click_types(db, click_types) -> int:
    from src.models import Interaction

    present = dict(db.execute(
        sa.select(Interaction.type, sa.func.count()).group_by(Interaction.type)
    ).all())
    print(f"\n[click_types] interaction types in db: {present}")
    missing = [t for t in click_types if t not in present]
    if missing:
        print(f"  !! click type(s) never logged: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    main()
