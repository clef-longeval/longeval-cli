"""Snapshot config and classifier-verdict filters, from one small YAML file.

One file configures one snapshot::

    snapshot: snapshot-1      # name verdicts are stored under
    from: 2025-03-01          # snapshot start (UTC, inclusive)
    to: 2025-06-01            # snapshot end (UTC, exclusive)
    published_from: 2024-12-01  # documents published in [published_from,
    published_to: 2025-06-01    #   published_to) are candidates (UTC)
    click_types: [get-pdf, works]  # interactions that count as a click
    queries:
      require: {is-boolean: 0}     # verdict must exist, with one of these labels
      min: {clicks: 31}            # verdict's numeric value must be >= n
      exclude: {manual: remove}    # dropped if the verdict has one of these labels
    sessions:
      exclude: {bot: bot}
    interactions:
      exclude: {click-delay: late}

``snapshot`` names the snapshot and ``from``/``to`` give its date range, a
half-open UTC interval ``[from, to)`` (dates, or datetimes; naive values are
read as UTC). ``classify``, ``export clicks`` and ``export queries`` all read
the same file, so the range is stated once. Rules may be added gradually; a
file without any selects everything.

Each level section (``queries``, ``sessions``, ``results``, ``interactions``:
the table its classifiers label) takes the same three rules. A query, session
or result that fails them is left out together with its searches; an
interaction that fails them does not count as a click.

A verdict counts if it was stored under the file's ``snapshot`` or globally
(snapshot ``""``), so classifiers that don't depend on a date range need not be
re-run per snapshot. A row with no verdict from a ``require``/``min``
classifier is dropped; with none from an ``exclude`` classifier it is kept.
Whether a classifier has actually been run is up to the user.

``published_from``/``published_to`` bound the documents' publication dates,
also a half-open UTC interval: ``collect-document-candidates`` lists the
documents published in it. They are separate from ``from``/``to`` because a
snapshot's documents need not be published while its searches happen.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import sqlalchemy as sa
import yaml
from sqlalchemy.orm import Session

from .models import LEVELS, Query

_KEYS = {
    "snapshot",
    "from",
    "to",
    "published_from",
    "published_to",
    "click_types",
    *LEVELS,
}
_RULES = {"require", "min", "exclude"}


class FilterError(ValueError):
    """The criteria file is malformed."""


@dataclass(frozen=True)
class Rules:
    """The rules for one level (see the module docstring)."""

    require: dict[str, tuple[str, ...]] = field(default_factory=dict)
    min: dict[str, float] = field(default_factory=dict)
    exclude: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class SnapshotConfig:
    snapshot: str = ""
    start: datetime | None = None  # inclusive, UTC
    end: datetime | None = None  # exclusive, UTC
    published_start: datetime | None = None  # inclusive, UTC
    published_end: datetime | None = None  # exclusive, UTC
    click_types: tuple[str, ...] = ()  # interaction types that count as a click
    queries: Rules = Rules()
    sessions: Rules = Rules()
    results: Rules = Rules()
    interactions: Rules = Rules()


@dataclass(frozen=True)
class FilterReport:
    total: int
    steps: list[tuple[str, int]]  # (rule, queries remaining after it)

    @property
    def kept(self) -> int:
        return self.steps[-1][1] if self.steps else self.total


def load_snapshot_config(path: Path | str) -> SnapshotConfig:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise FilterError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise FilterError("a criteria must be a mapping")
    _check_keys(data, _KEYS, "")

    criteria = SnapshotConfig(
        snapshot=str(data.get("snapshot", "")),
        start=_moment(data.get("from"), "from"),
        end=_moment(data.get("to"), "to"),
        published_start=_moment(data.get("published_from"), "published_from"),
        published_end=_moment(data.get("published_to"), "published_to"),
        click_types=_as_strings(data.get("click_types", [])),
        **{level: _rules(data.get(level), level) for level in LEVELS},
    )
    if criteria.start and criteria.end and criteria.start >= criteria.end:
        raise FilterError("`from` must be before `to`")
    if (
        criteria.published_start
        and criteria.published_end
        and criteria.published_start >= criteria.published_end
    ):
        raise FilterError("`published_from` must be before `published_to`")
    return criteria


def _check_keys(data: dict, allowed: set[str], where: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        names = ", ".join(sorted(map(str, unknown)))
        raise FilterError(f"unknown key(s){where}: {names}")


def _rules(raw, level: str) -> Rules:
    if raw is None:
        return Rules()
    if not isinstance(raw, dict):
        raise FilterError(f"`{level}` must map require/min/exclude -> rules")
    _check_keys(raw, _RULES, f" in `{level}`")
    return Rules(
        require=_labels(raw.get("require")),
        min=_numbers(raw.get("min")),
        exclude=_labels(raw.get("exclude")),
    )


def _moment(raw, key: str) -> datetime | None:
    """Read a YAML date/datetime/ISO string as a timezone-aware UTC datetime."""
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = datetime.fromisoformat(raw)
        except ValueError:
            raise FilterError(f"`{key}` is not a date: {raw!r}") from None
    elif isinstance(raw, date) and not isinstance(raw, datetime):
        raw = datetime(raw.year, raw.month, raw.day)
    elif not isinstance(raw, datetime):
        raise FilterError(f"`{key}` is not a date: {raw!r}")
    if raw.tzinfo is None:
        return raw.replace(tzinfo=timezone.utc)
    return raw.astimezone(timezone.utc)


def _labels(raw) -> dict[str, tuple[str, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise FilterError("`require`/`exclude` must map classifier -> label(s)")
    # Labels are strings in the database; YAML would read a bare `2` as an int.
    return {str(name): _as_strings(labels) for name, labels in raw.items()}


def _as_strings(labels) -> tuple[str, ...]:
    return tuple(str(v) for v in (labels if isinstance(labels, list) else [labels]))


def _numbers(raw) -> dict[str, float]:
    if raw is None:
        return {}
    if not isinstance(raw, dict) or not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw.values()
    ):
        raise FilterError("`min` must map classifier -> number")
    return {str(name): float(v) for name, v in raw.items()}


def rule_clauses(criteria: SnapshotConfig, level: str, entity_id) -> list:
    """``level``'s rules as SQL conditions on the column ``entity_id``.

    ``entity_id`` names the labelled row in the caller's statement, e.g.
    ``Query.id``, or ``SearchResult.session_id`` to filter results by their
    session's verdicts.
    """
    return [clause for _, clause in _rules_sql(criteria, level, entity_id)]


def filter_queries(session: Session, criteria: SnapshotConfig) -> list[tuple[str, str]]:
    """Return ``(query_id, query_text)`` for every query that passes the filter."""
    rows = session.execute(
        sa.select(Query.id, Query.query)
        .where(*rule_clauses(criteria, "queries", Query.id))
        .order_by(Query.id)
    )
    return [(query_id, text or "") for query_id, text in rows]


def filter_report(session: Session, criteria: SnapshotConfig) -> FilterReport:
    """Count the queries left after each query rule, applied in file order."""
    total = _count(session, [])
    steps, applied = [], []
    for description, clause in _rules_sql(criteria, "queries", Query.id):
        applied.append(clause)
        steps.append((description, _count(session, applied)))
    return FilterReport(total, steps)


def _count(session: Session, clauses: list) -> int:
    count = sa.select(sa.func.count()).select_from(Query).where(*clauses)
    return session.scalar(count) or 0


def _rules_sql(criteria: SnapshotConfig, level: str, entity_id):
    verdicts, verdict_entity = LEVELS[level]
    rules: Rules = getattr(criteria, level)

    def verdict(classifier: str, *conditions):
        return sa.exists().where(
            verdict_entity == entity_id,
            verdicts.classifier == classifier,
            verdicts.snapshot.in_(("", criteria.snapshot)),
            *conditions,
        )

    for name, labels in rules.require.items():
        yield f"require {name} in {list(labels)}", verdict(
            name, verdicts.label.in_(labels)
        )
    for name, minimum in rules.min.items():
        yield f"min {name} >= {minimum:g}", verdict(name, verdicts.value >= minimum)
    for name, labels in rules.exclude.items():
        yield f"exclude {name} in {list(labels)}", ~verdict(
            name, verdicts.label.in_(labels)
        )
