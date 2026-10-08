"""Command-line interface for the LongEval-Sci offline database workflows."""

from __future__ import annotations

from pathlib import Path

import click
import sqlalchemy as sa

from .classifying import ClassifierConfig, classify
from .classifying.loader import load_classifiers
from .clicks import QuerySessionStats, iter_query_sessions_from_db
from .database import (
    create_database_engine,
    create_schema,
    database_url,
    make_session_factory,
)
from .discretizing import DISCRETIZERS, make_discretizer
from .documents import (
    fetch_documents,
    find_snapshot_document_ids,
    load_document_source,
    read_document_ids,
)
from .exporting import (
    write_qrels,
    write_queries_tsv,
    write_query_sessions,
    write_snapshot,
)
from .filtering import (
    FilterError,
    SnapshotConfig,
    filter_report,
    load_snapshot_config,
)
from .importing import ImportConfig, import_clicks, import_queries
from .importing.importer import ImportAbort
from .importing.loader import load_parser
from .models import LEVELS, Interaction, Query, SearchResult, SearchSession

_database_url_option = click.option(
    "--database-url", "url", default=None, help="Target database URL."
)
_parser_option = click.option(
    "--parser",
    "parser_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Python file defining a top-level `parser` object (see "
        "src/importing/loader.py), e.g. examples/minimal-example/parser.py."
    ),
)
_paths_argument = click.argument(
    "paths", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path)
)
_classifier_option = click.option(
    "--classifier",
    "classifier_spec",
    required=True,
    help=(
        "Python file defining a top-level `classifiers` object (one "
        "classifier or a list, of any levels), e.g. "
        "examples/minimal-example/classifiers.py (see src/classifying/)."
    ),
)

_source_option = click.option(
    "--source",
    "source_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Python file defining a top-level `source` object (see "
        "src/documents/source.py), e.g. examples/longeval-sci-v2/core_source.py."
    ),
)



@click.group()
def main() -> None:
    """LongEval-CLI: Create Dynamic Test Collections from Search Logs."""


@main.command("init-db")
@_database_url_option
def init_db(url: str | None) -> None:
    """Create any missing tables in the configured database."""
    engine = create_database_engine(url)
    create_schema(engine)
    click.echo(f"Schema ready at {database_url(url)}")


@main.command("database-status")
@_database_url_option
def database_status(url: str | None) -> None:
    """Report table row counts for the configured database."""
    engine = create_database_engine(url)
    click.echo(f"Database: {database_url(url)}")
    with engine.connect() as connection:
        for label, model in (
            ("sessions", SearchSession),
            ("queries", Query),
            ("results", SearchResult),
            ("interactions", Interaction),
            *(
                (f"{verdicts.__tablename__.replace('_', ' ')}", verdicts)
                for verdicts, _ in LEVELS.values()
            ),
        ):
            try:
                count = connection.execute(
                    sa.select(sa.func.count()).select_from(model)
                ).scalar_one()
            except sa.exc.OperationalError:
                count = "not created"
            click.echo(f"  {label}: {count}")


@main.command("import-queries")
@_paths_argument
@_parser_option
@_database_url_option
@click.option("--batch-size", default=5_000, show_default=True)
@click.option(
    "--strict", is_flag=True, help="Abort on the first rejected or conflicting record."
)
@click.option(
    "--max-errors",
    type=int,
    default=None,
    help="Abort once more than this many records have been rejected.",
)
@click.option("--limit", type=int, default=None, help="Stop after N records.")
def import_queries_command(
    paths: tuple[Path, ...],
    parser_path: Path,
    url: str | None,
    batch_size: int,
    strict: bool,
    max_errors: int | None,
    limit: int | None,
) -> None:
    """Phase 1: import search events (queries + rankings) from PATHS."""
    config = ImportConfig(
        inputs=list(paths),
        database_url=database_url(url),
        parser=load_parser(parser_path),
        batch_size=batch_size,
        strict=strict,
        max_errors=max_errors,
        limit=limit,
    )
    _run(import_queries, config)


@main.command("import-clicks")
@_paths_argument
@_parser_option
@_database_url_option
@click.option("--batch-size", default=5_000, show_default=True)
@click.option(
    "--strict", is_flag=True, help="Abort on the first rejected record."
)
@click.option(
    "--max-errors",
    type=int,
    default=None,
    help="Abort once more than this many records have been rejected.",
)
@click.option("--limit", type=int, default=None, help="Stop after N records.")
def import_clicks_command(
    paths: tuple[Path, ...],
    parser_path: Path,
    url: str | None,
    batch_size: int,
    strict: bool,
    max_errors: int | None,
    limit: int | None,
) -> None:
    """Phase 2: import click/interaction events from PATHS.

    Each one is linked, via the search id it was logged against, to the
    SearchResult `import-queries` already created for it — run that first.
    """
    config = ImportConfig(
        inputs=list(paths),
        database_url=database_url(url),
        parser=load_parser(parser_path),
        batch_size=batch_size,
        strict=strict,
        max_errors=max_errors,
        limit=limit,
    )
    _run(import_clicks, config)


@main.command("classify")
@_classifier_option
@_database_url_option
@click.option("--batch-size", default=5_000, show_default=True)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Stop each classifier after N verdicts. Such a run is partial: it only "
    "adds or updates verdicts, never removes stale ones.",
)
@click.option(
    "--filter",
    "filter_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Snapshot filter YAML: its `snapshot` names the verdicts, and its range, "
        "click types and rules are what aggregating classifiers see. "
        "Default: global, unbounded."
    ),
)
def classify_command(
    classifier_spec: str,
    url: str | None,
    batch_size: int,
    limit: int | None,
    filter_file: Path | None,
) -> None:
    """Run the classifiers a file defines and store their verdicts."""
    try:
        snapshot = (
            load_snapshot_config(filter_file) if filter_file else SnapshotConfig()
        )
        classifiers = load_classifiers(classifier_spec)
        stored = classify(
            ClassifierConfig(
                database_url=database_url(url),
                classifiers=classifiers,
                snapshot=snapshot,
                batch_size=batch_size,
                limit=limit,
            )
        )
    except (FilterError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        "Classification complete."
        if limit is None
        else f"Partial classification (--limit {limit}): older verdicts kept."
    )
    where = f", snapshot {snapshot.snapshot!r}" if snapshot.snapshot else ""
    for classifier in classifiers:
        table = LEVELS[classifier.level][0].__tablename__
        click.echo(
            f"{classifier.name}: {stored[classifier.name]} verdicts ({table}{where})"
        )

@main.command("fetch-documents")
@_source_option
@_database_url_option
@click.option(
    "--snapshot",
    default=None,
    help="Only fetch the documents selected for this snapshot. "
    "Default: those selected for any snapshot.",
)
@click.option(
    "--ids",
    "ids_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Fetch the ids in this file (one per line; gzipped if it ends in .gz) "
    "instead of the selected ones.",
)
@click.option(
    "--max-workers",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Batches requested in parallel.",
)
@click.option(
    "--full-text-max-chars",
    type=click.IntRange(min=0),
    default=None,
    help="Store full texts longer than this as empty. Default: keep all.",
)
def fetch_documents_command(
    source_path: Path,
    url: str | None,
    snapshot: str | None,
    ids_file: Path | None,
    max_workers: int,
    full_text_max_chars: int | None,
) -> None:
    """Fetch documents from a source into the database.

    Ids already fetched, or known to be missing, are skipped, so an
    interrupted run can be started again; ids whose batch failed are retried.
    """
    if snapshot is not None and ids_file is not None:
        raise click.UsageError("--snapshot and --ids can't be combined")
    engine = create_database_engine(url)
    create_schema(engine)
    session_factory = make_session_factory(engine)
    try:
        source = load_document_source(source_path)
        if ids_file is not None:
            document_ids = read_document_ids(ids_file)
        else:
            with session_factory() as session:
                document_ids = find_snapshot_document_ids(session, snapshot)
        if not document_ids:
            raise ValueError(
                "no document ids to fetch: run select-documents first, or pass --ids"
            )
        report = fetch_documents(
            session_factory,
            source,
            document_ids,
            max_workers=max_workers,
            full_text_max_chars=full_text_max_chars,
        )
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"  {report.requested:>8}  ids requested")
    click.echo(f"  {report.skipped:>8}  skipped (fetched or missing before)")
    click.echo(f"  {report.fetched:>8}  fetched")
    click.echo(f"  {report.not_found:>8}  not found")
    if report.full_texts_dropped:
        click.echo(f"  {report.full_texts_dropped:>8}  full texts dropped (too long)")
    if report.failed:
        click.echo(f"  {report.failed:>8}  failed: run again to retry", err=True)
    click.echo("Fetch complete.")

@main.group("export")
def export_group() -> None:
    """Export collection files from the database."""


@export_group.command("queries")
@click.argument(
    "filter_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="TSV file to write (header: query_id, query); gzipped if it ends in .gz.",
)
@_database_url_option
def export_queries_tsv_command(
    filter_file: Path, output: Path, url: str | None
) -> None:
    """Export the queries passing the FILTER_FILE YAML, reporting each rule.

    The TSV doubles as the manual-review sheet: add a `manual_label` column,
    then import it with `classify` and a ManualLabelsClassifier.
    """
    engine = create_database_engine(url)
    create_schema(engine)
    try:
        criteria = load_snapshot_config(filter_file)
        with make_session_factory(engine)() as session:
            report = filter_report(session, criteria)
            count = write_queries_tsv(session, criteria, output)
    except FilterError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"  {report.total:>8}  all queries")
    for description, remaining in report.steps:
        click.echo(f"  {remaining:>8}  {description}")
    click.echo(f"Wrote {count} queries to {output}")


@export_group.command("clicks")
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--filter",
    "filter_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Snapshot filter YAML (as for `export queries`): its `click_types` count "
        "as a click and its `from`/`to` bound the snapshot."
    ),
)
@click.option("--shard-size", default=100_000, show_default=True)
@click.option("--batch-size", default=500, show_default=True)
@_database_url_option
def export_clicks_command(
    output_dir: Path,
    filter_file: Path,
    shard_size: int,
    batch_size: int,
    url: str | None,
) -> None:
    """Write a frozen, sharded, gzip-JSONL copy of the click-model query sessions.

    Every displayed item is exported with a binary click flag. See
    src/clicks/query_session.py for how malformed data is handled.
    """
    engine = create_database_engine(url)
    create_schema(engine)
    stats = QuerySessionStats()
    try:
        config = load_snapshot_config(filter_file)
        with make_session_factory(engine)() as session:
            manifest = write_query_sessions(
                iter_query_sessions_from_db(
                    session, config, stats, batch_size=batch_size
                ),
                output_dir,
                config,
                stats,
                shard_size=shard_size,
            )
    except (FilterError, ValueError, FileExistsError) as exc:
        raise click.ClickException(str(exc)) from exc

    counts = manifest["counts"]
    click.echo(f"  {counts['query_sessions']:>8}  query sessions")
    click.echo(f"  {counts['items']:>8}  displayed items")
    click.echo(f"  {counts['clicks']:>8}  clicks")
    for reason, number in counts["skipped"].items():
        click.echo(f"  {number:>8}  skipped: {reason}")
    click.echo(f"Wrote {len(manifest['shards'])} shard(s) to {output_dir}")


def _discretizer_options(required: bool):
    """The discretizer options shared by `export qrels` and `export snapshot`."""
    options = [
        click.option(
            "--discretizer",
            "discretizer_name",
            type=click.Choice(DISCRETIZERS),
            # no explicit default: click >= 8.3 lets `default=None` satisfy
            # `required=True`
            required=required,
            help="Built-in discretizer (see src/discretizing.py)."
            + ("" if required else " Without it the scores themselves are "
               "written as labels."),
        ),
        click.option(
            "--thresholds",
            type=str,
            default=None,
            help="threshold: comma-separated, increasing (default 0.1,0.9).",
        ),
        click.option(
            "--gmax", type=int, default=None, help="chapelle: top grade (default 2)."
        ),
        click.option(
            "--rounding",
            type=click.Choice(["nearest", "floor"]),
            default=None,
            help="chapelle: how to round the grade (default nearest).",
        ),
        click.option(
            "--score-column",
            default="score",
            show_default=True,
            help="Column of SCORES to turn into labels (e.g. `clicks` for raw "
            "qrels).",
        ),
        click.option(
            "--keep-all-queries",
            is_flag=True,
            help="Also keep queries without any label above 0 (dropped by "
            "default when discretizing).",
        ),
    ]

    def decorate(command):
        for option in reversed(options):
            command = option(command)
        return command

    return decorate


def _build_discretizer(
    name: str | None, thresholds: str | None, gmax: int | None, rounding: str | None
):
    if name:
        return make_discretizer(
            name, thresholds=_floats(thresholds), gmax=gmax, rounding=rounding
        )
    if thresholds or gmax or rounding:
        raise ValueError("--thresholds/--gmax/--rounding need --discretizer")
    return None


@export_group.command("qrels")
@click.argument(
    "scores", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="TREC qrels file to write (query_id 0 doc_id label); gzipped if it "
    "ends in .gz.",
)
@_discretizer_options(required=False)
def export_qrels_command(
    scores: Path,
    output: Path,
    discretizer_name: str | None,
    thresholds: str | None,
    gmax: int | None,
    rounding: str | None,
    score_column: str,
    keep_all_queries: bool,
) -> None:
    """Write TREC qrels from a Parquet SCORES file (query_id, doc_id, score)."""
    try:
        discretizer = _build_discretizer(discretizer_name, thresholds, gmax, rounding)
        report = write_qrels(
            scores,
            output,
            discretizer,
            score_column=score_column,
            keep_all_queries=keep_all_queries,
        )
    except (ValueError, KeyError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc

    _echo_qrels(report)
    click.echo(f"Wrote qrels to {output}")


@export_group.command("snapshot")
@click.argument(
    "filter_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("output_root", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--scores",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Parquet file with query_id, doc_id and a score column, computed for "
    "this snapshot's selection (e.g. by a trained click model).",
)
@_discretizer_options(required=True)
@_database_url_option
def export_snapshot_command(
    filter_file: Path,
    output_root: Path,
    scores: Path,
    discretizer_name: str,
    thresholds: str | None,
    gmax: int | None,
    rounding: str | None,
    score_column: str,
    keep_all_queries: bool,
    url: str | None,
) -> None:
    """Package the snapshot named in FILTER_FILE into OUTPUT_ROOT/<snapshot>/.

    Writes qrels.trec.gz, queries.tsv.gz (only the queries with qrels) and a
    manifest.json. Documents are not packaged yet.
    """
    engine = create_database_engine(url)
    create_schema(engine)
    try:
        config = load_snapshot_config(filter_file)
        discretizer = _build_discretizer(discretizer_name, thresholds, gmax, rounding)
        with make_session_factory(engine)() as session:
            report = write_snapshot(
                session,
                config,
                scores,
                output_root,
                discretizer,
                score_column=score_column,
                keep_all_queries=keep_all_queries,
            )
    except (FilterError, ValueError, KeyError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc

    _echo_qrels(report.qrels)
    click.echo(f"  {report.queries_passing:>8}  queries passing the filter")
    dropped = report.queries_passing - report.queries_written
    if dropped:
        click.echo(f"  {dropped:>8}  of them left out (no qrels)")
    click.echo(f"Wrote snapshot {config.snapshot} to {report.path}")


def _echo_qrels(report) -> None:
    click.echo(f"  {report.pairs_read:>8}  pairs read")
    click.echo(f"  {report.pairs_written:>8}  pairs written")
    click.echo(f"  {report.queries_written:>8}  queries")
    if report.queries_dropped:
        click.echo(f"  {report.queries_dropped:>8}  queries dropped (no label > 0)")


def _floats(text: str | None) -> tuple[float, ...] | None:
    return tuple(float(part) for part in text.split(",")) if text else None


def _run(command, config: ImportConfig) -> None:
    try:
        stats = command(config)  # the importer draws its own progress bar
    except ImportAbort as exc:
        click.echo(f"Import failed: {exc}", err=True)
        raise SystemExit(1) from exc

    click.echo("Import complete.")
    _echo_counts(stats)


def _echo_counts(stats) -> None:
    for name, count in stats.items():
        click.echo(f"{name.replace('_', ' ')}: {count}")


if __name__ == "__main__":
    main()
