"""Command-line interface for the LongEval-Sci offline database workflows."""

from __future__ import annotations

from pathlib import Path

import click
import sqlalchemy as sa

from .classifying import ClassifierConfig, classify_interactions, classify_queries
from .classifying.loader import load_interaction_classifier, load_query_classifier
from .database import create_database_engine, create_schema, database_url
from .importing import ImportConfig, import_clicks, import_queries
from .importing.importer import ImportAbort
from .importing.loader import load_parser
from .models import (
    Interaction,
    InteractionClassification,
    Query,
    QueryClassification,
    SearchResult,
    SearchSession,
)

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
        "src/importing/loader.py), e.g. scripts/longeval-sci-v2/parser.py."
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
        "A classifier file (defining a top-level `query_classifier` or "
        "`interaction_classifier` object) or a `module.path:name` import "
        "spec for a ready-to-use one, e.g. "
        "`src.classifying.builtins:IsBooleanClassifier` (see "
        "src/classifying/loader.py)."
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
            ("query classifications", QueryClassification),
            ("interaction classifications", InteractionClassification),
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


@main.command("classify-queries")
@_classifier_option
@_database_url_option
@click.option("--batch-size", default=5_000, show_default=True)
@click.option("--limit", type=int, default=None, help="Stop after N queries.")
def classify_queries_command(
    classifier_spec: str, url: str | None, batch_size: int, limit: int | None
) -> None:
    """Run a QueryClassifier over every stored query."""
    config = ClassifierConfig(
        database_url=database_url(url),
        classifier=load_query_classifier(classifier_spec),
        batch_size=batch_size,
        limit=limit,
    )
    _run_classification(classify_queries, config)


@main.command("classify-interactions")
@_classifier_option
@_database_url_option
@click.option("--batch-size", default=5_000, show_default=True)
@click.option("--limit", type=int, default=None, help="Stop after N interactions.")
def classify_interactions_command(
    classifier_spec: str, url: str | None, batch_size: int, limit: int | None
) -> None:
    """Run an InteractionClassifier over every stored interaction."""
    config = ClassifierConfig(
        database_url=database_url(url),
        classifier=load_interaction_classifier(classifier_spec),
        batch_size=batch_size,
        limit=limit,
    )
    _run_classification(classify_interactions, config)


def _run_classification(command, config: ClassifierConfig) -> None:
    stats = command(config)
    click.echo("Classification complete.")
    click.echo(f"considered: {stats.considered}")
    click.echo(f"classified: {stats.classified}")
    click.echo(f"unlabeled: {stats.unlabeled}")


def _run(command, config: ImportConfig) -> None:
    try:
        stats = command(config)
    except ImportAbort as exc:
        click.echo(f"Import failed: {exc}")
        raise SystemExit(1) from exc

    click.echo("Import complete.")
    click.echo(f"searches: {stats.searches_seen}")
    click.echo(f"interactions: {stats.interactions_seen}")
    click.echo(f"rejected: {stats.rejected_records}")
    click.echo(f"conflicts: {stats.conflicts}")
    click.echo(f"unresolved searches: {stats.unresolved_searches}")
    click.echo(f"missing documents: {stats.missing_documents}")
    click.echo(f"session mismatches: {stats.session_mismatches}")
    click.echo(f"known mappings: {stats.known_mappings}")


if __name__ == "__main__":
    main()
