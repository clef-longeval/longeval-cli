from __future__ import annotations

import json

import sqlalchemy as sa
from click.testing import CliRunner

from src.cli import main
from src.database import create_database_engine
from src.importing.loader import DEFAULT_PARSER_PATH
from src.models import Query
from tests.conftest import make_tar_gz, sqlite_url


def _search(search_id, serp, query="q", uid="u1", date=1000) -> str:
    return json.dumps(
        {"date": date, "search_id": search_id, "serp": serp, "query": query, "uid": uid}
    )


def _interaction(track_id, type_, date=1005) -> str:
    return json.dumps({"date": date, "trackId": track_id, "type": type_})


def test_cli_import_queries_then_clicks_smoke(tmp_path):
    lines = [_search("s1", [1]), _interaction("s1-1", "works")]
    archive = make_tar_gz(tmp_path / "archive.tar.gz", {"a.log": lines})
    db_url = sqlite_url(tmp_path)
    runner = CliRunner()

    queries_result = runner.invoke(
        main,
        [
            "import-queries",
            str(archive),
            "--parser",
            str(DEFAULT_PARSER_PATH),
            "--database-url",
            db_url,
            "--batch-size",
            "10",
        ],
    )
    assert queries_result.exit_code == 0, queries_result.output
    assert "searches: 1" in queries_result.output
    assert "known mappings: 1" in queries_result.output

    clicks_result = runner.invoke(
        main,
        [
            "import-clicks",
            str(archive),
            "--parser",
            str(DEFAULT_PARSER_PATH),
            "--database-url",
            db_url,
            "--batch-size",
            "10",
        ],
    )
    assert clicks_result.exit_code == 0, clicks_result.output
    assert "interactions: 1" in clicks_result.output


def test_cli_import_queries_reports_failure(tmp_path):
    lines = [_search("s1", [1], query="q"), _search("s1", [2], query="q2")]
    archive = make_tar_gz(tmp_path / "archive.tar.gz", {"a.log": lines})
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "import-queries",
            str(archive),
            "--parser",
            str(DEFAULT_PARSER_PATH),
            "--database-url",
            sqlite_url(tmp_path),
            "--strict",
        ],
    )

    assert result.exit_code == 1
    assert "Import failed" in result.output


def test_cli_import_queries_requires_parser(tmp_path):
    lines = [_search("s1", [1])]
    archive = make_tar_gz(tmp_path / "archive.tar.gz", {"a.log": lines})
    runner = CliRunner()

    result = runner.invoke(
        main,
        ["import-queries", str(archive), "--database-url", sqlite_url(tmp_path)],
    )

    assert result.exit_code != 0
    assert "--parser" in result.output


def test_cli_import_queries_with_custom_parser(tmp_path):
    """--parser lets a user override normalization without touching ingestion code."""
    lines = [_search("s1", [1], query="Hello World")]
    archive = make_tar_gz(tmp_path / "archive.tar.gz", {"a.log": lines})
    parser_path = tmp_path / "parser.py"
    parser_path.write_text(
        "from src.importing.loader import load_parser\n"
        "class UppercaseIdParser:\n"
        "    def __init__(self):\n"
        "        self._inner = load_parser(None)\n"
        "    def query_id(self, query):\n"
        "        return query.upper()\n"
        "    def iter_searches(self, lines):\n"
        "        return self._inner.iter_searches(lines)\n"
        "    def iter_interactions(self, lines):\n"
        "        return self._inner.iter_interactions(lines)\n"
        "parser = UppercaseIdParser()\n"
    )
    db_url = sqlite_url(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "import-queries",
            str(archive),
            "--parser",
            str(parser_path),
            "--database-url",
            db_url,
        ],
    )

    assert result.exit_code == 0, result.output
    engine = create_database_engine(db_url)
    with sa.orm.Session(engine) as session:
        assert session.get(Query, "HELLO WORLD") is not None
