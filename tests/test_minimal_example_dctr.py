from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

pytest.importorskip("clax")
pq = pytest.importorskip("pyarrow.parquet")

from src.cli import main  # noqa: E402
from src.clicks import QuerySessionStats, iter_query_sessions_from_db  # noqa: E402
from src.database import create_database_engine, make_session_factory  # noqa: E402
from src.exporting import write_query_sessions  # noqa: E402
from src.filtering import SnapshotConfig  # noqa: E402
from src.importing.loader import load_parser_module  # noqa: E402
from tests.conftest import sqlite_url  # noqa: E402
from tests.test_clicks import CLICK, _seed  # noqa: E402

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/minimal-example/train_dctr_from_export.py"
)


def test_script_trains_clax_dctr_from_an_export(tmp_path):
    script = load_parser_module(_SCRIPT)
    db_url = sqlite_url(tmp_path)
    _seed(db_url)
    export = tmp_path / "clicks"
    config = SnapshotConfig(click_types=tuple(CLICK))
    stats = QuerySessionStats()
    with make_session_factory(create_database_engine(db_url))() as session:
        write_query_sessions(
            iter_query_sessions_from_db(session, config, stats), export, config, stats
        )

    out = tmp_path / "dctr.parquet"
    result = CliRunner().invoke(
        script.main, [str(export), "--output", str(out), "--epochs", "2"]
    )

    assert result.exit_code == 0, result.output
    table = pq.read_table(out).to_pydict()
    assert ("q1", "B") in set(zip(table["query_id"], table["doc_id"]))
    assert all(0.0 < p < 1.0 for p in table["score"])

    def export_qrels(*args: str) -> list[str]:
        qrels = tmp_path / "qrels.txt"
        result = CliRunner().invoke(
            main, ["export", "qrels", str(out), "--output", str(qrels), *args]
        )
        assert result.exit_code == 0, result.output
        return qrels.read_text().splitlines()

    probabilities = export_qrels()
    assert len(probabilities) == len(table["score"])
    assert all(len(line.split()) == 4 for line in probabilities)

    graded = export_qrels("--discretizer", "threshold", "--keep-all-queries")
    assert {line.split()[3] for line in graded} <= {"0", "1", "2"}
