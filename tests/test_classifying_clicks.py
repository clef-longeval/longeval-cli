"""The built-in classifiers that read the database: clicks, click delay, bots."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import sqlalchemy as sa
from click.testing import CliRunner

from src.classifying import (
    ClassifierConfig,
    ClickDelayClassifier,
    ClicksClassifier,
    SearchBurstClassifier,
    classify,
)
from src.cli import main
from src.database import create_database_engine, create_schema, make_session_factory
from src.filtering import Rules, SnapshotConfig
from src.models import (
    Interaction,
    InteractionClassification,
    Query,
    QueryClassification,
    SearchResult,
    SearchSession,
    SessionClassification,
)
from tests.conftest import sqlite_url

T0 = datetime(2025, 3, 10, 12, tzinfo=timezone.utc)


def _seed(db_url: str) -> None:
    """q0 (session s0): 3 clicks, one 30 h late, one `author`.
    q1 (session bot): 1 click 40 days later; the bot searched 3 times in 20 s."""
    engine = create_database_engine(db_url)
    create_schema(engine)
    with make_session_factory(engine)() as session:
        session.add_all([Query(id="q0", query="a"), Query(id="q1", query="b")])
        session.add_all(
            [
                SearchSession(id="s0", started_at=T0),
                SearchSession(id="bot", started_at=T0),
            ]
        )
        session.add_all(
            [
                SearchResult(id=1, query_id="q0", session_id="s0", queried_at=T0),
                SearchResult(id=2, query_id="q1", session_id="bot", queried_at=T0),
                SearchResult(
                    id=3,
                    query_id="q1",
                    session_id="bot",
                    queried_at=T0 + timedelta(seconds=10),
                ),
                SearchResult(
                    id=4,
                    query_id="q1",
                    session_id="bot",
                    queried_at=T0 + timedelta(seconds=20),
                ),
            ]
        )
        session.flush()
        for id_, result_id, delay_h, type_ in [
            (1, 1, 1, "get-pdf"),
            (2, 1, 30, "get-pdf"),
            (3, 1, 2, "author"),
            (4, 2, 24 * 40, "works"),
        ]:
            session.add(
                Interaction(
                    id=id_,
                    result_id=result_id,
                    document_id="d",
                    type=type_,
                    date=T0 + timedelta(hours=delay_h),
                )
            )
        session.commit()


def _run(db_url: str, classifier, snapshot: SnapshotConfig | None = None) -> None:
    classify(
        ClassifierConfig(
            db_url, classifier, snapshot=snapshot or SnapshotConfig(), progress=False
        )
    )


def _verdicts(db_url: str, model=QueryClassification, snapshot: str = "") -> dict:
    entity = {
        QueryClassification: "query_id",
        SessionClassification: "session_id",
        InteractionClassification: "interaction_id",
    }[model]
    with sa.orm.Session(create_database_engine(db_url)) as session:
        rows = session.scalars(sa.select(model).where(model.snapshot == snapshot))
        return {getattr(r, entity): (r.label, r.value) for r in rows}


def test_clicks_counts_every_type_without_click_types(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed(db_url)

    _run(db_url, ClicksClassifier())

    assert _verdicts(db_url) == {"q0": ("3", 3.0), "q1": ("1", 1.0)}


def test_clicks_follows_click_types_and_the_snapshot_range(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed(db_url)
    march = SnapshotConfig(
        snapshot="march",
        start=T0 - timedelta(days=1),
        end=T0 + timedelta(days=20),
        click_types=("get-pdf", "works"),
    )

    _run(db_url, ClicksClassifier(), march)

    assert _verdicts(db_url, snapshot="march") == {
        "q0": ("2", 2.0),
        "q1": ("1", 1.0),
    }


def test_clicks_uses_search_date_at_snapshot_boundaries(tmp_path):
    db_url = sqlite_url(tmp_path)
    engine = create_database_engine(db_url)
    create_schema(engine)
    start = T0
    end = T0 + timedelta(days=1)
    with make_session_factory(engine)() as session:
        session.add_all(
            [Query(id="before", query="before"), Query(id="inside", query="inside")]
        )
        session.add_all(
            [
                SearchResult(
                    id=1,
                    query_id="before",
                    queried_at=start - timedelta(seconds=1),
                ),
                SearchResult(
                    id=2,
                    query_id="inside",
                    queried_at=end - timedelta(seconds=1),
                ),
            ]
        )
        session.flush()
        session.add_all(
            [
                Interaction(
                    id=1, result_id=1, document_id="d", type="works", date=start
                ),
                Interaction(
                    id=2, result_id=2, document_id="d", type="works", date=end
                ),
            ]
        )
        session.commit()

    _run(
        db_url,
        ClicksClassifier(),
        SnapshotConfig(snapshot="window", start=start, end=end, click_types=("works",)),
    )

    assert _verdicts(db_url, snapshot="window") == {"inside": ("1", 1.0)}


def test_clicks_skips_excluded_interactions_and_sessions(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed(db_url)
    _run(db_url, [ClickDelayClassifier(), SearchBurstClassifier(searches=3)])
    rules = SnapshotConfig(
        click_types=("get-pdf", "works"),
        sessions=Rules(exclude={"bot": ("bot",)}),
        interactions=Rules(exclude={"click-delay": ("late",)}),
    )

    _run(db_url, ClicksClassifier(), rules)

    assert _verdicts(db_url) == {"q0": ("1", 1.0)}


def test_click_delay_labels_only_late_interactions(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed(db_url)

    _run(db_url, ClickDelayClassifier(max_hours=24))

    assert _verdicts(db_url, InteractionClassification) == {
        2: ("late", 30.0),
        4: ("late", 960.0),
    }


def test_search_burst_labels_only_fast_sessions(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed(db_url)

    _run(db_url, SearchBurstClassifier("bot", searches=3, within_seconds=20))
    assert _verdicts(db_url, SessionClassification) == {"bot": ("bot", 3.0)}

    _run(db_url, SearchBurstClassifier("slow", searches=3, within_seconds=19))
    assert set(_verdicts(db_url, SessionClassification)) == {"bot"}


def test_cli_snapshot_name_and_range_come_from_the_filter(tmp_path):
    db_url = sqlite_url(tmp_path)
    _seed(db_url)
    classifier = tmp_path / "clicks.py"
    classifier.write_text(
        "from src.classifying.builtins import ClicksClassifier\n"
        "classifiers = [ClicksClassifier('clicks')]\n"
    )
    snapshot = tmp_path / "march.yaml"
    snapshot.write_text("snapshot: march\nfrom: 2025-03-01\nto: 2025-03-31\n")

    result = CliRunner().invoke(
        main,
        [
            "classify",
            "--classifier",
            str(classifier),
            "--database-url",
            db_url,
            "--filter",
            str(snapshot),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "clicks: 2 verdicts (query_classifications, snapshot 'march')" in (
        result.output
    )
    assert _verdicts(db_url, snapshot="march") == {
        "q0": ("3", 3.0),
        "q1": ("1", 1.0),
    }
