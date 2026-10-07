from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import sqlalchemy as sa

from src.database import create_database_engine, create_schema
from src.importing import ImportConfig, import_clicks, import_queries
from src.importing.importer import ImportAbort
from src.models import Interaction, SearchResult, SearchSession
from tests.conftest import SampleParser, make_tar_gz, sqlite_url


def _search(search_id, serp, query="q", uid="u1", date=1000) -> str:
    payload = {"date": date, "search_id": search_id, "serp": serp, "query": query}
    if uid is not None:
        payload["uid"] = uid
    return json.dumps(payload)


def _interaction(track_id, type_=None, uid=None, date=1005) -> str:
    payload = {"date": date, "trackId": track_id}
    if type_ is not None:
        payload["type"] = type_
    if uid is not None:
        payload["uid"] = uid
    return json.dumps(payload)


def _basic_archive(tmp_path: Path) -> Path:
    lines = [
        _search("s1", [1, 2], query="q1", date=1000),
        _interaction("s1-1", type_="works", uid="u1", date=1005),
        _interaction("s1-1", type_="works", uid="u1", date=1006),
        _interaction("s1-2", type_="data-provider", date=1010),
    ]
    return make_tar_gz(tmp_path / "archive.tar.gz", {"search_log.log": lines})


def _config(archive, db_url, *, parser=None, **kwargs) -> ImportConfig:
    return ImportConfig(
        inputs=[archive],
        database_url=db_url,
        parser=parser or SampleParser(),
        **kwargs,
    )


def test_full_import_creates_sessions_results_interactions(tmp_path):
    archive = _basic_archive(tmp_path)
    db_url = sqlite_url(tmp_path)

    query_stats = import_queries(_config(archive, db_url, batch_size=10))
    click_stats = import_clicks(_config(archive, db_url, batch_size=10))

    assert query_stats["searches_seen"] == 1
    assert query_stats["rejected_records"] == 0
    assert query_stats["known_mappings"] == 1
    assert click_stats["interactions_seen"] == 3
    assert click_stats["rejected_records"] == 0

    engine = create_database_engine(db_url)
    with sa.orm.Session(engine) as session:
        results = session.query(SearchResult).all()
        assert len(results) == 1
        assert results[0].items == {"1": {"docid": "1"}, "2": {"docid": "2"}}
        assert results[0].page == 1
        assert results[0].results_per_page == 10
        assert results[0].query.query == "q1"

        interactions = session.query(Interaction).all()
        assert len(interactions) == 3
        doc1_clicks = [i for i in interactions if i.document_id == "1"]
        assert len(doc1_clicks) == 2  # repeated clicks stay separate rows

        sessions = session.query(SearchSession).all()
        assert len(sessions) == 1
        assert sessions[0].id == "u1"
        assert sessions[0].ended_at >= sessions[0].started_at


def test_import_queries_is_idempotent(tmp_path):
    archive = _basic_archive(tmp_path)
    db_url = sqlite_url(tmp_path)

    import_queries(_config(archive, db_url, batch_size=10))
    stats2 = import_queries(_config(archive, db_url, batch_size=10))

    assert stats2["searches_seen"] == 1
    assert stats2["known_mappings"] == 1
    engine = create_database_engine(db_url)
    with sa.orm.Session(engine) as session:
        assert session.query(SearchResult).count() == 1


def test_import_clicks_is_idempotent(tmp_path):
    archive = _basic_archive(tmp_path)
    db_url = sqlite_url(tmp_path)
    import_queries(_config(archive, db_url, batch_size=10))

    import_clicks(_config(archive, db_url, batch_size=10))
    stats2 = import_clicks(_config(archive, db_url, batch_size=10))

    assert stats2["interactions_seen"] == 3
    engine = create_database_engine(db_url)
    with sa.orm.Session(engine) as session:
        assert session.query(Interaction).count() == 3  # not doubled


@pytest.mark.parametrize("batch_size", [0, 32_767])
def test_out_of_range_batch_size_is_rejected(tmp_path, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        _config(tmp_path / "a.tar.gz", sqlite_url(tmp_path), batch_size=batch_size)


def test_duplicate_identical_search_resolves_to_existing_result(tmp_path):
    lines = [_search("s1", [1]), _search("s1", [1])]
    archive = make_tar_gz(tmp_path / "archive.tar.gz", {"a.log": lines})
    db_url = sqlite_url(tmp_path)

    stats = import_queries(_config(archive, db_url, batch_size=10))

    assert stats["searches_seen"] == 2
    engine = create_database_engine(db_url)
    with sa.orm.Session(engine) as session:
        assert session.query(SearchResult).count() == 1


def test_conflicting_duplicate_search_is_counted_and_first_occurrence_wins(tmp_path):
    lines = [_search("s1", [1], query="q"), _search("s1", [2], query="different")]
    archive = make_tar_gz(tmp_path / "archive.tar.gz", {"a.log": lines})
    db_url = sqlite_url(tmp_path)

    stats = import_queries(_config(archive, db_url, batch_size=10))

    assert stats["conflicts"] == 1
    engine = create_database_engine(db_url)
    with sa.orm.Session(engine) as session:
        results = session.query(SearchResult).all()
        assert len(results) == 1
        assert results[0].query.query == "q"  # the first occurrence was kept


def test_conflicting_duplicate_search_across_batches_is_counted(tmp_path):
    lines = [_search("s1", [1], query="q"), _search("s1", [2], query="different")]
    archive = make_tar_gz(tmp_path / "archive.tar.gz", {"a.log": lines})
    db_url = sqlite_url(tmp_path)

    stats = import_queries(_config(archive, db_url, batch_size=1))

    assert stats["conflicts"] == 1
    engine = create_database_engine(db_url)
    with sa.orm.Session(engine) as session:
        assert session.query(SearchResult).count() == 1


def test_strict_aborts_on_conflicting_duplicate_search(tmp_path):
    lines = [_search("s1", [1], query="q"), _search("s1", [2], query="different")]
    archive = make_tar_gz(tmp_path / "archive.tar.gz", {"a.log": lines})
    config = _config(archive, sqlite_url(tmp_path), batch_size=10, strict=True)

    with pytest.raises(ImportAbort):
        import_queries(config)


def test_unresolved_interaction_search_is_counted_not_inserted(tmp_path):
    lines = [_interaction("missing-doc1", type_="works")]
    archive = make_tar_gz(tmp_path / "archive.tar.gz", {"a.log": lines})
    db_url = sqlite_url(tmp_path)

    stats = import_clicks(_config(archive, db_url, batch_size=10))

    assert stats["unresolved_searches"] == 1
    engine = create_database_engine(db_url)
    with sa.orm.Session(engine) as session:
        assert session.query(Interaction).count() == 0


def test_document_not_in_result_counted_as_missing(tmp_path):
    lines = [_search("s1", [1, 2]), _interaction("s1-999", type_="works")]
    archive = make_tar_gz(tmp_path / "archive.tar.gz", {"a.log": lines})
    db_url = sqlite_url(tmp_path)
    import_queries(_config(archive, db_url, batch_size=10))

    stats = import_clicks(_config(archive, db_url, batch_size=10))

    assert stats["missing_documents"] == 1
    engine = create_database_engine(db_url)
    with sa.orm.Session(engine) as session:
        assert session.query(Interaction).count() == 0


def test_session_mismatch_is_recorded_but_interaction_is_imported(tmp_path):
    lines = [_search("s1", [1]), _interaction("s1-1", type_="works", uid="u2")]
    archive = make_tar_gz(tmp_path / "archive.tar.gz", {"a.log": lines})
    db_url = sqlite_url(tmp_path)
    import_queries(_config(archive, db_url, batch_size=10))

    stats = import_clicks(_config(archive, db_url, batch_size=10))

    assert stats["session_mismatches"] == 1
    engine = create_database_engine(db_url)
    with sa.orm.Session(engine) as session:
        interactions = session.query(Interaction).all()
        assert len(interactions) == 1
        assert interactions[0].session_id == "u1"  # resolved result's session wins


def test_strict_aborts_on_first_rejected_record(tmp_path):
    lines = ["not valid json", _search("s1", [])]
    archive = make_tar_gz(tmp_path / "archive.tar.gz", {"a.log": lines})
    config = _config(archive, sqlite_url(tmp_path), strict=True)

    with pytest.raises(ImportAbort):
        import_queries(config)


def test_max_errors_aborts_after_threshold(tmp_path):
    lines = ["bad"] * 5
    archive = make_tar_gz(tmp_path / "archive.tar.gz", {"a.log": lines})
    config = _config(archive, sqlite_url(tmp_path), max_errors=2)

    with pytest.raises(ImportAbort):
        import_queries(config)


def test_limit_bounds_records_processed(tmp_path):
    lines = [_search(f"s{i}", [1], date=1000 + i) for i in range(5)]
    archive = make_tar_gz(tmp_path / "archive.tar.gz", {"a.log": lines})
    config = _config(archive, sqlite_url(tmp_path), limit=2)

    stats = import_queries(config)

    assert stats["searches_seen"] == 2


def test_second_import_adds_to_nonempty_database(tmp_path):
    archive1 = make_tar_gz(tmp_path / "a1.tar.gz", {"a.log": [_search("s1", [1])]})
    db_url = sqlite_url(tmp_path)
    import_queries(_config(archive1, db_url, batch_size=10))

    archive2 = make_tar_gz(
        tmp_path / "a2.tar.gz",
        {"b.log": [_search("s2", [3], query="q2", uid="u2", date=2000)]},
    )
    stats2 = import_queries(_config(archive2, db_url, batch_size=10))

    assert stats2["searches_seen"] == 1
    assert stats2["known_mappings"] == 2
    engine = create_database_engine(db_url)
    with sa.orm.Session(engine) as session:
        assert session.query(SearchResult).count() == 2


def test_interaction_type_filter_is_parser_driven(tmp_path):
    lines = [
        _search("s1", [1]),
        _interaction("s1-1", type_="works"),
        _interaction("s1-1", type_="data-provider", date=1006),
    ]
    archive = make_tar_gz(tmp_path / "archive.tar.gz", {"a.log": lines})
    db_url = sqlite_url(tmp_path)
    import_queries(_config(archive, db_url, batch_size=10))

    parser = SampleParser(accepted_interaction_types=frozenset({"works"}))
    stats = import_clicks(_config(archive, db_url, batch_size=10, parser=parser))

    assert stats["rejected_records"] == 1  # the filtered-out data-provider record
    engine = create_database_engine(db_url)
    with sa.orm.Session(engine) as session:
        assert session.query(Interaction).count() == 1


def test_sqlite_foreign_keys_enforced(tmp_path):
    db_url = sqlite_url(tmp_path)
    engine = create_database_engine(db_url)
    create_schema(engine)
    with sa.orm.Session(engine) as session:
        session.add(Interaction(result_id=999, document_id="x", type="works"))
        with pytest.raises(sa.exc.IntegrityError):
            session.commit()


@pytest.mark.skipif(
    not os.environ.get("LONGEVAL_TEST_POSTGRES_URL"),
    reason="set LONGEVAL_TEST_POSTGRES_URL to run the Postgres integration test",
)
def test_import_against_postgres(tmp_path):
    archive = _basic_archive(tmp_path)
    db_url = os.environ["LONGEVAL_TEST_POSTGRES_URL"]

    query_stats = import_queries(_config(archive, db_url, batch_size=10))
    click_stats = import_clicks(_config(archive, db_url, batch_size=10))

    assert query_stats["searches_seen"] == 1
    assert click_stats["interactions_seen"] == 3
