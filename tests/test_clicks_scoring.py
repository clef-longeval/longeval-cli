from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("clax")
pq = pytest.importorskip("pyarrow.parquet")

from clax.models import DocumentCTRModel  # noqa: E402
from flax import nnx  # noqa: E402

from src.clicks import QueryDocIdMap, QuerySession  # noqa: E402
from src.clicks.scoring import score_candidates, write_scored_docs  # noqa: E402
from tests.test_clicks import _t  # noqa: E402


def _query_session(result_id: int, query: str, *docs: str) -> QuerySession:
    n = len(docs)
    return QuerySession(
        result_id, query, None, _t(1), docs, tuple(range(1, n + 1)), (0,) * n
    )


def _model(query_doc_ids: QueryDocIdMap) -> DocumentCTRModel:
    return DocumentCTRModel(query_doc_pairs=len(query_doc_ids), rngs=nnx.Rngs(0))


def _trained_ids() -> QueryDocIdMap:
    query_doc_ids = QueryDocIdMap()
    query_doc_ids.assign([("q1", "A"), ("q1", "B"), ("q2", "A")])
    query_doc_ids.frozen = True
    return query_doc_ids


def test_scores_each_candidate_pair_once_with_its_query_session_count():
    query_doc_ids = _trained_ids()
    model = _model(query_doc_ids)
    stream = [
        _query_session(1, "q1", "A", "B"),
        _query_session(2, "q1", "B", "A"),
        _query_session(3, "q2", "A"),
    ]

    scored = score_candidates(model, stream, query_doc_ids, batch_size=2)

    assert sorted(zip(scored.query_ids, scored.doc_ids)) == [
        ("q1", "A"), ("q1", "B"), ("q2", "A")
    ]
    assert dict(zip(zip(scored.query_ids, scored.doc_ids), scored.query_sessions)) == {
        ("q1", "A"): 2, ("q1", "B"): 2, ("q2", "A"): 1
    }
    assert scored.report.query_sessions == 3
    assert scored.report.unseen_items == 0
    # DCTR: the score is the pair's click probability, whatever the query session
    expected = np.asarray(model.ctr.prob({"query_doc_ids": np.arange(3)})).reshape(-1)
    by_pair = dict(zip(zip(scored.query_ids, scored.doc_ids), scored.scores))
    for pair_id, pair in enumerate(query_doc_ids):
        assert by_pair[pair] == pytest.approx(float(expected[pair_id]), rel=1e-5)
    assert all(0 < s < 1 for s in scored.scores)


def test_pairs_unknown_to_the_model_are_skipped_and_counted():
    query_doc_ids = _trained_ids()
    stream = [
        _query_session(1, "q1", "A", "NEW", "B"),
        _query_session(2, "q9", "A"),  # nothing known: dropped entirely
    ]

    scored = score_candidates(_model(query_doc_ids), stream, query_doc_ids)

    assert sorted(zip(scored.query_ids, scored.doc_ids)) == [("q1", "A"), ("q1", "B")]
    assert scored.report.query_sessions == 1
    assert scored.report.unseen_items == 2


def test_write_scored_docs_follows_the_contract(tmp_path):
    query_doc_ids = _trained_ids()
    scored = score_candidates(
        _model(query_doc_ids), [_query_session(1, "q1", "A")], query_doc_ids
    )

    write_scored_docs(scored, tmp_path / "out" / "scores.parquet")

    table = pq.read_table(tmp_path / "out" / "scores.parquet").to_pydict()
    assert list(table) == ["query_id", "doc_id", "score", "query_sessions"]
    assert table["query_id"] == ["q1"] and table["doc_id"] == ["A"]
    assert table["query_sessions"] == [1]


def test_scored_docs_feed_export_qrels(tmp_path):
    from src.exporting import write_qrels

    query_doc_ids = _trained_ids()
    scored = score_candidates(
        _model(query_doc_ids), [_query_session(1, "q1", "A", "B")], query_doc_ids
    )
    write_scored_docs(scored, tmp_path / "s.parquet")

    report = write_qrels(tmp_path / "s.parquet", tmp_path / "q.txt")

    assert report.pairs_written == 2


def test_score_from_db_and_file_agree(tmp_path):
    from src.clicks import QuerySessionStats, iter_query_sessions_from_db
    from src.clicks.scoring import score_from_db, score_from_file
    from src.database import create_database_engine, make_session_factory
    from src.exporting import write_query_sessions
    from src.filtering import SnapshotConfig
    from tests.conftest import sqlite_url
    from tests.test_clicks import CLICK, _seed

    url = sqlite_url(tmp_path)
    _seed(url)
    config = SnapshotConfig(click_types=tuple(CLICK))
    query_doc_ids = QueryDocIdMap()
    with make_session_factory(create_database_engine(url))() as session:
        query_doc_ids.assign(
            (i.query_id, d)
            for i in iter_query_sessions_from_db(session, config)
            for d in i.doc_ids
        )
        stats = QuerySessionStats()
        write_query_sessions(
            iter_query_sessions_from_db(session, config, stats),
            tmp_path / "clicks",
            config,
            stats,
        )
    query_doc_ids.frozen = True
    model = _model(query_doc_ids)

    from_db = score_from_db(model, query_doc_ids, config, database_url=url)
    from_export = score_from_file(model, query_doc_ids, tmp_path / "clicks")

    assert from_db.report.pairs == len(query_doc_ids)
    assert from_db.query_ids == from_export.query_ids
    assert from_db.doc_ids == from_export.doc_ids
    assert np.allclose(from_db.scores, from_export.scores)
