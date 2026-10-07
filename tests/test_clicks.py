from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone

import pytest
import sqlalchemy as sa
from click.testing import CliRunner

from src.cli import main
from src.clicks import (
    QueryDocIdMap,
    QuerySessionFormatError,
    QuerySessionStats,
    iter_query_sessions_from_db,
    iter_query_sessions_from_file,
    read_manifest,
)
from src.database import create_database_engine, create_schema, make_session_factory
from src.exporting import write_query_sessions
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

CLICK = frozenset({"click"})


def _t(day: int, hour: int = 10) -> datetime:
    return datetime(2026, 1, day, hour, tzinfo=timezone.utc)


def _items(*docs: str) -> dict:
    return {str(i): {"docid": d, "type": "BASE"} for i, d in enumerate(docs, 1)}


def _seed(db_url: str) -> None:
    engine = create_database_engine(db_url)
    create_schema(engine)
    with make_session_factory(engine)() as session:
        session.add_all([Query(id="q1", query="one"), Query(id="q2", query="two")])
        session.add(
            QueryClassification(
                query_id="q1", classifier="flag", snapshot="", label="ok",
                created_at=_t(1),
            )
        )
        session.add_all(
            [
                # ids are not chronological; r1/r2 share a timestamp (tie -> id)
                SearchResult(id=2, query_id="q1", queried_at=_t(1),
                             items=_items("A", "B", "C", "D")),
                SearchResult(id=1, query_id="q2", queried_at=_t(1),
                             items=_items("A", "B")),
                SearchResult(id=3, query_id="q1", queried_at=_t(2),
                             items=_items("A", "B", "A")),
                SearchResult(id=4, query_id="q1", queried_at=_t(3),
                             items=_items("A", "B", "A")),
                SearchResult(id=5, query_id="q2", queried_at=_t(3), items="garbage"),
                SearchResult(id=6, query_id="q2", queried_at=None,
                             items=_items("A")),
                SearchResult(id=7, query_id="q2", queried_at=_t(5),
                             items=_items("A", "B", "C")),
            ]
        )
        session.flush()
        session.add_all(
            [
                Interaction(result_id=2, document_id="B", type="click"),
                Interaction(result_id=2, document_id="B", type="click"),
                Interaction(result_id=2, document_id="C", type="view"),
                Interaction(result_id=2, document_id="ZZ", type="click"),
                Interaction(result_id=3, document_id="B", type="click"),
                Interaction(result_id=4, document_id="A", type="click"),
                Interaction(result_id=7, document_id="C", type="click"),
            ]
        )
        session.commit()


@pytest.fixture
def db_url(tmp_path) -> str:
    url = sqlite_url(tmp_path)
    _seed(url)
    return url


def _stream(db_url, stats=None, *, batch_size=500, **kwargs):
    kwargs.setdefault("click_types", CLICK)
    config = SnapshotConfig(**kwargs)
    with make_session_factory(create_database_engine(db_url))() as session:
        return list(
            iter_query_sessions_from_db(session, config, stats, batch_size=batch_size)
        )


def test_every_displayed_item_gets_a_flag_and_repeats_count_once(db_url):
    stats = QuerySessionStats()
    by_id = {i.result_id: i for i in _stream(db_url, stats)}

    first = by_id[2]
    assert first.doc_ids == ("A", "B", "C", "D")
    assert first.positions == (1, 2, 3, 4)
    # B clicked twice -> one click; "view" excluded; ZZ not displayed
    assert first.clicks == (0, 1, 0, 0)
    assert by_id[1].clicks == (0, 0)  # unclicked query sessions are kept
    assert stats.unmatched_interactions == 1


def test_accepted_click_types_are_configurable(db_url):
    both = _stream(db_url, click_types=("click", "view"))
    assert {i.result_id: i.clicks for i in both}[2] == (0, 1, 1, 0)
    with pytest.raises(ValueError, match="click_types"):
        _stream(db_url, click_types=())


def test_bad_data_is_skipped_or_kept_by_documented_rules(db_url):
    stats = QuerySessionStats()
    by_id = {i.result_id: i for i in _stream(db_url, stats)}

    assert 5 not in by_id and stats.skipped["malformed_items"] == 1
    assert 4 not in by_id and stats.skipped["ambiguous_click"] == 1
    assert 6 not in by_id  # no timestamp: left out of the stream
    # duplicated document, but the click is on the unambiguous one
    assert by_id[3].doc_ids == ("A", "B", "A") and by_id[3].clicks == (0, 1, 0)
    assert stats.query_sessions == 4 == len(by_id)


def test_order_is_chronological_then_by_id_and_independent_of_batch_size(db_url):
    big = _stream(db_url)
    assert [i.result_id for i in big] == [1, 2, 3, 7]
    assert _stream(db_url, batch_size=1) == big
    assert _stream(db_url, batch_size=3) == big


def test_snapshot_is_half_open_utc(db_url):
    day2_to_3 = _stream(db_url, start=_t(2), end=_t(5))
    assert [i.result_id for i in day2_to_3] == [3]  # day 5 10:00 is excluded
    exactly = _stream(db_url, start=_t(5), end=_t(6))
    assert [i.result_id for i in exactly] == [7]  # start is included


def test_query_filter_restricts_query_sessions(db_url):
    only_q1 = _stream(db_url, queries=Rules(require={"flag": ("ok",)}))
    assert [i.result_id for i in only_q1] == [2, 3]
    assert {i.query_id for i in only_q1} == {"q1"}


def test_an_excluded_session_drops_its_query_sessions(db_url):
    with make_session_factory(create_database_engine(db_url))() as session:
        session.add(SearchSession(id="bot", started_at=_t(1)))
        session.get(SearchResult, 2).session_id = "bot"
        session.add(
            SessionClassification(
                session_id="bot", classifier="bot", label="bot", created_at=_t(1)
            )
        )
        session.commit()

    kept = _stream(db_url, sessions=Rules(exclude={"bot": ("bot",)}))

    assert [i.result_id for i in kept] == [1, 3, 7]


def test_an_excluded_interaction_is_not_a_click(db_url):
    with make_session_factory(create_database_engine(db_url))() as session:
        late = session.scalars(
            sa.select(Interaction.id).where(
                Interaction.result_id == 2, Interaction.document_id == "B"
            )
        ).all()
        session.add_all(
            InteractionClassification(
                interaction_id=i, classifier="click-delay", label="late",
                created_at=_t(1),
            )
            for i in late
        )
        session.commit()

    rules = Rules(exclude={"click-delay": ("late",)})
    by_id = {i.result_id: i for i in _stream(db_url, interactions=rules)}

    assert by_id[2].clicks == (0, 0, 0, 0)  # still shown, no longer clicked
    assert by_id[3].clicks == (0, 1, 0)


def test_export_round_trips_to_identical_query_sessions(db_url, tmp_path):
    expected = _stream(db_url)
    out = tmp_path / "export"
    config = SnapshotConfig(click_types=CLICK)
    stats = QuerySessionStats()
    with make_session_factory(create_database_engine(db_url))() as session:
        manifest = write_query_sessions(
            iter_query_sessions_from_db(session, config, stats),
            out, config, stats, shard_size=3,
        )

    assert list(iter_query_sessions_from_file(out)) == expected
    assert [s["records"] for s in manifest["shards"]] == [3, 1]
    assert manifest["counts"]["query_sessions"] == 4
    assert manifest["counts"]["skipped"] == {
        "ambiguous_click": 1, "malformed_items": 1,
    }
    assert manifest["selection"]["click_types"] == ["click"]
    assert "sqlite" not in json.dumps(manifest)  # no database location leaks
    assert read_manifest(out)["format_version"] == 1
    with pytest.raises(FileExistsError):
        write_query_sessions([], out, config, QuerySessionStats())


def test_reader_detects_corruption_and_unknown_versions(db_url, tmp_path):
    out = tmp_path / "export"
    config = SnapshotConfig(click_types=CLICK)
    stats = QuerySessionStats()
    with make_session_factory(create_database_engine(db_url))() as session:
        manifest = write_query_sessions(
            iter_query_sessions_from_db(session, config, stats), out, config, stats
        )
    shard = out / manifest["shards"][0]["path"]

    original = shard.read_bytes()
    shard.write_bytes(original + b"x")  # trailing garbage changes the checksum
    with pytest.raises(QuerySessionFormatError):
        list(iter_query_sessions_from_file(out))
    shard.write_bytes(original)

    record = json.loads(gzip.decompress(original).decode().splitlines()[0])
    record["clicks"] = record["clicks"][:-1]  # lengths no longer agree
    shard.write_bytes(gzip.compress(json.dumps(record).encode() + b"\n"))
    with pytest.raises(QuerySessionFormatError, match="differ in length"):
        list(iter_query_sessions_from_file(out, verify=False))
    with pytest.raises(QuerySessionFormatError, match="checksum"):
        list(iter_query_sessions_from_file(out))

    (out / "manifest.json").write_text(
        json.dumps({**manifest, "format_version": 2})
    )
    with pytest.raises(QuerySessionFormatError, match="version"):
        list(iter_query_sessions_from_file(out))


def test_export_clicks_cli(db_url, tmp_path):
    out = tmp_path / "cli-export"
    snapshot = tmp_path / "snapshot.yaml"
    snapshot.write_text("from: 2026-01-02\nclick_types: [click]\n")
    result = CliRunner().invoke(
        main,
        ["export", "clicks", str(out),
         "--filter", str(snapshot), "--database-url", db_url],
    )
    assert result.exit_code == 0, result.output
    assert "skipped: ambiguous_click" in result.output
    assert [i.result_id for i in iter_query_sessions_from_file(out)] == [3, 7]
    assert read_manifest(out)["selection"]["snapshot"]["from"].startswith("2026-01-02")

    missing = CliRunner().invoke(
        main, ["export", "clicks", str(tmp_path / "x"), "--database-url", db_url]
    )
    assert missing.exit_code != 0 and "--filter" in missing.output


# --- CLAX batches and the pair-ID mapping ------------------------------------

np = pytest.importorskip("numpy")
from src.clicks.clax_loaders import iter_batches  # noqa: E402


def test_batches_are_padded_deterministically(db_url, tmp_path):
    query_sessions = _stream(db_url)  # lengths 4, 2, 3, 3
    batches = list(
        iter_batches(query_sessions, QueryDocIdMap(), batch_size=3, max_positions=3)
    )

    assert [b["clicks"].shape for b in batches] == [(3, 3), (1, 3)]
    first = batches[0]
    assert first["positions"].dtype == np.int16
    assert first["query_doc_ids"].dtype == np.int32
    assert first["mask"].dtype == np.bool_
    # r1 (id 1): two items then padding; r2 (id 2): D dropped (position 4 > 3)
    assert first["mask"].tolist() == [
        [True, True, False], [True, True, True], [True, True, True],
    ]
    assert first["positions"][0].tolist() == [1, 2, 0]
    assert first["n"].tolist() == [2, 3, 3]
    assert first["clicks"][1].tolist() == [0, 1, 0]
    assert first["clicks"][0, 2] == 0 and first["query_doc_ids"][0, 2] == 0


def test_query_doc_ids_are_stable_and_survive_a_parquet_round_trip(db_url, tmp_path):
    query_sessions = _stream(db_url)
    path = tmp_path / "model" / "query_doc_ids.parquet"

    def ids(query_doc_ids):
        return [
            b["query_doc_ids"].tolist()
            for b in iter_batches(query_sessions, query_doc_ids, batch_size=2)
        ]

    query_doc_ids = QueryDocIdMap()
    first = ids(query_doc_ids)
    assert ids(QueryDocIdMap()) == first  # same stream -> same IDs
    assert ids(query_doc_ids) == first  # a second pass assigns nothing new
    size = len(query_doc_ids)
    # (q1, A) and (q2, A) are different pairs; IDs are dense from 0
    assert sorted({i for b in first for row in b for i in row}) == list(range(size))

    query_doc_ids.save(path)
    loaded = QueryDocIdMap.load(path)
    assert len(loaded) == size and loaded.frozen
    assert ids(loaded) == first
    assert loaded.lookup("q2", "B") == query_doc_ids.lookup("q2", "B") is not None
    assert loaded.lookup("q1", "no-such-doc") is None
    with pytest.raises(KeyError):
        loaded.assign([("q1", "no-such-doc")])


def test_file_and_database_sources_give_identical_batches(db_url, tmp_path):
    out = tmp_path / "export"
    config = SnapshotConfig(click_types=CLICK)
    stats = QuerySessionStats()
    with make_session_factory(create_database_engine(db_url))() as session:
        write_query_sessions(
            iter_query_sessions_from_db(session, config, stats), out, config, stats
        )
        direct = list(iter_batches(
            iter_query_sessions_from_db(session, config), QueryDocIdMap(),
            batch_size=2,
        ))
    replay = list(iter_batches(
        iter_query_sessions_from_file(out), QueryDocIdMap(), batch_size=2
    ))
    assert len(direct) == len(replay)
    for left, right in zip(direct, replay, strict=True):
        assert left.keys() == right.keys()
        assert all(np.array_equal(left[k], right[k]) for k in left)


def test_loaders_split_and_reiterate(db_url, tmp_path):
    pytest.importorskip("numpy")
    from src.clicks.clax_loaders import (
        loaders_from_db,
        loaders_from_file,
        loaders_from_stream,
    )

    config = SnapshotConfig(click_types=CLICK)

    def open_stream():
        with make_session_factory(create_database_engine(db_url))() as session:
            yield from iter_query_sessions_from_db(session, config)

    query_doc_ids, train, validation = loaders_from_stream(
        open_stream, holdout=2, batch_size=1
    )
    assert query_doc_ids.frozen
    # 4 query sessions: indices 0, 2 validate; 1, 3 train
    assert (len(train), len(validation)) == (2, 2)
    first = [b["query_doc_ids"].tolist() for b in train]
    assert first == [b["query_doc_ids"].tolist() for b in train]  # re-iterable
    with pytest.raises(ValueError, match="holdout"):
        loaders_from_stream(open_stream, holdout=1)

    with make_session_factory(create_database_engine(db_url))() as session:
        direct = loaders_from_db(session, config, holdout=2, batch_size=1)
        assert direct[0].frozen and (len(direct[1]), len(direct[2])) == (2, 2)
        # the default batch_size (1024) exceeds the database page cap (900)
        default = loaders_from_db(session, config, holdout=2)
        assert (default[1].size, default[2].size) == (2, 2)
        out = tmp_path / "export"
        stats = QuerySessionStats()
        write_query_sessions(
            iter_query_sessions_from_db(session, config, stats), out, config, stats
        )

    replay = loaders_from_file(out, holdout=2, batch_size=1)
    assert [b["query_doc_ids"].tolist() for b in replay[1]] == first
    assert len(replay[0]) == len(direct[0])
