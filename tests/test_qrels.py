from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from src.cli import main
from src.discretizing import (
    AnyClickDiscretizer,
    ChapelleDiscretizer,
    ThresholdDiscretizer,
    make_discretizer,
)
from src.exporting import write_qrels

ROWS = [  # query, doc, score, clicks
    ("q1", "d1", 0.05, 0),
    ("q1", "d2", 0.5, 3),
    ("q1", "d3", 0.95, 9),
    ("q2", "d1", 0.01, 0),
    ("q2", "d2", 0.09, 0),
    ("q3", "d1", 0.1, 1),
]


@pytest.fixture
def scores(tmp_path) -> Path:
    path = tmp_path / "scores.parquet"
    pq.write_table(
        pa.table(
            {
                "query_id": [r[0] for r in ROWS],
                "doc_id": [r[1] for r in ROWS],
                "score": [r[2] for r in ROWS],
                "clicks": [r[3] for r in ROWS],
            }
        ),
        path,
    )
    return path


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def test_threshold_discretizer_counts_thresholds_reached():
    d = ThresholdDiscretizer((0.1, 0.9))
    assert [d.label(s) for s in (0.0, 0.0999, 0.1, 0.89, 0.9, 1.0)] == [
        0, 0, 1, 1, 2, 2,
    ]
    for bad in ((), (0.9, 0.1), (0.5, 0.5)):
        with pytest.raises(ValueError):
            ThresholdDiscretizer(bad)


def test_any_click_discretizer_omits_unclicked_pairs():
    d = AnyClickDiscretizer()
    assert (d.label(0), d.label(1), d.label(7)) == (None, 1, 1)


def test_chapelle_discretizer_inverts_the_dcg_discount_mapping():
    d = ChapelleDiscretizer(gmax=2)
    # P = (2**g - 1) / 4, so the exact grades map back to themselves
    assert [d.label(p) for p in (0.0, 0.25, 0.75)] == [0, 1, 2]
    assert [d.label(p) for p in (0.1, 0.11, 0.45, 0.47, 1.0)] == [0, 1, 1, 2, 2]
    floor = ChapelleDiscretizer(gmax=2, rounding="floor")
    assert [floor.label(p) for p in (0.24, 0.25, 0.74, 0.75, 1.0)] == [0, 1, 1, 2, 2]
    assert ChapelleDiscretizer(gmax=1).label(0.5) == 1  # binary: P = (2**g - 1) / 2
    for bad in ({"gmax": 0}, {"rounding": "up"}):
        with pytest.raises(ValueError):
            ChapelleDiscretizer(**bad)
    with pytest.raises(ValueError, match="probability"):
        d.label(1.5)


def test_qrels_without_a_discretizer_hold_the_probabilities(scores, tmp_path):
    out = tmp_path / "qrels.txt"
    report = write_qrels(scores, out)

    assert _lines(out)[:3] == ["q1 0 d1 0.05", "q1 0 d2 0.5", "q1 0 d3 0.95"]
    assert len(_lines(out)) == 6 == report.pairs_written
    assert report.queries_dropped == 0


def test_discretized_qrels_drop_queries_without_a_relevant_document(
    scores, tmp_path
):
    out = tmp_path / "qrels.txt"
    report = write_qrels(scores, out, ThresholdDiscretizer((0.1, 0.9)))

    assert _lines(out) == [
        "q1 0 d1 0", "q1 0 d2 1", "q1 0 d3 2", "q3 0 d1 1",
    ]
    assert (report.pairs_read, report.pairs_written) == (6, 4)
    assert (report.queries_written, report.queries_dropped) == (2, 1)

    kept = tmp_path / "all.txt"
    write_qrels(scores, kept, ThresholdDiscretizer((0.1, 0.9)), keep_all_queries=True)
    assert "q2 0 d2 0" in _lines(kept) and len(_lines(kept)) == 6


def test_raw_qrels_label_every_clicked_pair_one(scores, tmp_path):
    out = tmp_path / "raw.txt"
    write_qrels(scores, out, AnyClickDiscretizer(), score_column="clicks")
    assert _lines(out) == ["q1 0 d2 1", "q1 0 d3 1", "q3 0 d1 1"]


def test_make_discretizer_builds_the_builtins_and_rejects_stray_parameters():
    assert make_discretizer("threshold").thresholds == (0.1, 0.9)
    assert make_discretizer("threshold", thresholds=(0.5,)).label(0.6) == 1
    assert make_discretizer("chapelle").gmax == 2
    assert make_discretizer("chapelle", gmax=1, rounding="floor").rounding == "floor"
    assert isinstance(make_discretizer("any-click"), AnyClickDiscretizer)
    with pytest.raises(ValueError, match="does not take gmax"):
        make_discretizer("threshold", gmax=2)
    with pytest.raises(ValueError, match="unknown"):
        make_discretizer("nope")


def test_export_qrels_cli(scores, tmp_path):
    out = tmp_path / "qrels.txt"
    result = CliRunner().invoke(
        main,
        ["export", "qrels", str(scores), "--output", str(out),
         "--discretizer", "threshold"],
    )
    assert result.exit_code == 0, result.output
    assert "1  queries dropped" in result.output
    assert _lines(out)[0] == "q1 0 d1 0"

    chapelle = CliRunner().invoke(
        main,
        ["export", "qrels", str(scores), "--output", str(out),
         "--discretizer", "chapelle", "--gmax", "1"],
    )
    assert chapelle.exit_code == 0, chapelle.output
    stray = CliRunner().invoke(
        main, ["export", "qrels", str(scores), "--output", str(out), "--gmax", "1"]
    )
    assert stray.exit_code != 0 and "--discretizer" in stray.output

    missing = CliRunner().invoke(
        main,
        ["export", "qrels", str(scores), "--output", str(out),
         "--score-column", "nope"],
    )
    assert missing.exit_code != 0
