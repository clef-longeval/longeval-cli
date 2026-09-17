from __future__ import annotations

import pytest

from src.importing.sources import discover_inputs, fingerprint_file, stream_source
from tests.conftest import make_plain_gz, make_tar_gz


def test_stream_plain_file(tmp_path):
    path = tmp_path / "log.log"
    path.write_text("line one\nline two\n")

    lines = list(stream_source(path))

    assert [line.text for line in lines] == ["line one", "line two"]
    assert [line.record_number for line in lines] == [1, 2]
    assert lines[0].source_reference == lines[1].source_reference


def test_stream_gzip_file(tmp_path):
    path = tmp_path / "log.log.gz"
    make_plain_gz(path, ["a", "b", "c"])

    lines = list(stream_source(path))

    assert [line.text for line in lines] == ["a", "b", "c"]


def test_stream_tar_gz_single_member(tmp_path):
    path = tmp_path / "archive.tar.gz"
    make_tar_gz(path, {"search_log.log": ["a", "b"]})

    lines = list(stream_source(path))

    assert [line.text for line in lines] == ["a", "b"]
    assert lines[0].source_reference.endswith(":search_log.log")
    assert lines[0].record_number == 1
    assert lines[1].record_number == 2


def test_stream_tar_gz_multiple_members_in_order(tmp_path):
    path = tmp_path / "archive.tar.gz"
    make_tar_gz(path, {"a.log": ["1"], "b.log": ["2", "3"]})

    lines = list(stream_source(path))

    assert [line.text for line in lines] == ["1", "2", "3"]
    refs = [line.source_reference for line in lines]
    assert refs[0] != refs[1]
    assert refs[1] == refs[2]
    assert refs[0].endswith(":a.log")
    assert refs[1].endswith(":b.log")


def test_stream_tar_gz_record_numbers_restart_per_member(tmp_path):
    path = tmp_path / "archive.tar.gz"
    make_tar_gz(path, {"a.log": ["1"], "b.log": ["2", "3"]})

    lines = list(stream_source(path))

    assert [line.record_number for line in lines] == [1, 1, 2]


def test_discover_inputs_sorted_and_expands_directory(tmp_path):
    (tmp_path / "b.log").write_text("x\n")
    (tmp_path / "a.log").write_text("y\n")

    files = discover_inputs([tmp_path])

    assert [f.name for f in files] == ["a.log", "b.log"]


def test_discover_inputs_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_inputs([tmp_path / "does-not-exist.log"])


def test_fingerprint_stable_across_calls(tmp_path):
    path = tmp_path / "log.log"
    path.write_text("hello\n")

    assert fingerprint_file(path) == fingerprint_file(path)


def test_fingerprint_differs_for_different_sized_files(tmp_path):
    a = tmp_path / "a.log"
    b = tmp_path / "b.log"
    a.write_text("short\n")
    b.write_text("a much longer line of content\n")

    assert fingerprint_file(a) != fingerprint_file(b)
