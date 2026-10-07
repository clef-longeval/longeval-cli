from __future__ import annotations

import pytest

from src.importing.loader import load_parser
from tests.conftest import SAMPLE_PARSER_PATH, SampleParser


def test_load_parser_loads_the_parser_object_from_a_file():
    parser = load_parser(SAMPLE_PARSER_PATH)

    assert isinstance(parser, SampleParser)
    assert load_parser(SAMPLE_PARSER_PATH) is parser  # cached by path


def test_load_parser_loads_a_custom_file(tmp_path):
    path = tmp_path / "parser.py"
    path.write_text(
        "class MyParser:\n"
        "    def query_id(self, query):\n"
        "        return 'custom:' + query\n"
        "parser = MyParser()\n"
    )

    parser = load_parser(path)

    assert parser.query_id("hello") == "custom:hello"


def test_load_parser_requires_a_top_level_parser_object(tmp_path):
    path = tmp_path / "parser.py"
    path.write_text("not_parser = 1\n")

    with pytest.raises(ValueError):
        load_parser(path)
