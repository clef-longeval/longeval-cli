from __future__ import annotations

import pytest

from src.importing.loader import load_parser
from tests.conftest import LongEvalSciParser


def test_load_parser_defaults_to_the_project_parser_when_no_path_given():
    parser = load_parser(None)

    assert isinstance(parser, LongEvalSciParser)


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
