"""Dynamic loading of a user-supplied parser file.

This is the "configurable parser" extension point: a user points the CLI at
their own ``parser.py`` instead of picking a name from a built-in registry.
That file can implement a log format from scratch, or subclass an existing
one (e.g. the project's own default, below) and override just what it needs
to (query normalization via ``query_id``, field mappings, etc).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

#: The LongEval-Sci project's own parser (scripts/longeval-sci-v2/parser.py),
#: loaded the same way as any user-supplied ``--parser`` file — it's an
#: ordinary example of the extension point, not a built-in class inside this
#: package.
DEFAULT_PARSER_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "longeval-sci-v2" / "parser.py"
)

_module_cache: dict[Path, ModuleType] = {}


def load_parser_module(path: Path | str | None) -> ModuleType:
    """Load and cache the module a ``parser.py`` file defines.

    Most callers want :func:`load_parser`; use this directly when you need
    more than the ready-made ``parser`` instance, e.g. the class itself to
    subclass or construct with different arguments. Caching by resolved path
    means two calls for the same file return objects from the *same*
    module, so classes from it compare with ``isinstance`` as expected.
    """
    path = Path(path).resolve() if path is not None else DEFAULT_PARSER_PATH
    if path not in _module_cache:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"could not load a module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _module_cache[path] = module
    return _module_cache[path]


def load_parser(path: Path | str | None):
    """Load the ``parser`` object from ``path``, or the project default.

    ``path`` must be a Python file with a top-level ``parser`` variable
    exposing ``iter_searches(lines)``, ``iter_interactions(lines)``, and
    ``query_id(text)`` (see ``scripts/longeval-sci-v2/parser.py``), e.g.::

        from src.importing.loader import load_parser

        class MyParser:
            def __init__(self):
                self._inner = load_parser(None)  # delegate to the default

            def query_id(self, query: str) -> str:
                return my_normalization(query)

            def iter_searches(self, lines):
                return self._inner.iter_searches(lines)

            def iter_interactions(self, lines):
                return self._inner.iter_interactions(lines)

        parser = MyParser()
    """
    module = load_parser_module(path)
    try:
        return module.parser
    except AttributeError:
        raise ValueError(
            f"{path} must define a top-level `parser` object, e.g. "
            "`parser = LongEvalSciParser()`"
        ) from None
