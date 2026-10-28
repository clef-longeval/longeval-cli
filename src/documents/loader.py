"""Loading a document source from a Python file.

Mirrors ``importing/loader.py``'s "configurable parser" extension point:
``--source`` points at a file exposing a top-level ``source`` object, an
instance of a :class:`~src.documents.source.BaseDocumentSource` subclass
(see ``examples/longeval-sci-v2/core_source.py``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from .source import BaseDocumentSource

_module_cache: dict[Path, ModuleType] = {}


def load_document_source_module(path: Path | str) -> ModuleType:
    """Load and cache the module a source *file* defines.

    Caching by resolved path means two calls for the same file return
    objects from the *same* module, so classes from it compare with
    ``isinstance`` as expected.
    """
    path = Path(path).resolve()
    if path not in _module_cache:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"could not load a module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _module_cache[path] = module
    return _module_cache[path]


def load_document_source(path: Path | str) -> BaseDocumentSource:
    """Load the ``source`` object a file defines.

    It has to subclass :class:`BaseDocumentSource`: callers rely on the
    base's ``NotImplementedError`` to tell which optional hooks it supports.
    """
    if not Path(path).is_file():
        raise ValueError(f"document source file not found: {path}")
    module = load_document_source_module(path)
    try:
        source = module.source
    except AttributeError:
        raise ValueError(
            f"{path} must define a top-level `source` object, e.g. "
            "`source = CoreDocumentSource()`"
        ) from None
    if not isinstance(source, BaseDocumentSource):
        raise ValueError(
            f"{path}: `source` must be a BaseDocumentSource, "
            f"got {type(source).__name__}"
        )
    return source
