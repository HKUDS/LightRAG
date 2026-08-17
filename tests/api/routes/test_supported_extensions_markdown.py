"""Upload allowlist exposes the native markdown suffixes (.md / .textpack).

``DocumentManager.supported_extensions`` derives the uploadable suffix set
live from the parser registry + routing: a suffix is advertised only when an
unhinted ``x.<suffix>`` routes to an engine that actually supports it. These
tests pin that ``.md`` and ``.textpack`` are uploadable (and routable), and
that ``.textpack`` resolves to the native engine.
"""

import importlib
import sys

import pytest

# document_routes parses argv at import time; guard it like the sibling tests.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
_parser_routing = importlib.import_module("lightrag.parser.routing")
sys.argv = _original_argv

DocumentManager = _document_routes.DocumentManager
resolve_file_parser_engine = _parser_routing.resolve_file_parser_engine


@pytest.fixture
def doc_manager(tmp_path, monkeypatch):
    # Clear any ambient LIGHTRAG_PARSER so the default routing applies:
    # .md -> legacy (opt-in for native), .textpack -> native.
    monkeypatch.delenv("LIGHTRAG_PARSER", raising=False)
    return DocumentManager(str(tmp_path))


def test_textpack_and_md_are_uploadable(doc_manager):
    extensions = doc_manager.supported_extensions
    assert ".textpack" in extensions
    assert ".md" in extensions


def test_bare_textpack_is_supported_file_via_native(doc_manager, monkeypatch):
    monkeypatch.delenv("LIGHTRAG_PARSER", raising=False)
    assert doc_manager.is_supported_file("note.textpack") is True
    assert resolve_file_parser_engine("note.textpack") == "native"


def test_md_is_supported_file(doc_manager, monkeypatch):
    monkeypatch.delenv("LIGHTRAG_PARSER", raising=False)
    assert doc_manager.is_supported_file("doc.md") is True
