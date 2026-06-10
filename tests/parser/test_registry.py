"""Unit tests for the parser registry (lightrag.parser.registry)."""

import subprocess
import sys

import pytest

from lightrag.parser import registry


def test_supported_engines_are_user_selectable_only():
    engines = registry.supported_parser_engines()
    assert engines == frozenset({"native", "legacy", "mineru", "docling"})
    # Internal format handlers are registered but not user-selectable.
    assert "reuse" not in engines
    assert "passthrough" not in engines


def test_suffix_capabilities_lookup():
    assert "pdf" in registry.suffix_capabilities("mineru")
    assert registry.suffix_capabilities("native") == frozenset({"docx"})
    assert registry.suffix_capabilities("unknown-engine") == frozenset()


def test_get_parser_unknown_returns_none():
    assert registry.get_parser("does-not-exist") is None


def test_get_parser_instances_are_cached():
    a = registry.get_parser("native")
    b = registry.get_parser("native")
    assert a is b and a is not None


def test_register_parser_requires_default_concurrency_with_env():
    bad = registry.ParserSpec(
        engine_name="bad-engine",
        impl="x:Y",
        suffixes=frozenset(),
        concurrency_env="MAX_PARALLEL_PARSE_BAD",
    )
    with pytest.raises(ValueError):
        registry.register_parser(bad)


def test_mineru_endpoint_requirement_tracks_api_mode(monkeypatch):
    monkeypatch.setenv("MINERU_API_MODE", "official")
    assert registry.engine_endpoint_requirement("mineru") == "MINERU_API_TOKEN"
    monkeypatch.setenv("MINERU_API_MODE", "local")
    assert registry.engine_endpoint_requirement("mineru") == "MINERU_LOCAL_ENDPOINT"


def test_capability_queries_do_not_import_parser_impls():
    """Importing the registry + querying capabilities must not pull a parser
    implementation (and therefore httpx). Run in a clean subprocess so other
    tests' imports don't pollute sys.modules."""
    code = (
        "import sys; import lightrag.parser.registry as r; "
        "r.supported_parser_engines(); r.suffix_capabilities('mineru'); "
        "r.engine_endpoint_configured('docling'); "
        "assert 'httpx' not in sys.modules, 'httpx leaked'; "
        "assert 'lightrag.parser.external.mineru.parser' not in sys.modules; "
        "print('clean')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout
