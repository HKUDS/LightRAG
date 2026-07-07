"""Unit tests for the parser registry (lightrag.parser.registry)."""

import subprocess
import sys


from lightrag.parser import registry


def test_supported_engines_are_user_selectable_only():
    engines = registry.supported_parser_engines()
    assert engines == frozenset(
        {"native", "legacy", "mineru", "docling", "paddleocr_vl"}
    )
    # Internal format handlers are registered but not user-selectable.
    assert "reuse" not in engines
    assert "passthrough" not in engines


def test_suffix_capabilities_lookup():
    assert "pdf" in registry.suffix_capabilities("mineru")
    assert "pdf" in registry.suffix_capabilities("paddleocr_vl")
    assert "jpg" in registry.suffix_capabilities("paddleocr_vl")
    assert registry.suffix_capabilities("native") == frozenset(
        {"docx", "md", "textpack"}
    )
    assert registry.suffix_capabilities("unknown-engine") == frozenset()


def test_get_parser_unknown_returns_none():
    assert registry.get_parser("does-not-exist") is None


def test_get_parser_instances_are_cached():
    a = registry.get_parser("native")
    b = registry.get_parser("native")
    assert a is b and a is not None


def test_register_parser_roundtrips_concurrency():
    # The registrant bakes any env override into a concrete ``concurrency``
    # value at registration; the spec carries it verbatim.
    spec = registry.ParserSpec(
        engine_name="third-party-engine",
        impl="x:Y",
        suffixes=frozenset({"foo"}),
        queue_group="third-party-engine",
        concurrency=7,
    )
    try:
        registry.register_parser(spec)
        snapshot = registry.parser_specs_snapshot()
        assert snapshot["third-party-engine"].concurrency == 7
        assert snapshot["third-party-engine"].queue_group == "third-party-engine"
    finally:
        # Keep the module-level registry clean for other tests.
        registry._REGISTRY.pop("third-party-engine", None)


def test_mineru_endpoint_requirement_tracks_api_mode(monkeypatch):
    monkeypatch.setenv("MINERU_API_MODE", "official")
    assert registry.engine_endpoint_requirement("mineru") == "MINERU_API_TOKEN"
    monkeypatch.setenv("MINERU_API_MODE", "local")
    assert registry.engine_endpoint_requirement("mineru") == "MINERU_LOCAL_ENDPOINT"


def test_paddleocr_vl_endpoint_requirement(monkeypatch):
    monkeypatch.delenv("PADDLEOCR_VL_API_MODE", raising=False)
    monkeypatch.delenv("PADDLEOCR_VL_API_TOKEN", raising=False)
    assert registry.engine_endpoint_configured("paddleocr_vl") is False
    assert registry.engine_endpoint_requirement("paddleocr_vl") == (
        "PADDLEOCR_VL_API_TOKEN"
    )
    monkeypatch.setenv("PADDLEOCR_VL_API_TOKEN", "token")
    assert registry.engine_endpoint_configured("paddleocr_vl") is True

    monkeypatch.setenv("PADDLEOCR_VL_API_MODE", "local")
    monkeypatch.delenv("PADDLEOCR_VL_LOCAL_ENDPOINT", raising=False)
    assert registry.engine_endpoint_configured("paddleocr_vl") is False
    assert registry.engine_endpoint_requirement("paddleocr_vl") == (
        "PADDLEOCR_VL_LOCAL_ENDPOINT"
    )
    monkeypatch.setenv("PADDLEOCR_VL_LOCAL_ENDPOINT", "http://local-paddle")
    assert registry.engine_endpoint_configured("paddleocr_vl") is True


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
