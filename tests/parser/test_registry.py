"""Unit tests for the parser registry (lightrag.parser.registry)."""

import os
import subprocess
import sys

import pytest

from lightrag.parser import registry

pytestmark = pytest.mark.offline


def test_supported_engines_are_user_selectable_only():
    engines = registry.supported_parser_engines()
    assert engines == frozenset({"native", "legacy", "mineru", "docling"})
    # Internal format handlers are registered but not user-selectable.
    assert "reuse" not in engines
    assert "passthrough" not in engines


def test_suffix_capabilities_lookup():
    assert "pdf" in registry.suffix_capabilities("mineru")
    assert registry.suffix_capabilities("native") == frozenset(
        {"docx", "md", "textpack"}
    )
    assert registry.suffix_capabilities("unknown-engine") == frozenset()


def test_docling_additional_suffixes_are_routable():
    env = os.environ.copy()
    env["DOCLING_ENDPOINT"] = "http://docling.test"
    env["DOCLING_ADDITIONAL_SUFFIXES"] = "doc, .PPT, csv"
    code = (
        "from lightrag.parser import registry; "
        "from lightrag.parser.routing import validate_parser_routing_config; "
        "expected = {'doc', 'ppt', 'csv'}; "
        "assert expected <= registry.suffix_capabilities('docling'); "
        "assert expected <= registry.available_engine_suffixes(); "
        "[validate_parser_routing_config(f'{suffix}:docling') "
        "for suffix in expected]"
    )

    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr


def test_docling_additional_suffixes_follow_late_env(monkeypatch):
    monkeypatch.delenv("DOCLING_ADDITIONAL_SUFFIXES", raising=False)
    baseline = registry.suffix_capabilities("docling")

    monkeypatch.setenv("DOCLING_ENDPOINT", "http://docling.test")
    monkeypatch.setenv("DOCLING_ADDITIONAL_SUFFIXES", " .LateFmt ")

    assert "latefmt" in registry.suffix_capabilities("docling")
    assert "latefmt" in registry.available_engine_suffixes()

    monkeypatch.delenv("DOCLING_ADDITIONAL_SUFFIXES")
    assert registry.suffix_capabilities("docling") == baseline


def test_mineru_additional_suffixes_are_routable(monkeypatch):
    """A MinerU deployment can declare the formats its own endpoint handles.

    ``MINERU_LOCAL_ENDPOINT`` is set because ``available_engine_suffixes`` gates
    on ``_mineru_endpoint_configured()``, which is mode-dependent.
    """
    from lightrag.parser.routing import (
        ParserRoutingConfigError,
        validate_parser_routing_config,
    )

    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://mineru.test:8000")
    assert "ppt" not in registry.suffix_capabilities("mineru")
    with pytest.raises(ParserRoutingConfigError, match="ppt:mineru"):
        validate_parser_routing_config("ppt:mineru")

    monkeypatch.setenv("MINERU_ADDITIONAL_SUFFIXES", " .DOC , xls, ppt")

    expected = {"doc", "xls", "ppt"}
    assert expected <= registry.parser_specs_snapshot()["mineru"].suffixes
    assert expected <= registry.suffix_capabilities("mineru")
    assert expected <= registry.available_engine_suffixes()

    # The startup validator accepts rules that only these suffixes make valid.
    validate_parser_routing_config("doc:mineru;xls:mineru;ppt:mineru")


def test_engine_suffix_env_vars_do_not_cross_talk(monkeypatch):
    """Each spec reads only its own ``extra_suffixes_env``.

    Guards against a copy-paste error in either spec's env name, which would
    silently hand one engine the other's deployment configuration.
    """
    docling_baseline = registry.suffix_capabilities("docling")
    mineru_baseline = registry.suffix_capabilities("mineru")

    monkeypatch.setenv("MINERU_ADDITIONAL_SUFFIXES", "mineruonly")
    assert "mineruonly" in registry.suffix_capabilities("mineru")
    assert registry.suffix_capabilities("docling") == docling_baseline

    monkeypatch.delenv("MINERU_ADDITIONAL_SUFFIXES")
    monkeypatch.setenv("DOCLING_ADDITIONAL_SUFFIXES", "doclingonly")
    assert "doclingonly" in registry.suffix_capabilities("docling")
    assert registry.suffix_capabilities("mineru") == mineru_baseline


def test_spec_suffixes_is_the_whole_capability(monkeypatch):
    """``spec.suffixes`` alone must answer "what can this engine parse?".

    Consumers read the field (``available_engine_suffixes``) or the one-line
    accessor over it (``suffix_capabilities``); neither knows an env var
    exists. If the extension ever moves back out of the field into a caller,
    one of those two paths silently loses the operator's suffixes.
    """
    monkeypatch.setenv("DOCLING_ADDITIONAL_SUFFIXES", "doc")
    spec = registry.parser_specs_snapshot()["docling"]
    assert "doc" in spec.suffixes
    assert spec.suffixes == registry.suffix_capabilities("docling")


def test_extra_suffixes_env_is_declarative_for_any_engine(monkeypatch):
    """The mechanism is a spec field, not a docling-specific branch."""
    monkeypatch.setenv("MYENGINE_EXTRA_SUFFIXES", "foo, .BAR")
    spec = registry.ParserSpec(
        engine_name="third-party-engine",
        impl="x:Y",
        suffixes=frozenset({"baz"}),
        extra_suffixes_env="MYENGINE_EXTRA_SUFFIXES",
    )
    try:
        registry.register_parser(spec)
        assert registry.suffix_capabilities("third-party-engine") == frozenset(
            {"baz", "foo", "bar"}
        )
    finally:
        registry._REGISTRY.pop("third-party-engine", None)


def test_spec_without_extra_suffixes_env_ignores_the_variable(monkeypatch):
    monkeypatch.setenv("DOCLING_ADDITIONAL_SUFFIXES", "doc")
    assert "doc" not in registry.suffix_capabilities("native")
    assert "doc" not in registry.suffix_capabilities("legacy")


def test_suffixes_stays_a_required_argument():
    """A registrant who forgets ``suffixes`` must not get an engine that
    silently matches nothing — the descriptor must not act as a default."""
    with pytest.raises(TypeError, match="suffixes"):
        registry.ParserSpec(engine_name="x", impl="a:B")


def test_suffixes_are_frozen_on_assignment():
    """A registrant passing a mutable set cannot have it mutated afterwards."""
    mutable = {"foo"}
    spec = registry.ParserSpec(engine_name="x", impl="a:B", suffixes=mutable)
    mutable.add("bar")
    assert spec.suffixes == frozenset({"foo"})
    assert isinstance(spec.suffixes, frozenset)


#: Every engine that declares an ``extra_suffixes_env``, with its baseline set.
_SUFFIX_ENV_ENGINES = [
    ("DOCLING_ADDITIONAL_SUFFIXES", "docling", registry._DOCLING_SUFFIXES),
    ("MINERU_ADDITIONAL_SUFFIXES", "mineru", registry._MINERU_SUFFIXES),
]


@pytest.mark.parametrize("env_name,engine,baseline", _SUFFIX_ENV_ENGINES)
@pytest.mark.parametrize("value", ["*.doc", "doc;ppt", "doc ppt"])
def test_malformed_env_suffixes_are_excluded_and_reported(
    monkeypatch, env_name, engine, baseline, value
):
    """A token that can never equal ``Path.suffix`` must not be admitted, and
    must be surfaced instead of leaving the operator's intent silently unmet."""
    from lightrag.parser.routing import (
        ParserRoutingConfigError,
        validate_parser_suffix_env_vars,
    )

    monkeypatch.setenv(env_name, value)
    assert registry.suffix_capabilities(engine) == baseline
    assert registry.malformed_env_suffixes() == {env_name: (value,)}
    with pytest.raises(ParserRoutingConfigError, match=env_name):
        validate_parser_suffix_env_vars()


@pytest.mark.parametrize("env_name,engine,baseline", _SUFFIX_ENV_ENGINES)
def test_wellformed_env_suffixes_pass_startup_validation(
    monkeypatch, env_name, engine, baseline
):
    from lightrag.parser.routing import validate_parser_suffix_env_vars

    monkeypatch.setenv(env_name, " .DOC , ppt ,")
    assert registry.malformed_env_suffixes() == {}
    validate_parser_suffix_env_vars()

    monkeypatch.delenv(env_name)
    assert registry.suffix_capabilities(engine) == baseline
    assert registry.malformed_env_suffixes() == {}
    validate_parser_suffix_env_vars()


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
