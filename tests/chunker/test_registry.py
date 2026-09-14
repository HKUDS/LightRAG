"""Closed-set selection, lazy imports and isolated plugin discovery."""

import asyncio
import logging
import sys
import subprocess
from pathlib import Path
from dataclasses import asdict, dataclass, replace
from importlib import metadata
from types import SimpleNamespace

import pytest

from lightrag.chunker import plugins, registry

pytestmark = pytest.mark.offline


def test_registry_import_does_not_load_builtin_or_plugin_implementations():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from lightrag.chunker.registry import ChunkerSpec, register_chunker; register_chunker(ChunkerSpec('demo', 'absent:chunk', '1', 'Demo')); assert not any(name in sys.modules for name in ('lightrag.chunker.token_size', 'lightrag.chunker.recursive_character', 'lightrag.chunker.semantic_vector', 'lightrag.chunker.paragraph_semantic', 'absent'))",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_DUPLICATES", set())
    monkeypatch.setattr(plugins, "_loaded", False)
    monkeypatch.setattr(plugins, "_failures", [])
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: [])
    monkeypatch.setattr(logging.getLogger("lightrag"), "propagate", True)


def spec(name="acme", **kwargs):
    return registry.ChunkerSpec(
        name=name,
        impl="test_chunker_impl:chunk",
        version="1",
        description="Example chunker",
        **kwargs,
    )


def install_impl(monkeypatch, callback):
    monkeypatch.setitem(
        sys.modules, "test_chunker_impl", SimpleNamespace(chunk=callback)
    )


def ep(name, register):
    return SimpleNamespace(
        name=name,
        value=f"{name}:register",
        dist=SimpleNamespace(name=f"dist-{name}", version="1"),
        load=lambda: register,
    )


def test_registration_imports_no_implementation(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("registration imported an implementation")

    monkeypatch.setattr(registry.importlib, "import_module", unexpected)
    registry.register_chunker(spec(), origin="package-a")
    registry.register_chunker(spec("another"), origin="package-b")
    assert registry.registered_chunker_names() == ("acme", "another")


@pytest.mark.parametrize(
    "name",
    [
        "",
        "Upper",
        "a.b",
        "a:b",
        "-a",
        "_a",
        "a b",
        "a,b",
        "a=b",
        "a(b)",
        "a" * 65,
        "f",
        "r",
        "v",
        "p",
        "c",
        "fixed_token",
        "recursive_character",
        "semantic_vector",
        "paragraph_semantic",
    ],
)
def test_reject_invalid_or_reserved_names(name):
    with pytest.raises(ValueError):
        registry.register_chunker(spec(name))


@pytest.mark.parametrize("name", ["0", "a_b-c", "a" * 64])
def test_valid_names(name):
    registry.register_chunker(spec(name))
    assert name in registry.registered_chunker_names()


@pytest.mark.parametrize("value", ["pkg:func", "pkg.func", "/tmp/plugin", "pkg\\func"])
def test_operator_import_paths_have_actionable_error(value):
    with pytest.raises(ValueError, match="[Ii]mport paths.*entry.point"):
        registry.resolve_chunker(value)


def test_unknown_name_lists_available_names():
    registry.register_chunker(spec())
    with pytest.raises(ValueError, match="unknown.*missing.*acme"):
        registry.resolve_chunker("missing")


def test_unset_leaves_callback_default_untouched():
    assert registry.resolve_chunker("") is None
    assert registry.resolve_chunker(None) is None
    assert registry.chunker_identity(lambda *args: []) is None


def test_resolves_only_selected_implementation(monkeypatch):
    def callback(*args):
        return args

    install_impl(monkeypatch, callback)
    registry.register_chunker(spec())
    registry.register_chunker(
        registry.ChunkerSpec("broken", "does_not_exist:chunk", "1", "Broken")
    )
    bound = registry.resolve_chunker("acme")
    assert bound(1, 2, 3, 4, 5, 6) == (1, 2, 3, 4, 5, 6)
    assert registry.chunker_identity(bound) == {
        "name": "acme",
        "version": "1",
        "authoritative": False,
    }
    with pytest.raises(ValueError, match="broken.*does_not_exist"):
        registry.resolve_chunker("broken")


@pytest.mark.parametrize(
    "callback",
    [
        42,
        lambda a: [],
        lambda a, b, c, d, e, f, g: [],
        lambda a, b, c, d, e, f, *, required: [],
    ],
)
def test_selected_noncallable_or_wrong_arity_fails(monkeypatch, callback):
    install_impl(monkeypatch, callback)
    registry.register_chunker(spec())
    with pytest.raises(ValueError, match="acme"):
        registry.resolve_chunker("acme")


def test_non_introspectable_callable_is_allowed(monkeypatch):
    class Callback:
        __signature__ = "unavailable"

        def __call__(self, *args):
            return args

    install_impl(monkeypatch, Callback())
    registry.register_chunker(spec())
    assert registry.resolve_chunker("acme")(*range(6)) == tuple(range(6))


def test_duplicate_selected_fails_but_unselected_does_not(monkeypatch, caplog):
    install_impl(monkeypatch, lambda *args: [])
    registry.register_chunker(spec(), origin="first")
    registry.register_chunker(spec(), origin="second")
    assert "first" in caplog.text and "second" in caplog.text
    assert registry.resolve_chunker(None) is None
    registry.register_chunker(spec("other"))
    assert registry.resolve_chunker("other") is not None
    with pytest.raises(ValueError, match="duplicate.*acme"):
        registry.resolve_chunker("acme")


def test_discovery_once_and_failed_provider_rolls_back(monkeypatch, caplog):
    calls = []

    def good():
        calls.append("good")
        registry.register_chunker(spec())
        registry.register_chunker(spec("second"))

    def broken():
        registry.register_chunker(spec("partial"))
        registry.register_chunker(spec())
        raise RuntimeError("provider failed")

    monkeypatch.setattr(
        plugins,
        "entry_points",
        lambda **kwargs: [ep("a-good", good), ep("z-broken", broken)],
    )
    assert plugins.load_third_party_chunkers() == ["a-good"]
    assert plugins.load_third_party_chunkers() == []
    assert calls == ["good"]
    assert registry.registered_chunker_names() == ("acme", "second")
    assert not registry._DUPLICATES
    assert "z-broken" in caplog.text and "dist-z-broken" in caplog.text
    with pytest.raises(ValueError, match="unknown.*partial"):
        registry.resolve_chunker("partial")


def test_repeated_startup_reuses_discovery_but_retains_failure_diagnostics(
    monkeypatch, caplog
):
    calls = []
    install_impl(monkeypatch, lambda *args: args)

    def good():
        calls.append("registered")
        registry.register_chunker(spec())

    def broken():
        raise RuntimeError("unrelated provider failed")

    def discover(**kwargs):
        calls.append("discovered")
        return [ep("good", good), ep("broken", broken)]

    monkeypatch.setattr(plugins, "entry_points", discover)
    for _ in range(2):
        caplog.clear()
        assert plugins.load_and_resolve_chunker("acme")(*range(6)) == tuple(range(6))
        assert len(caplog.records) == 1
        assert "selected chunker 'acme' validated" in caplog.text
    assert calls == ["discovered", "registered"]


def test_startup_logs_one_line_and_no_selector_impact(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="lightrag")
    registry.log_chunker_selection(None)
    assert len(caplog.records) == 1
    caplog.clear()
    install_impl(monkeypatch, lambda *args: [])
    registry.register_chunker(spec(), origin="origin-package")
    registry.log_chunker_selection(registry.resolve_chunker("acme"))
    assert len(caplog.records) == 1
    assert all(
        s in caplog.text
        for s in ["acme", "origin-package", "no-selector", "Example chunker"]
    )


def test_async_callback_stays_on_loop_and_cannot_opt_into_executor(monkeypatch):
    async def callback(*args):
        return asyncio.get_running_loop()

    install_impl(monkeypatch, callback)
    registry.register_chunker(spec())

    async def run():
        assert (
            await registry.resolve_chunker("acme")(*range(6))
            is asyncio.get_running_loop()
        )

    asyncio.run(run())
    registry.register_chunker(spec("offload", executor_safe=True))
    with pytest.raises(ValueError, match="async.*executor"):
        registry.resolve_chunker("offload")


def test_executor_safe_uses_existing_bounded_pool(monkeypatch):
    import lightrag.utils as utils

    calls = []

    def callback(*args):
        return args

    install_impl(monkeypatch, callback)

    async def offload(fn, *args):
        calls.append(fn)
        return fn(*args)

    monkeypatch.setattr(utils, "run_in_chunking_executor", offload)
    registry.register_chunker(spec(executor_safe=True))
    assert asyncio.run(registry.resolve_chunker("acme")(*range(6))) == tuple(range(6))
    assert calls == [callback]


@pytest.mark.parametrize(
    "changes",
    [
        {"name": None},
        {"impl": ""},
        {"impl": None},
        {"version": 1},
        {"description": ""},
        {"description": None},
        {"description": "two\nlines"},
        {"executor_safe": "true"},
    ],
)
def test_invalid_spec_metadata_is_rejected(changes):
    with pytest.raises(ValueError):
        registry.register_chunker(replace(spec(), **changes))


@pytest.mark.parametrize("value", [42, "UPPER", " name", "a,b", "x" * 65])
def test_invalid_selection_is_rejected(value):
    with pytest.raises(ValueError, match="CUSTOM_CHUNKER"):
        registry.resolve_chunker(value)


@pytest.mark.parametrize(
    "reference",
    ["no-colon", ":missing_module", "test_chunker_impl:", "test_chunker_impl:missing"],
)
def test_malformed_reference_or_missing_attribute_fails_at_resolution(
    monkeypatch, reference
):
    install_impl(monkeypatch, lambda *args: [])
    registry.register_chunker(replace(spec(), impl=reference))
    with pytest.raises(ValueError, match="cannot load selected chunker"):
        registry.resolve_chunker("acme")


def test_async_callable_object_cannot_offload(monkeypatch):
    class Callback:
        async def __call__(self, *args):
            return args

    install_impl(monkeypatch, Callback())
    registry.register_chunker(spec(executor_safe=True))
    with pytest.raises(ValueError, match="async.*executor"):
        registry.resolve_chunker("acme")


def test_binding_remains_callable_in_dataclass_snapshots(monkeypatch):
    @dataclass
    class Config:
        chunking_func: object

    install_impl(monkeypatch, lambda *args: args)
    registry.register_chunker(spec())
    snapshot = asdict(Config(registry.resolve_chunker("acme")))
    assert snapshot["chunking_func"](*range(6)) == tuple(range(6))


def test_real_distribution_entry_point_imports_only_selected_impl(
    tmp_path, monkeypatch
):
    package = tmp_path / "registry_test_plugin"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "plugin.py").write_text(
        "from lightrag.chunker.registry import ChunkerSpec, register_chunker\n"
        "def register():\n"
        "    for name in ('first', 'second'):\n"
        "        register_chunker(ChunkerSpec(name, f'registry_test_plugin.{name}:chunk', '1', name))\n",
        encoding="utf-8",
    )
    for name in ("first", "second"):
        (package / f"{name}.py").write_text(
            "def chunk(*args):\n    return args\n", encoding="utf-8"
        )
    info = tmp_path / "registry_test_plugin-1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: registry-test-plugin\nVersion: 1\n",
        encoding="utf-8",
    )
    (info / "entry_points.txt").write_text(
        "[lightrag.chunkers]\nregistry-test = registry_test_plugin.plugin:register\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    entries = metadata.Distribution.at(info).entry_points
    monkeypatch.setattr(
        plugins, "entry_points", lambda *, group: entries.select(group=group)
    )
    try:
        assert plugins.load_third_party_chunkers() == ["registry-test"]
        assert "registry_test_plugin.first" not in sys.modules
        assert "registry_test_plugin.second" not in sys.modules
        assert registry.resolve_chunker("first")(*range(6)) == tuple(range(6))
        assert "registry_test_plugin.first" in sys.modules
        assert "registry_test_plugin.second" not in sys.modules
    finally:
        for name in (
            "registry_test_plugin.first",
            "registry_test_plugin.second",
            "registry_test_plugin.plugin",
            "registry_test_plugin",
        ):
            sys.modules.pop(name, None)
