"""Unit tests for third-party parser discovery (lightrag.parser.plugins)."""

import pytest

from lightrag.parser import plugins, registry


class _FakeEntryPoint:
    """Stand-in for importlib.metadata.EntryPoint (name/value/load)."""

    def __init__(self, name, value, fn):
        self.name = name
        self.value = value
        self._fn = fn

    def load(self):
        if isinstance(self._fn, Exception):
            raise self._fn
        return self._fn


@pytest.fixture(autouse=True)
def _reset_loaded_flag(monkeypatch):
    """Each test starts from the not-yet-loaded state."""
    monkeypatch.setattr(plugins, "_loaded", False)


@pytest.fixture
def third_party_cleanup():
    """Remove engines a test registered so the module-level registry stays
    clean for other tests."""
    names = []
    yield names
    for name in names:
        registry._REGISTRY.pop(name, None)


def _install_eps(monkeypatch, eps):
    def _fake_entry_points(*, group):
        assert group == plugins.ENTRY_POINT_GROUP
        return list(eps)

    monkeypatch.setattr(plugins, "entry_points", _fake_entry_points)


def test_load_discovers_and_registers(monkeypatch, third_party_cleanup):
    def _register():
        registry.register_parser(
            registry.ParserSpec(
                engine_name="plug-engine",
                impl="x:Y",
                suffixes=frozenset({"foo"}),
                queue_group="plug-engine",
                concurrency=1,
            )
        )

    _install_eps(monkeypatch, [_FakeEntryPoint("plug", "pkg.mod:register", _register)])
    third_party_cleanup.append("plug-engine")

    assert plugins.load_third_party_parsers() == ["plug"]
    assert "plug-engine" in registry.supported_parser_engines()


def test_load_is_idempotent_per_process(monkeypatch):
    calls = {"n": 0}

    def _register():
        calls["n"] += 1

    _install_eps(monkeypatch, [_FakeEntryPoint("plug", "pkg.mod:register", _register)])

    assert plugins.load_third_party_parsers() == ["plug"]
    # Second call is a no-op (server lifespan + CLI may both call it).
    assert plugins.load_third_party_parsers() == []
    assert calls["n"] == 1
    # force=True re-runs (test escape hatch).
    assert plugins.load_third_party_parsers(force=True) == ["plug"]
    assert calls["n"] == 2


def test_broken_plugin_is_skipped_not_fatal(monkeypatch, third_party_cleanup):
    def _good_register():
        registry.register_parser(
            registry.ParserSpec(
                engine_name="good-engine",
                impl="x:Y",
                suffixes=frozenset({"bar"}),
                queue_group="good-engine",
                concurrency=1,
            )
        )

    _install_eps(
        monkeypatch,
        [
            _FakeEntryPoint("broken", "bad.mod:register", ImportError("boom")),
            _FakeEntryPoint("good", "ok.mod:register", _good_register),
        ],
    )
    third_party_cleanup.append("good-engine")

    # The broken plugin is logged + skipped; the good one still loads.
    assert plugins.load_third_party_parsers() == ["good"]
    assert "good-engine" in registry.supported_parser_engines()


def test_cli_sees_entry_point_engine(monkeypatch, third_party_cleanup):
    """End-to-end: a plugin-registered engine becomes a valid --engine choice
    in the debug CLI (which calls load_third_party_parsers in main)."""
    from lightrag.parser import cli

    def _register():
        registry.register_parser(
            registry.ParserSpec(
                engine_name="cli-plug-engine",
                impl="x:Y",
                suffixes=frozenset({"baz"}),
                queue_group="cli-plug-engine",
                concurrency=1,
            )
        )

    _install_eps(
        monkeypatch, [_FakeEntryPoint("cliplug", "pkg.mod:register", _register)]
    )
    third_party_cleanup.append("cli-plug-engine")

    # Missing input file -> exit 1 AFTER argparse accepted the engine choice
    # (an unknown engine would exit 2 from argparse instead).
    rc = cli.main(["/nonexistent/x.baz", "--engine", "cli-plug-engine"])
    assert rc == 1
