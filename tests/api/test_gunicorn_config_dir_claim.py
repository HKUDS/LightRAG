"""The Gunicorn master claims the directory its WORKERS will ask for.

The claim is held by the open file description, which ``fork`` shares: the
master takes it in ``on_starting``, before forking, and every worker counts
itself into the inherited claim rather than opening a second descriptor.

That only works while both name the SAME directory. When the workers moved to
``config_dir`` and the master kept claiming ``WORKING_DIR`` keyed on
``LIGHTRAG_KV_STORAGE``, the master's claim was on a path nobody inherits:
each worker opened its own descriptor on the config lock, the first won, and
every other worker was refused at startup with
``WorkingDirectoryInUseError``. A custom ``LIGHTRAG_CONFIG_DIR`` was left with
no master-held claim at all.

Both sides now resolve through ``configuration_selection_from_env``, and these
tests pin that they agree. See docs/design/ConfigurationStorageContract.md.
"""

from __future__ import annotations

import importlib
import sys

import pytest

pytestmark = pytest.mark.offline


@pytest.fixture
def gunicorn_config(monkeypatch):
    # The module parses argv at import time; pytest's own would fail it.
    monkeypatch.setattr(sys, "argv", ["lightrag-gunicorn"])
    module = importlib.import_module("lightrag.api.gunicorn_config")
    monkeypatch.setattr(module, "workers", 4, raising=False)
    return module


@pytest.fixture(autouse=True)
def _clean_claims():
    from lightrag.kg import working_dir_lock as wdl

    held = dict(wdl._claims)
    wdl._claims.clear()
    yield
    for claim in wdl._claims.values():
        try:
            claim.handle.close()
        except OSError:  # pragma: no cover - defensive
            pass
    wdl._claims.clear()
    wdl._claims.update(held)


def _env(monkeypatch, tmp_path, **overrides):
    monkeypatch.setenv("WORKING_DIR", str(tmp_path))
    for name in ("LIGHTRAG_CONFIG_STORAGE", "LIGHTRAG_CONFIG_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage")
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)


def test_the_master_claims_the_default_config_dir(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    from lightrag.config_store import default_config_dir
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    _env(monkeypatch, tmp_path)
    gunicorn_config.on_starting(object())
    capsys.readouterr()

    assert holds_working_dir_lock(default_config_dir(str(tmp_path))) is True
    assert holds_working_dir_lock(str(tmp_path)) is False


def test_the_master_follows_a_custom_config_dir(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    """The case that had no master claim at all before."""
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    elsewhere = tmp_path / "conf"
    _env(monkeypatch, tmp_path, LIGHTRAG_CONFIG_DIR=str(elsewhere))
    gunicorn_config.on_starting(object())
    capsys.readouterr()

    assert holds_working_dir_lock(str(elsewhere)) is True


def test_a_server_backed_configuration_claims_nothing(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    """A file-backed BUSINESS store no longer drags the claim in with it: the
    configuration storage is what the claim is for."""
    from lightrag.config_store import default_config_dir
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    _env(monkeypatch, tmp_path, LIGHTRAG_CONFIG_STORAGE="PGKVStorage")
    gunicorn_config.on_starting(object())
    capsys.readouterr()

    assert holds_working_dir_lock(default_config_dir(str(tmp_path))) is False
    assert holds_working_dir_lock(str(tmp_path)) is False


def test_the_master_and_a_worker_agree_on_the_directory(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    """The property the whole thing rests on, asserted directly rather than
    inferred: what the master claims is what a ``LightRAG`` instance in a
    worker would ask for."""
    import numpy as np

    from lightrag import LightRAG
    from lightrag.config_store import configuration_selection_from_env
    from lightrag.kg.shared_storage import (
        finalize_share_data,
        initialize_share_data,
    )
    from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface

    class _StubTokenizer(TokenizerInterface):
        def encode(self, content: str) -> list[int]:
            return [ord(c) for c in content]

        def decode(self, tokens: list[int]) -> str:
            return "".join(chr(t) for t in tokens)

    async def _llm(prompt, **kwargs):  # pragma: no cover - never called
        return ""

    async def _embed(texts, **kwargs):  # pragma: no cover - never called
        return np.zeros((len(texts), 8), dtype=np.float32)

    elsewhere = tmp_path / "conf"
    _env(monkeypatch, tmp_path, LIGHTRAG_CONFIG_DIR=str(elsewhere))

    master_storage, master_dir = configuration_selection_from_env(
        kv_storage="JsonKVStorage", working_dir=str(tmp_path)
    )

    initialize_share_data(workers=1)
    try:
        rag = LightRAG(
            working_dir=str(tmp_path),
            workspace="tenant",
            llm_model_func=_llm,
            embedding_func=EmbeddingFunc(
                embedding_dim=8, max_token_size=1024, func=_embed, model_name="m"
            ),
            tokenizer=Tokenizer("stub", _StubTokenizer()),
        )
        assert rag.config_dir == master_dir
        assert rag.config_storage == master_storage
    finally:
        finalize_share_data()
    capsys.readouterr()


def test_an_unusable_selection_is_refused_in_the_master(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    """Redis is not in the category. Refusing in the master fails the server
    once, clearly, instead of once per worker."""
    _env(monkeypatch, tmp_path, LIGHTRAG_CONFIG_STORAGE="RedisKVStorage")

    with pytest.raises(ValueError, match="RedisKVStorage"):
        gunicorn_config.on_starting(object())
    capsys.readouterr()


def test_on_exit_releases_what_on_starting_claimed(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    elsewhere = tmp_path / "conf"
    _env(monkeypatch, tmp_path, LIGHTRAG_CONFIG_DIR=str(elsewhere))

    gunicorn_config.on_starting(object())
    assert holds_working_dir_lock(str(elsewhere)) is True
    gunicorn_config.on_exit(object())
    capsys.readouterr()

    assert holds_working_dir_lock(str(elsewhere)) is False


def test_the_master_follows_the_parsed_working_dir_over_the_environment(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    """``--working-dir`` is parsed into ``global_args`` and handed to every
    worker's ``LightRAG``; nothing writes it back to ``WORKING_DIR``.

    A master that read the environment claimed the directory the workers do
    NOT use: its claim was on a path nobody inherits, the first worker took
    the real one for itself, and every worker after it was refused. Same
    defect as the one this file opens with, reached through the CLI instead
    of through the config category.
    """
    from lightrag.config_store import default_config_dir
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    parsed = tmp_path / "from-cli"
    parsed.mkdir()
    environment = tmp_path / "from-env"
    environment.mkdir()

    _env(monkeypatch, environment)
    monkeypatch.setattr(gunicorn_config, "working_dir", str(parsed), raising=False)

    assert gunicorn_config.resolved_working_dir() == str(parsed)

    gunicorn_config.on_starting(object())
    capsys.readouterr()

    assert holds_working_dir_lock(default_config_dir(str(parsed))) is True
    assert holds_working_dir_lock(default_config_dir(str(environment))) is False


def test_the_environment_is_the_fallback_when_nothing_parsed_a_directory(
    gunicorn_config, monkeypatch, tmp_path
):
    """A master started some other way than ``lightrag-gunicorn`` sets no
    module attribute, and the environment is all there is."""
    _env(monkeypatch, tmp_path)
    monkeypatch.setattr(gunicorn_config, "working_dir", None, raising=False)

    assert gunicorn_config.resolved_working_dir() == str(tmp_path)


@pytest.mark.parametrize("custom_dir", [False, True])
def test_redis_default_claims_json_configuration_directory(
    gunicorn_config, monkeypatch, tmp_path, capsys, custom_dir
):
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    config_dir = tmp_path / ("custom_config" if custom_dir else "_lightrag_config")
    overrides = {"LIGHTRAG_KV_STORAGE": "RedisKVStorage"}
    if custom_dir:
        overrides["LIGHTRAG_CONFIG_DIR"] = str(config_dir)
    _env(monkeypatch, tmp_path, **overrides)
    gunicorn_config.on_starting(object())
    assert holds_working_dir_lock(str(config_dir)) is True
    capsys.readouterr()
