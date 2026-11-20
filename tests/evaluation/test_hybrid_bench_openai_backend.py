"""OpenAI-compatible backend wiring pins, offline (no network, no services).

Construction pins inspect the built partials; call-parameter pins drive the
REAL product binding against a recording fake OpenAI client (the factory is
monkeypatched, so both the client constructor kwargs and the wire call kwargs
are asserted). Credentials in this file are obvious unit-test placeholders.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lightrag.evaluation.hybrid_bench.cli import _base_config, parse_args
from lightrag.evaluation.hybrid_bench.rag_setup import (
    BenchConfig,
    build_embedding_func,
    build_llm_func,
)

pytestmark = pytest.mark.offline

_BINDING_ENV_VARS = (
    "LLM_BINDING_HOST",
    "LLM_BINDING_API_KEY",
    "EMBEDDING_BINDING_HOST",
    "EMBEDDING_BINDING_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
)


@pytest.fixture(autouse=True)
def _clean_binding_env(monkeypatch):
    for var in _BINDING_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _cli_argv(*extra) -> list[str]:
    return ["--dataset-dir", "d", "--output-dir", "o", *extra]


def _openai_config(**overrides) -> BenchConfig:
    return BenchConfig(
        llm_backend="openai",
        embedding_backend="openai",
        llm_model="chat-model",
        embedding_model="embed-model",
        **overrides,
    )


@pytest.mark.parametrize(
    ("flag_host", "env_host", "expected"),
    [
        ("https://flag.example/v1", "https://env.example/v1", "https://flag.example/v1"),
        ("", "https://env.example/v1", "https://env.example/v1"),
        ("", None, None),
    ],
    ids=["flag-wins", "env-fallback", "unset-means-sdk-default"],
)
def test_openai_llm_host_precedence(monkeypatch, flag_host, env_host, expected):
    if env_host is not None:
        monkeypatch.setenv("LLM_BINDING_HOST", env_host)
    func = build_llm_func(_openai_config(llm_host=flag_host))
    assert func.args == ("chat-model",)
    assert func.keywords["base_url"] == expected


@pytest.mark.parametrize(
    ("binding_key", "openai_key", "expected"),
    [
        ("binding-key", "openai-key", "binding-key"),
        (None, "openai-key", "openai-key"),
        (None, None, None),
    ],
    ids=["binding-wins", "openai-fallback", "unset-defers-to-binding"],
)
def test_openai_llm_api_key_precedence(monkeypatch, binding_key, openai_key, expected):
    if binding_key is not None:
        monkeypatch.setenv("LLM_BINDING_API_KEY", binding_key)
    if openai_key is not None:
        monkeypatch.setenv("OPENAI_API_KEY", openai_key)
    assert build_llm_func(_openai_config()).keywords["api_key"] == expected


@pytest.mark.parametrize(
    ("thinking", "expected_extra_body"),
    [
        (None, None),
        (True, {"enable_thinking": True}),
        (False, {"enable_thinking": False}),
    ],
    ids=["unset-sends-nothing", "explicit-on", "explicit-off"],
)
def test_openai_llm_thinking_flag_rides_extra_body(thinking, expected_extra_body):
    func = build_llm_func(_openai_config(llm_enable_thinking=thinking))
    assert func.keywords.get("extra_body") == expected_extra_body


@pytest.mark.parametrize("cap", [None, 2048, 4096])
def test_openai_llm_num_predict_maps_to_max_tokens(cap):
    func = build_llm_func(_openai_config(llm_num_predict=cap))
    if cap is None:
        assert "max_tokens" not in func.keywords
    else:
        assert func.keywords["max_tokens"] == cap


def test_openai_llm_timeout_comes_from_request_timeout():
    func = build_llm_func(_openai_config(request_timeout=123.0))
    assert func.keywords["timeout"] == 123.0


def test_thinking_flag_does_not_leak_into_ollama_path():
    func = build_llm_func(BenchConfig(llm_enable_thinking=False))
    assert "extra_body" not in func.keywords
    assert func.keywords["host"] == "http://127.0.0.1:11434"


def test_openai_embedding_func_attrs_and_partial():
    embedding = build_embedding_func(_openai_config())
    assert embedding.embedding_dim == 1024
    assert embedding.model_name == "embed_model"
    keywords = embedding.func.keywords
    assert keywords["model"] == "embed-model"
    assert keywords["embedding_dim"] == 1024
    assert keywords["base_url"] is None
    assert keywords["api_key"] is None


def test_openai_embedding_host_and_key_come_from_embedding_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "https://embed.example/v1")
    monkeypatch.setenv("EMBEDDING_BINDING_API_KEY", "embed-key")
    monkeypatch.setenv("LLM_BINDING_HOST", "https://llm.example/v1")
    monkeypatch.setenv("LLM_BINDING_API_KEY", "llm-key")
    keywords = build_embedding_func(_openai_config()).func.keywords
    assert keywords["base_url"] == "https://embed.example/v1"
    assert keywords["api_key"] == "embed-key"


def test_openai_embedding_dim_is_configurable():
    embedding = build_embedding_func(_openai_config(embedding_dim=768))
    assert embedding.embedding_dim == 768
    assert embedding.func.keywords["embedding_dim"] == 768


class _RecordingCompletions:
    """Fake chat.completions endpoint recording every wire call."""

    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", reasoning_content=""),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )


class _RecordingEmbeddings:
    """Fake embeddings endpoint recording every wire call."""

    def __init__(self, dim: int):
        self.calls = []
        self._dim = dim

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[0.0] * self._dim)
                for _ in kwargs["input"]
            ],
            usage=None,
        )


@pytest.fixture
def openai_client_spy(monkeypatch):
    """Replace the product's client factory with a recording fake."""
    import lightrag.llm.openai as openai_mod

    constructed = []
    completions = _RecordingCompletions()
    embeddings = _RecordingEmbeddings(dim=1024)

    async def close():
        pass

    def fake_factory(**kwargs):
        constructed.append(kwargs)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
            embeddings=embeddings,
            close=close,
        )

    monkeypatch.setattr(openai_mod, "create_openai_async_client", fake_factory)
    return SimpleNamespace(
        constructed=constructed, completions=completions, embeddings=embeddings
    )


async def test_openai_llm_wire_params_through_real_binding(
    monkeypatch, openai_client_spy
):
    monkeypatch.setenv("LLM_BINDING_API_KEY", "unit-test-key")
    config = _openai_config(
        llm_host="https://gateway.example/v1",
        llm_enable_thinking=False,
        llm_num_predict=4096,
    )
    func = build_llm_func(config)
    answer = await func("hello", system_prompt=None, history_messages=[])
    assert answer == "ok"
    assert len(openai_client_spy.constructed) == 1
    client_kwargs = openai_client_spy.constructed[0]
    assert client_kwargs["base_url"] == "https://gateway.example/v1"
    assert client_kwargs["api_key"] == "unit-test-key"
    assert len(openai_client_spy.completions.calls) == 1
    call = openai_client_spy.completions.calls[0]
    assert call["model"] == "chat-model"
    assert call["max_tokens"] == 4096
    assert call["extra_body"] == {"enable_thinking": False}
    assert call["messages"][-1] == {"role": "user", "content": "hello"}


async def test_openai_embedding_wire_params_through_real_binding(
    monkeypatch, openai_client_spy
):
    monkeypatch.setenv("EMBEDDING_BINDING_API_KEY", "unit-test-key")
    config = _openai_config(embedding_host="https://gateway.example/v1")
    embedding = build_embedding_func(config)
    vectors = await embedding.func(["a", "b"])
    assert vectors.shape == (2, 1024)
    assert len(openai_client_spy.constructed) == 1
    client_kwargs = openai_client_spy.constructed[0]
    assert client_kwargs["base_url"] == "https://gateway.example/v1"
    assert client_kwargs["api_key"] == "unit-test-key"
    assert len(openai_client_spy.embeddings.calls) == 1
    call = openai_client_spy.embeddings.calls[0]
    assert call["model"] == "embed-model"
    assert call["input"] == ["a", "b"]
    assert call["dimensions"] == 1024


@pytest.mark.parametrize(
    ("flag", "attr", "value"),
    [
        ("--llm-backend", "llm_backend", "openai"),
        ("--embedding-backend", "embedding_backend", "openai"),
    ],
)
def test_cli_openai_backend_choices_reach_config(flag, attr, value):
    args = parse_args(_cli_argv(flag, value))
    assert getattr(_base_config(args, "ws"), attr) == value


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], None),
        (["--llm-enable-thinking"], True),
        (["--no-llm-enable-thinking"], False),
    ],
    ids=["unset", "on", "off"],
)
def test_cli_thinking_flag_reaches_config(flags, expected):
    args = parse_args(_cli_argv(*flags))
    assert args.llm_enable_thinking is expected
    assert _base_config(args, "ws").llm_enable_thinking is expected


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--llm-backend", "azure"), ("--embedding-backend", "vllm")],
)
def test_cli_unknown_backend_rejected(flag, value):
    with pytest.raises(SystemExit) as exc:
        parse_args(_cli_argv(flag, value))
    assert exc.value.code == 2