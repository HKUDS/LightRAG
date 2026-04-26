"""Offline tests for role-specific LLM runtime configuration."""

import asyncio
import logging
from argparse import Namespace

import numpy as np
import pytest

from lightrag import LightRAG, ROLES, RoleLLMConfig
from lightrag.llm.binding_options import OpenAILLMOptions
from lightrag.utils import EmbeddingFunc, Tokenizer, priority_limit_async_func_call


pytestmark = pytest.mark.offline


@pytest.fixture
def lightrag_logger_propagating(monkeypatch):
    """Force the lightrag logger to propagate so caplog can capture records."""
    monkeypatch.setattr(logging.getLogger("lightrag"), "propagate", True)


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 16)


async def _base_llm(*args, **kwargs) -> str:
    return "base"


_ROLE_FIELD_SUFFIXES = (
    ("_llm_model_func", "func"),
    ("_llm_model_kwargs", "kwargs"),
    ("_llm_model_max_async", "max_async"),
    ("_llm_timeout", "timeout"),
)


def _make_rag(tmp_path, **kwargs) -> LightRAG:
    """Create a LightRAG for role tests.

    Accepts both the canonical ``role_llm_configs={...}`` style and shorthand
    ``{role}_llm_model_func`` / ``{role}_llm_model_kwargs`` etc. keyword
    arguments. Shorthand kwargs are folded into ``role_llm_configs`` so the
    body of each test reads clearly.
    """
    role_configs: dict[str, RoleLLMConfig] = {}
    explicit = kwargs.pop("role_llm_configs", None)
    if explicit is not None:
        for name, cfg in explicit.items():
            role_configs[name] = (
                cfg if isinstance(cfg, RoleLLMConfig) else RoleLLMConfig(**dict(cfg))
            )

    for spec in ROLES:
        bucket = {}
        for suffix, target in _ROLE_FIELD_SUFFIXES:
            key = f"{spec.name}{suffix}"
            if key in kwargs:
                bucket[target] = kwargs.pop(key)
        if bucket:
            existing = role_configs.get(spec.name)
            if existing is not None:
                for target, value in bucket.items():
                    if getattr(existing, target) is None:
                        setattr(existing, target, value)
            else:
                role_configs[spec.name] = RoleLLMConfig(**bucket)

    if role_configs:
        kwargs["role_llm_configs"] = role_configs

    return LightRAG(
        working_dir=str(tmp_path / "role-runtime"),
        workspace="role-runtime",
        llm_model_func=_base_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=16,
            max_token_size=4096,
            func=_mock_embedding,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        **kwargs,
    )


def _captured_messages(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def _role_config_headers(caplog) -> list[str]:
    return [
        message
        for message in _captured_messages(caplog)
        if "Role LLM Configuration" in message
    ]


def _clear_role_provider_env(monkeypatch, role: str, options_cls) -> None:
    for arg_item in options_cls.args_env_name_type_value():
        monkeypatch.delenv(f"{role.upper()}_{arg_item['env_name']}", raising=False)


ROLE_MAX_ASYNC_ENV_KEYS = (
    "MAX_ASYNC_EXTRACT_LLM",
    "MAX_ASYNC_KEYWORD_LLM",
    "MAX_ASYNC_QUERY_LLM",
    "MAX_ASYNC_VLM_LLM",
)


@pytest.mark.asyncio
async def test_priority_queue_stats_track_running_and_queued():
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_func(value: str, **_kwargs):
        started.set()
        await release.wait()
        return value

    wrapped = priority_limit_async_func_call(1, queue_name="test LLM func")(slow_func)

    first = asyncio.create_task(wrapped("first"))
    await started.wait()
    second = asyncio.create_task(wrapped("second"))
    await asyncio.sleep(0.05)

    stats = await wrapped.get_queue_stats()
    assert stats["max_async"] == 1
    assert stats["running"] == 1
    assert stats["queued"] == 1
    assert stats["in_flight"] == 2
    assert stats["submitted_total"] == 2

    release.set()
    assert await asyncio.gather(first, second) == ["first", "second"]
    await asyncio.sleep(0)

    stats = await wrapped.get_queue_stats()
    assert stats["running"] == 0
    assert stats["queued"] == 0
    assert stats["completed_total"] == 2
    assert stats["rejected_total"] == 0

    await wrapped.shutdown()


@pytest.mark.asyncio
async def test_priority_queue_graceful_shutdown_timeout_falls_back_to_force(
    caplog, lightrag_logger_propagating
):
    started = asyncio.Event()

    async def stuck_func(value: str, **_kwargs):
        started.set()
        await asyncio.sleep(60)
        return value

    wrapped = priority_limit_async_func_call(1, queue_name="stuck LLM func")(stuck_func)

    in_flight = asyncio.create_task(wrapped("hold"))
    await started.wait()

    with caplog.at_level("WARNING", logger="lightrag"):
        await wrapped.shutdown(graceful=True, timeout=0.1)

    assert any(
        "Graceful drain timed out" in record.getMessage() for record in caplog.records
    )

    with pytest.raises(asyncio.CancelledError):
        await in_flight

    stats = await wrapped.get_queue_stats()
    assert stats["cancelled_total"] >= 1


@pytest.mark.asyncio
async def test_priority_queue_rejects_submissions_after_shutdown():
    async def fast_func(value: str, **_kwargs):
        return value

    wrapped = priority_limit_async_func_call(1, queue_name="reject LLM func")(fast_func)

    assert await wrapped("warmup") == "warmup"
    await wrapped.shutdown()

    with pytest.raises(RuntimeError, match="Queue is shutting down"):
        await wrapped("rejected")

    stats = await wrapped.get_queue_stats()
    assert stats["rejected_total"] == 1


def test_role_max_async_defaults_inherit_base(tmp_path, monkeypatch):
    for env_key in ROLE_MAX_ASYNC_ENV_KEYS:
        monkeypatch.delenv(env_key, raising=False)

    rag = _make_rag(tmp_path, llm_model_max_async=10)

    assert rag._role_llm_states["extract"].max_async is None
    assert rag._role_llm_states["keyword"].max_async is None
    assert rag._role_llm_states["query"].max_async is None
    assert rag._role_llm_states["vlm"].max_async is None
    assert rag._get_effective_role_llm_max_async("extract") == 10
    assert rag._get_effective_role_llm_max_async("keyword") == 10
    assert rag._get_effective_role_llm_max_async("query") == 10
    assert rag._get_effective_role_llm_max_async("vlm") == 10


def test_role_max_async_env_override_keeps_other_roles_inherited(tmp_path, monkeypatch):
    for env_key in ROLE_MAX_ASYNC_ENV_KEYS:
        monkeypatch.delenv(env_key, raising=False)
    monkeypatch.setenv("MAX_ASYNC_EXTRACT_LLM", "7")

    rag = _make_rag(tmp_path, llm_model_max_async=10)

    assert rag._role_llm_states["extract"].max_async == 7
    assert rag._role_llm_states["keyword"].max_async is None
    assert rag._role_llm_states["query"].max_async is None
    assert rag._role_llm_states["vlm"].max_async is None
    assert rag._get_effective_role_llm_max_async("extract") == 7
    assert rag._get_effective_role_llm_max_async("keyword") == 10
    assert rag._get_effective_role_llm_max_async("query") == 10
    assert rag._get_effective_role_llm_max_async("vlm") == 10


@pytest.mark.asyncio
async def test_role_functions_are_isolated_and_vlm_present(tmp_path):
    rag = _make_rag(tmp_path)

    funcs = [
        rag.llm_model_func,
        rag.role_llm_funcs["extract"],
        rag.role_llm_funcs["keyword"],
        rag.role_llm_funcs["query"],
        rag.role_llm_funcs["vlm"],
    ]
    assert all(callable(func) for func in funcs)
    assert len({id(func) for func in funcs}) == len(funcs)


@pytest.mark.asyncio
async def test_no_role_configs_keeps_base_raw_and_wraps_each_role(tmp_path):
    """Regression: base llm_model_func must stay raw; each role still gets
    its own queue wrapper around the base func when no override is given."""
    rag = _make_rag(tmp_path)

    # Base is the user-provided callable, untouched by any wrapper.
    assert rag.llm_model_func is _base_llm

    # Every role has a wrapped (queue-managed) func that's distinct from base.
    for spec in ROLES:
        wrapped = rag.role_llm_funcs[spec.name]
        assert callable(wrapped)
        assert wrapped is not _base_llm

    # All four role wrappers are independent (separate queues).
    wrappers = [rag.role_llm_funcs[spec.name] for spec in ROLES]
    assert len({id(w) for w in wrappers}) == len(wrappers)

    # Calling any role wrapper hits the base function.
    assert await rag.role_llm_funcs["extract"]("p") == "base"
    assert await rag.role_llm_funcs["vlm"]("p") == "base"

    # get_llm_queue_status no longer reports a 'base' entry.
    status = await rag.get_llm_queue_status()
    assert "base" not in status
    assert set(status) == {spec.name for spec in ROLES}


@pytest.mark.asyncio
async def test_role_llm_configs_accepts_dict_form(tmp_path):
    """Init accepts plain dicts in role_llm_configs (auto-normalized to RoleLLMConfig)."""

    async def query_fn(*args, **kwargs):
        return "query-via-dict"

    rag = LightRAG(
        working_dir=str(tmp_path / "dict-form"),
        workspace="dict-form",
        llm_model_func=_base_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=16, max_token_size=4096, func=_mock_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        role_llm_configs={"query": {"func": query_fn, "max_async": 5}},
    )

    assert rag._role_llm_states["query"].raw_func is query_fn
    assert rag._role_llm_states["query"].max_async == 5
    # Roles not present in the dict still wrap the base function.
    assert rag._role_llm_states["extract"].raw_func is _base_llm
    assert await rag.role_llm_funcs["query"]("ping") == "query-via-dict"


def test_role_llm_configs_rejects_unknown_role_keys(tmp_path):
    with pytest.raises(ValueError, match="qurey"):
        _make_rag(tmp_path, role_llm_configs={"qurey": {}})


def test_role_llm_config_logs_once_on_init_with_metadata(
    tmp_path, caplog, lightrag_logger_propagating
):
    with caplog.at_level("INFO", logger="lightrag"):
        rag = _make_rag(
            tmp_path,
            role_llm_configs={
                "query": RoleLLMConfig(
                    max_async=7,
                    timeout=42,
                    metadata={
                        "binding": "openai",
                        "model": "gpt-test",
                        "host": "https://api.example.com/v1",
                        "api_key": "secret-key",
                        "provider_options": {
                            "temperature": 0.1,
                            "token": "nested-token",
                        },
                        "bedrock_aws_options": {
                            "region_name": "us-east-1",
                            "aws_secret_access_key": "aws-secret",
                        },
                    },
                )
            },
        )

    snapshot = rag.get_llm_role_config("query")
    assert snapshot["binding"] == "openai"
    assert snapshot["model"] == "gpt-test"
    assert snapshot["host"] == "https://api.example.com/v1"
    assert snapshot["max_async"] == 7
    assert snapshot["timeout"] == 42

    headers = _role_config_headers(caplog)
    assert len(headers) == 1
    assert "initialized" in headers[0]
    messages = "\n".join(_captured_messages(caplog))
    assert " - query: openai/gpt-test" in messages
    assert "max_async=7" in messages
    assert "timeout=42" in messages
    assert "secret-key" not in messages
    assert "nested-token" not in messages
    assert "aws-secret" not in messages


@pytest.mark.asyncio
async def test_role_specific_kwargs_and_fallback(tmp_path):
    extract_calls = []
    vlm_calls = []

    async def extract_func(*args, **kwargs):
        extract_calls.append(kwargs)
        return "extract"

    async def vlm_func(*args, **kwargs):
        vlm_calls.append(kwargs)
        return "vlm"

    rag = _make_rag(
        tmp_path,
        llm_model_kwargs={"shared": "base"},
        extract_llm_model_func=extract_func,
        extract_llm_model_kwargs={"shared": "extract", "tag": "extract"},
        vlm_llm_model_func=vlm_func,
        vlm_llm_model_kwargs={"shared": "vlm", "tag": "vlm"},
    )

    await rag.role_llm_funcs["extract"]("extract prompt")
    await rag.role_llm_funcs["keyword"]("keyword prompt")
    await rag.role_llm_funcs["vlm"]("vlm prompt")

    assert extract_calls[-1]["tag"] == "extract"
    assert extract_calls[-1]["shared"] == "extract"
    assert "hashing_kv" in extract_calls[-1]

    # Keyword role falls back to base kwargs when no role kwargs are configured.
    # We do not inspect base function internals, but the call must succeed.
    assert vlm_calls[-1]["tag"] == "vlm"
    assert vlm_calls[-1]["shared"] == "vlm"


@pytest.mark.asyncio
async def test_update_llm_role_config_rewraps_without_double_call(tmp_path):
    call_count = 0
    seen_tags = []

    async def query_func(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        seen_tags.append(kwargs.get("tag"))
        return "query"

    rag = _make_rag(
        tmp_path,
        query_llm_model_func=query_func,
        query_llm_model_kwargs={"tag": "v1"},
    )

    await rag.role_llm_funcs["query"]("first")
    assert call_count == 1
    assert seen_tags[-1] == "v1"

    for value in (3, 5, 7):
        rag.update_llm_role_config("query", max_async=value)
        await rag.role_llm_funcs["query"]("next")

    rag.update_llm_role_config("query", model_kwargs={"tag": "v2"})
    await rag.role_llm_funcs["query"]("final")

    assert call_count == 5
    assert seen_tags[-1] == "v2"
    assert rag._role_llm_states["query"].max_async == 7
    await rag.wait_for_retired_llm_queues()


@pytest.mark.asyncio
async def test_aupdate_llm_role_config_drains_old_queue(tmp_path):
    started = asyncio.Event()
    release = asyncio.Event()

    async def old_query_func(*args, **kwargs):
        started.set()
        await release.wait()
        return "old"

    async def new_query_func(*args, **kwargs):
        return "new"

    rag = _make_rag(tmp_path, query_llm_model_func=old_query_func)

    old_call = asyncio.create_task(rag.role_llm_funcs["query"]("old"))
    await started.wait()

    update_call = asyncio.create_task(
        rag.aupdate_llm_role_config("query", model_func=new_query_func)
    )
    await asyncio.sleep(0.05)
    assert not update_call.done()
    assert await rag.role_llm_funcs["query"]("new") == "new"

    release.set()
    assert await old_call == "old"
    await update_call


@pytest.mark.asyncio
async def test_sync_update_tracks_retired_queue_cleanup(tmp_path):
    async def query_func(*args, **kwargs):
        return "old"

    async def new_query_func(*args, **kwargs):
        return "new"

    rag = _make_rag(tmp_path, query_llm_model_func=query_func)

    assert await rag.role_llm_funcs["query"]("before") == "old"
    rag.update_llm_role_config("query", model_func=new_query_func)
    assert await rag.role_llm_funcs["query"]("after") == "new"

    await rag.wait_for_retired_llm_queues()
    assert not rag._retired_llm_queue_cleanup_tasks


def test_sync_update_without_event_loop_skips_cleanup(
    tmp_path, caplog, lightrag_logger_propagating
):
    async def query_func(*args, **kwargs):
        return "old"

    async def new_query_func(*args, **kwargs):
        return "new"

    rag = _make_rag(tmp_path, query_llm_model_func=query_func)

    with caplog.at_level("WARNING", logger="lightrag"):
        rag.update_llm_role_config("query", model_func=new_query_func)

    assert not rag._retired_llm_queue_cleanup_tasks
    assert any(
        "no event loop is running" in record.getMessage() for record in caplog.records
    )

    async def call_new() -> str:
        return await rag.role_llm_funcs["query"]("after")

    assert asyncio.run(call_new()) == "new"


@pytest.mark.asyncio
async def test_aupdate_llm_role_config_with_builder_drains_old_queue(tmp_path):
    started = asyncio.Event()
    release = asyncio.Event()

    def builder(role, meta):
        model_name = meta["model"]

        if model_name == "old-model":

            async def built_func(*args, **kwargs):
                started.set()
                await release.wait()
                return model_name
        else:

            async def built_func(*args, **kwargs):
                return model_name

        return built_func, None

    rag = _make_rag(tmp_path)
    rag.register_role_llm_builder(builder)
    rag.set_role_llm_metadata(
        "query",
        binding="openai",
        model="seed",
        host="https://seed",
        api_key="seed-key",
    )

    rag.update_llm_role_config("query", binding="openai", model="old-model")
    await rag.wait_for_retired_llm_queues()

    in_flight = asyncio.create_task(rag.role_llm_funcs["query"]("hold"))
    await started.wait()

    update_call = asyncio.create_task(
        rag.aupdate_llm_role_config("query", binding="openai", model="new-model")
    )
    await asyncio.sleep(0.05)
    assert not update_call.done()
    assert await rag.role_llm_funcs["query"]("hello") == "new-model"

    release.set()
    assert await in_flight == "old-model"
    await update_call
    assert not rag._retired_llm_queue_cleanup_tasks


@pytest.mark.asyncio
async def test_aupdate_llm_role_config_updates_cache_identity(tmp_path):
    async def query_func(*_args, **_kwargs):
        return "query"

    rag = _make_rag(tmp_path)
    rag.register_role_llm_builder(lambda _role, _meta: (query_func, {}))

    await rag.aupdate_llm_role_config(
        "query",
        binding="openai",
        model="gpt-cache-test",
        host="https://api.example.com/v1",
    )

    identity = rag._build_global_config()["llm_cache_identities"]["query"]

    assert identity == {
        "role": "query",
        "binding": "openai",
        "model": "gpt-cache-test",
        "host": "https://api.example.com/v1",
    }
    await rag.wait_for_retired_llm_queues()


@pytest.mark.asyncio
async def test_update_llm_role_config_with_builder_metadata(tmp_path):
    built_calls = []

    def builder(role: str, meta: dict):
        async def built_func(*args, **kwargs):
            built_calls.append(
                {"role": role, "meta": dict(meta), "kwargs": dict(kwargs)}
            )
            return f"{meta['model']}"

        return built_func, {
            "runtime_host": meta["host"],
            "provider_options": meta["provider_options"],
        }

    rag = _make_rag(tmp_path)
    rag.register_role_llm_builder(builder)
    rag.set_role_llm_metadata(
        "query",
        binding="openai",
        model="old-model",
        host="https://old-host",
        api_key="old-key",
        provider_options={"temperature": 0.1},
    )

    rag.update_llm_role_config(
        "query",
        binding="gemini",
        model="gemini-2.0-flash",
        host="https://new-host",
        api_key="new-key",
        provider_options={"temperature": 0.3, "top_k": 8},
    )

    result = await rag.role_llm_funcs["query"]("hello")
    assert result == "gemini-2.0-flash"
    assert built_calls[-1]["role"] == "query"
    assert built_calls[-1]["meta"]["binding"] == "gemini"
    assert built_calls[-1]["meta"]["model"] == "gemini-2.0-flash"
    assert built_calls[-1]["kwargs"]["runtime_host"] == "https://new-host"
    assert built_calls[-1]["kwargs"]["provider_options"]["top_k"] == 8


def test_update_llm_role_config_logs_after_success(
    tmp_path, caplog, lightrag_logger_propagating
):
    async def built_func(*args, **kwargs):
        return "ok"

    def builder(role: str, meta: dict):
        return built_func, None

    rag = _make_rag(
        tmp_path,
        role_llm_configs={
            "query": RoleLLMConfig(
                metadata={
                    "base_binding": "openai",
                    "binding": "openai",
                    "model": "old-model",
                    "host": "https://old.example/v1",
                },
            )
        },
    )
    rag.register_role_llm_builder(builder)

    caplog.clear()
    with caplog.at_level("INFO", logger="lightrag"):
        rag.update_llm_role_config(
            "query",
            binding="gemini",
            model="gemini-2.0-flash",
            host="https://gemini.example/v1",
            api_key="new-secret",
            provider_options={"token": "nested-token"},
        )

    headers = _role_config_headers(caplog)
    assert len(headers) == 1
    assert "updated: query" in headers[0]
    messages = "\n".join(_captured_messages(caplog))
    assert " - query: gemini/gemini-2.0-flash" in messages
    assert "host=https://gemini.example/v1" in messages
    assert "is_cross_provider" not in messages
    assert "new-secret" not in messages
    assert "nested-token" not in messages


@pytest.mark.asyncio
async def test_aupdate_llm_role_config_logs_after_success(
    tmp_path, caplog, lightrag_logger_propagating
):
    async def new_query_func(*args, **kwargs):
        return "new-query"

    rag = _make_rag(
        tmp_path,
        role_llm_configs={
            "query": RoleLLMConfig(
                metadata={
                    "binding": "openai",
                    "model": "old-model",
                    "host": "https://old.example/v1",
                },
            )
        },
    )

    caplog.clear()
    with caplog.at_level("INFO", logger="lightrag"):
        await rag.aupdate_llm_role_config(
            "query",
            model_func=new_query_func,
            max_async=2,
            timeout=180,
        )

    headers = _role_config_headers(caplog)
    assert len(headers) == 1
    assert "updated: query" in headers[0]
    messages = "\n".join(_captured_messages(caplog))
    assert " - query: openai/old-model" in messages
    assert "max_async=2" in messages
    assert "timeout=180" in messages


@pytest.mark.asyncio
async def test_llm_role_config_and_queue_status_are_observable(tmp_path):
    rag = _make_rag(tmp_path, query_llm_model_kwargs={"tag": "query"})
    rag.set_role_llm_metadata(
        "query",
        binding="openai",
        model="gpt-test",
        host="https://api.example.com/v1",
        api_key="secret-key",
        provider_options={"temperature": 0.1},
    )

    all_configs = rag.get_llm_role_config()
    assert set(all_configs) == {"extract", "keyword", "query", "vlm"}
    assert all_configs["query"]["binding"] == "openai"
    assert all_configs["query"]["model"] == "gpt-test"
    # Auth-bearing fields are dropped from the observability snapshot,
    # not masked — there is no "***" placeholder to confuse consumers.
    assert "api_key" not in all_configs["query"]["metadata"]
    assert all_configs["query"]["has_model_kwargs"] is True

    # Raw secrets remain accessible to in-process components that legitimately
    # need them (role builder, provider clients), but are not exposed via the
    # public observability method.
    assert rag._role_llm_states["query"].metadata["api_key"] == "secret-key"

    queue_status = await rag.get_llm_queue_status()
    assert set(queue_status) == {"extract", "keyword", "query", "vlm"}
    assert queue_status["query"]["available"] is True
    assert queue_status["query"]["queue_name"] == "query LLM func"


@pytest.mark.asyncio
async def test_embedding_and_rerank_queue_status_are_observable(tmp_path):
    async def rerank_func(*args, **kwargs):
        return []

    rag = _make_rag(tmp_path, rerank_model_func=rerank_func)

    embedding_status = await rag.get_embedding_queue_status()
    rerank_status = await rag.get_rerank_queue_status()

    assert embedding_status["available"] is True
    assert embedding_status["queue_name"] == "Embedding func"
    assert embedding_status["max_async"] == rag.embedding_func_max_async
    assert rerank_status["available"] is True
    assert rerank_status["queue_name"] == "Rerank func"
    assert rerank_status["max_async"] == rag.rerank_model_max_async


def test_get_llm_role_config_strips_bedrock_and_password_fields(tmp_path):
    rag = _make_rag(tmp_path)
    rag.set_role_llm_metadata(
        "query",
        binding="bedrock",
        model="claude-3",
        password="proxy-password",
        provider_options={
            "temperature": 0.1,
            "extra_body": {
                "safe_option": True,
                "api_key": "nested-api-key",
                "headers": {
                    "Authorization": "Bearer nested-token",
                    "X-API-Key": "nested-api-key",
                    "Accept": "application/json",
                },
                "tools": [
                    {"name": "safe-tool", "token": "nested-token"},
                ],
            },
        },
        bedrock_aws_options={
            "region_name": "us-east-1",
            "aws_access_key_id": "AKIA-secret",
            "aws_secret_access_key": "TOPSECRET",
            "aws_session_token": "SESSION",
        },
    )

    snapshot = rag.get_llm_role_config("query")
    assert "password" not in snapshot["metadata"]
    provider_options = snapshot["metadata"]["provider_options"]
    assert provider_options["temperature"] == 0.1
    extra_body = provider_options["extra_body"]
    assert extra_body["safe_option"] is True
    assert "api_key" not in extra_body
    assert extra_body["headers"] == {"Accept": "application/json"}
    assert extra_body["tools"] == [{"name": "safe-tool"}]
    bedrock = snapshot["metadata"]["bedrock_aws_options"]
    # Non-secret fields stay; auth-bearing fields are removed entirely.
    assert bedrock["region_name"] == "us-east-1"
    assert "aws_access_key_id" not in bedrock
    assert "aws_secret_access_key" not in bedrock
    assert "aws_session_token" not in bedrock

    # Mutating the returned snapshot must not affect the live state.
    snapshot["metadata"]["bedrock_aws_options"]["region_name"] = "tampered"
    assert (
        rag._role_llm_states["query"].metadata["bedrock_aws_options"]["region_name"]
        == "us-east-1"
    )


def test_get_llm_role_config_has_no_secret_escape_hatch(tmp_path):
    """Security guarantee: no parameter on get_llm_role_config can flip
    secret stripping off. This pins down the public-API contract so a future
    change can't accidentally re-introduce an ``include_secrets`` knob."""
    rag = _make_rag(tmp_path)
    rag.set_role_llm_metadata("query", api_key="super-secret")

    with pytest.raises(TypeError):
        rag.get_llm_role_config("query", include_secrets=True)  # type: ignore[call-arg]

    assert "api_key" not in rag.get_llm_role_config("query")["metadata"]


@pytest.mark.asyncio
async def test_cross_provider_update_does_not_inherit_base_kwargs(tmp_path):
    built_calls = []

    def builder(role: str, meta: dict):
        async def built_func(*args, **kwargs):
            built_calls.append(
                {"role": role, "meta": dict(meta), "kwargs": dict(kwargs)}
            )
            return "ok"

        return built_func, None

    rag = _make_rag(
        tmp_path,
        llm_model_kwargs={
            "host": "http://base-host:11434",
            "options": {"temperature": 0.1},
            "api_key": "base-key",
        },
    )
    rag.register_role_llm_builder(builder)
    rag.set_role_llm_metadata(
        "query",
        base_binding="ollama",
        binding="ollama",
        model="base-ollama",
        host="http://base-host:11434",
        api_key="base-key",
        provider_options={"temperature": 0.1},
        is_cross_provider=False,
    )

    rag.update_llm_role_config(
        "query",
        binding="openai",
        model="gpt-4o-mini",
        host="https://api.example.com/v1",
        api_key="role-key",
        provider_options={"temperature": 0.4},
    )

    await rag.role_llm_funcs["query"]("hello")
    call_kwargs = built_calls[-1]["kwargs"]
    assert call_kwargs["hashing_kv"] is not None
    assert "host" not in call_kwargs
    assert "options" not in call_kwargs
    assert "api_key" not in call_kwargs


@pytest.mark.asyncio
async def test_update_llm_role_config_rolls_back_on_failure(
    tmp_path, caplog, lightrag_logger_propagating
):
    rag = _make_rag(tmp_path, extract_llm_model_kwargs={"tag": "before"})
    original_raw = rag._role_llm_states["extract"].raw_func
    original_wrapped = rag.role_llm_funcs["extract"]
    original_kwargs = dict(rag.role_llm_kwargs["extract"])

    def failing_builder(role: str, meta: dict):
        raise RuntimeError("boom")

    rag.register_role_llm_builder(failing_builder)
    rag.set_role_llm_metadata(
        "extract",
        binding="openai",
        model="base-model",
        host="https://base",
        api_key="key",
        provider_options={"temperature": 0.1},
    )

    caplog.clear()
    with caplog.at_level("INFO", logger="lightrag"):
        with pytest.raises(RuntimeError, match="boom"):
            rag.update_llm_role_config(
                "extract",
                binding="gemini",
                provider_options={"temperature": 0.9},
            )

    assert rag._role_llm_states["extract"].raw_func is original_raw
    assert rag.role_llm_funcs["extract"] is original_wrapped
    assert rag.role_llm_kwargs["extract"] == original_kwargs
    assert not _role_config_headers(caplog)


def test_options_dict_for_role_inherits_same_provider(monkeypatch):
    args = Namespace(
        openai_llm_temperature=0.2,
        openai_llm_top_p=0.8,
        openai_llm_extra_body={"base": True},
    )
    _clear_role_provider_env(monkeypatch, "extract", OpenAILLMOptions)
    monkeypatch.setenv("EXTRACT_OPENAI_LLM_TEMPERATURE", "0.7")

    options = OpenAILLMOptions.options_dict_for_role(args, "extract")

    assert options["temperature"] == 0.7
    assert options["top_p"] == 0.8
    assert options["extra_body"] == {"base": True}


def test_options_dict_for_role_resets_cross_provider(monkeypatch):
    args = Namespace(
        openai_llm_temperature=0.2,
        openai_llm_top_p=0.8,
        openai_llm_extra_body={"base": True},
    )
    _clear_role_provider_env(monkeypatch, "query", OpenAILLMOptions)
    monkeypatch.setenv("QUERY_OPENAI_LLM_TOP_P", "0.6")

    options = OpenAILLMOptions.options_dict_for_role(
        args, "query", is_cross_provider=True
    )

    assert options == {"top_p": 0.6}


def test_options_dict_for_role_parses_nested_extra_body_cross_provider(monkeypatch):
    args = Namespace(openai_llm_extra_body={"base": True})
    _clear_role_provider_env(monkeypatch, "keyword", OpenAILLMOptions)
    monkeypatch.setenv(
        "KEYWORD_OPENAI_LLM_EXTRA_BODY",
        '{"chat_template_kwargs": {"enable_thinking": false}}',
    )

    options = OpenAILLMOptions.options_dict_for_role(
        args, "keyword", is_cross_provider=True
    )

    assert options["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.asyncio
async def test_vlm_role_supports_runtime_update(tmp_path):
    vlm_calls = []

    async def vlm_func(*args, **kwargs):
        vlm_calls.append(kwargs)
        return "vlm"

    rag = _make_rag(
        tmp_path,
        vlm_llm_model_func=vlm_func,
        vlm_llm_model_kwargs={"tag": "initial"},
    )

    await rag.role_llm_funcs["vlm"]("before")
    rag.update_llm_role_config(
        "vlm",
        model_kwargs={"tag": "updated"},
        max_async=2,
        timeout=240,
    )
    await rag.role_llm_funcs["vlm"]("after")

    assert vlm_calls[0]["tag"] == "initial"
    assert vlm_calls[1]["tag"] == "updated"
    assert rag._role_llm_states["vlm"].max_async == 2
    assert rag._role_llm_states["vlm"].timeout == 240
