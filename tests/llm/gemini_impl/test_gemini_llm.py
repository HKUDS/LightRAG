import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


class _FakeGeminiAPIError(Exception):
    """Stand-in for google.genai.errors.APIError."""

    def __init__(self, code):
        self.code = code
        super().__init__(f"fake gemini api error {code}")


class _FakeGeminiClientError(_FakeGeminiAPIError):
    """Stand-in for google.genai.errors.ClientError (4xx)."""


class _FakeGeminiServerError(_FakeGeminiAPIError):
    """Stand-in for google.genai.errors.ServerError (5xx)."""


class _FakeOutcome:
    """Stand-in for tenacity's Future outcome: a failed call carrying exc."""

    failed = True

    def __init__(self, exc):
        self._exc = exc

    def exception(self):
        return self._exc


class _FakeRetryState:
    """Minimal tenacity RetryCallState double: just enough for retry_base."""

    def __init__(self, exc):
        self.outcome = _FakeOutcome(exc)


def _load_gemini_module(monkeypatch, request):
    fake_pm = SimpleNamespace(
        is_installed=lambda name: True,
        install=lambda name: None,
    )

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeHttpOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_types = SimpleNamespace(
        GenerateContentConfig=FakeGenerateContentConfig,
        HttpOptions=FakeHttpOptions,
        # gemini.py compares candidates[0].finish_reason against this enum on
        # the non-streaming path (truncation marker); string sentinels stand in
        # for the real google.genai enum members.
        FinishReason=SimpleNamespace(MAX_TOKENS="MAX_TOKENS", STOP="STOP"),
    )
    fake_genai_errors = SimpleNamespace(
        APIError=_FakeGeminiAPIError,
        ClientError=_FakeGeminiClientError,
        ServerError=_FakeGeminiServerError,
    )
    fake_genai = SimpleNamespace(
        Client=lambda **kwargs: SimpleNamespace(kwargs=kwargs),
        errors=fake_genai_errors,
    )
    fake_google_module = ModuleType("google")
    fake_google_module.genai = fake_genai
    fake_api_exceptions = SimpleNamespace(
        InternalServerError=type("InternalServerError", (Exception,), {}),
        ServiceUnavailable=type("ServiceUnavailable", (Exception,), {}),
        ResourceExhausted=type("ResourceExhausted", (Exception,), {}),
        GatewayTimeout=type("GatewayTimeout", (Exception,), {}),
        BadGateway=type("BadGateway", (Exception,), {}),
        DeadlineExceeded=type("DeadlineExceeded", (Exception,), {}),
        Aborted=type("Aborted", (Exception,), {}),
        Unknown=type("Unknown", (Exception,), {}),
    )
    fake_google_api_core = ModuleType("google.api_core")
    fake_google_api_core.exceptions = fake_api_exceptions

    monkeypatch.setitem(sys.modules, "pipmaster", fake_pm)
    monkeypatch.setitem(sys.modules, "google", fake_google_module)
    monkeypatch.setitem(
        sys.modules,
        "google.genai",
        SimpleNamespace(types=fake_types, errors=fake_genai_errors),
    )
    monkeypatch.setitem(sys.modules, "google.api_core", fake_google_api_core)
    monkeypatch.setitem(sys.modules, "google.api_core.exceptions", fake_api_exceptions)

    # Force a fresh import of lightrag.llm.gemini against the fakes above,
    # and restore the original module (or absence) on teardown — otherwise
    # subsequent tests (e.g. tests/llm/test_asymmetric_embedding.py) inherit
    # this stubbed `genai.types` namespace and break with AttributeError on
    # types.EmbedContentConfig. Note: clearing sys.modules alone is not
    # enough — Python also caches the submodule as an attribute on the parent
    # package, and `from lightrag.llm import gemini` resolves via that
    # attribute. Both pointers must be cleared.
    parent = sys.modules.get("lightrag.llm")
    original_gemini = sys.modules.get("lightrag.llm.gemini")
    original_parent_attr = getattr(parent, "gemini", None) if parent else None
    sys.modules.pop("lightrag.llm.gemini", None)
    if parent is not None and hasattr(parent, "gemini"):
        delattr(parent, "gemini")

    def _restore_gemini():
        if original_gemini is not None:
            sys.modules["lightrag.llm.gemini"] = original_gemini
        else:
            sys.modules.pop("lightrag.llm.gemini", None)
        if parent is not None:
            if original_parent_attr is not None:
                parent.gemini = original_parent_attr
            elif hasattr(parent, "gemini"):
                delattr(parent, "gemini")

    request.addfinalizer(_restore_gemini)

    return importlib.import_module("lightrag.llm.gemini")


def _make_fake_gemini_response(regular_text="", thought_text="", finish_reason=None):
    parts = []
    if thought_text:
        parts.append(SimpleNamespace(text=thought_text, thought=True))
    if regular_text:
        parts.append(SimpleNamespace(text=regular_text, thought=False))

    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=parts),
                finish_reason=finish_reason,
            ),
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=1,
            candidates_token_count=2,
            total_token_count=3,
        ),
    )


_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


@pytest.mark.offline
@pytest.mark.parametrize(
    "response_format",
    [
        pytest.param(_ANSWER_SCHEMA, id="bare-schema"),
        pytest.param(
            {
                "type": "json_schema",
                "json_schema": {"name": "answer_payload", "schema": _ANSWER_SCHEMA},
            },
            id="openai-json-schema-wrapper",
        ),
    ],
)
def test_gemini_maps_schema_response_format_to_response_json_schema(
    monkeypatch, request, response_format
):
    gemini_module = _load_gemini_module(monkeypatch, request)

    config = gemini_module._build_generation_config(
        base_config=None,
        system_prompt=None,
        response_format=response_format,
    )

    assert config.kwargs["response_mime_type"] == "application/json"
    assert config.kwargs["response_json_schema"] == _ANSWER_SCHEMA
    assert "response_schema" not in config.kwargs


@pytest.mark.offline
def test_gemini_safety_settings_mapping_is_converted_to_sdk_list_shape(
    monkeypatch, request
):
    """GeminiLLMOptions documents safety_settings as a JSON object (a plain
    {category: threshold} mapping) and the server forwards that mapping into
    generation_config unchanged. google.genai.types.GenerateContentConfig
    requires safety_settings to be a list of {category, threshold} entries
    and rejects a bare mapping outright, so every completion call made with
    this documented option crashed before a request was ever sent."""
    gemini_module = _load_gemini_module(monkeypatch, request)

    config = gemini_module._build_generation_config(
        base_config={
            "safety_settings": {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
            }
        },
        system_prompt=None,
        response_format=None,
    )

    assert config.kwargs["safety_settings"] == [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
    ]


@pytest.mark.offline
def test_gemini_safety_settings_list_shape_passes_through_unchanged(
    monkeypatch, request
):
    """A caller already passing the SDK's native list shape must not be
    altered by the dict-to-list normalization."""
    gemini_module = _load_gemini_module(monkeypatch, request)

    native_shape = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}]
    config = gemini_module._build_generation_config(
        base_config={"safety_settings": native_shape},
        system_prompt=None,
        response_format=None,
    )

    assert config.kwargs["safety_settings"] == native_shape


@pytest.mark.offline
def test_gemini_safety_settings_single_sdk_entry_is_wrapped_not_unpacked(
    monkeypatch, request
):
    """A single SDK entry written as a dict must become a one-item list.
    Read as a category-to-threshold mapping it would turn into the bogus
    categories "category" and "threshold", which the SDK only warns about,
    so the misconfiguration would reach the API instead of failing here."""
    gemini_module = _load_gemini_module(monkeypatch, request)

    entry = {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}
    config = gemini_module._build_generation_config(
        base_config={"safety_settings": entry},
        system_prompt=None,
        response_format=None,
    )

    assert config.kwargs["safety_settings"] == [entry]


@pytest.mark.offline
def test_gemini_rejects_typed_response_format(monkeypatch, request):
    gemini_module = _load_gemini_module(monkeypatch, request)

    class FakeSchemaModel:
        """A class object where Gemini expects a dict schema (typed/Pydantic)."""

    with pytest.raises(TypeError, match="typed/Pydantic"):
        gemini_module._validate_gemini_response_format(FakeSchemaModel)


@pytest.mark.offline
@pytest.mark.parametrize(
    ("base_url", "expected_http_options"),
    [
        pytest.param(
            "https://generativelanguage.googleapis.com",
            {"timeout": 1234},
            id="default-service-root-is-not-a-custom-base-url",
        ),
        pytest.param(
            "https://proxy.example.com",
            {"base_url": "https://proxy.example.com", "timeout": 1234},
            id="custom-base-url-is-preserved",
        ),
    ],
)
def test_gemini_client_http_options(
    monkeypatch, request, base_url, expected_http_options
):
    gemini_module = _load_gemini_module(monkeypatch, request)
    gemini_module._get_gemini_client.cache_clear()
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)

    client = gemini_module._get_gemini_client("test-key", base_url, 1234)

    assert client.kwargs["api_key"] == "test-key"
    assert client.kwargs["http_options"].kwargs == expected_http_options


@pytest.mark.offline
@pytest.mark.parametrize("value", ["true", "True", "1"])
def test_gemini_vertexai_env_is_parsed_like_the_sdk(monkeypatch, request, value):
    """google-genai enables Vertex AI for "true" or "1", so LightRAG must too.

    Otherwise "1" is read as AI Studio mode: an API key is demanded, and the
    client built with it is switched to Vertex AI by the SDK anyway.
    """
    gemini_module = _load_gemini_module(monkeypatch, request)
    gemini_module._get_gemini_client.cache_clear()
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", value)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("LLM_BINDING_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    key = gemini_module._ensure_api_key(None)
    client = gemini_module._get_gemini_client(key, None)

    assert key == ""
    assert client.kwargs["vertexai"] is True
    assert client.kwargs["project"] == "test-project"
    assert "api_key" not in client.kwargs


@pytest.mark.offline
@pytest.mark.parametrize("value", ["false", "0", ""])
def test_gemini_vertexai_env_off_uses_api_key(monkeypatch, request, value):
    gemini_module = _load_gemini_module(monkeypatch, request)
    gemini_module._get_gemini_client.cache_clear()
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", value)

    client = gemini_module._get_gemini_client("test-key", None)

    assert client.kwargs["api_key"] == "test-key"
    assert client.kwargs["vertexai"] is False


@pytest.mark.offline
@pytest.mark.asyncio
async def test_gemini_streaming_structured_output_disables_cot(monkeypatch, request):
    gemini_module = _load_gemini_module(monkeypatch, request)

    fake_stream_response = _make_fake_gemini_response(
        regular_text='{"answer":"ok"}',
        thought_text="this should not be included",
    )
    fake_stream_response.usage_metadata.cached_content_token_count = 5

    async def _single_chunk_stream(response):
        yield response

    async def _fake_generate_content_stream(**kwargs):
        return _single_chunk_stream(fake_stream_response)

    fake_client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(
                generate_content_stream=_fake_generate_content_stream
            )
        )
    )

    monkeypatch.setattr(gemini_module, "_get_gemini_client", lambda *args: fake_client)
    recorded: list[dict] = []
    tracker = SimpleNamespace(add_usage=recorded.append)

    stream = await gemini_module.gemini_complete_if_cache(
        model="gemini-model",
        prompt="hello",
        stream=True,
        enable_cot=True,
        response_format={"type": "json_object"},
        api_key="test-key",
        token_tracker=tracker,
    )
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert "".join(chunks) == '{"answer":"ok"}'
    # The streamed usage reaches the tracker, cached prompt tokens included.
    assert recorded == [
        {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
            "cached_prompt_tokens": 5,
        }
    ]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_gemini_streaming_decodes_escapes_split_across_chunks(
    monkeypatch, request
):
    # Thoughts and answer are two texts; the thought's open escape is flushed
    # before </think>, the answer's at the end of the stream.
    gemini_module = _load_gemini_module(monkeypatch, request)
    responses = [
        _make_fake_gemini_response(thought_text=r"think \ud83d"),
        _make_fake_gemini_response(thought_text=r"\ude00 \u4f"),
        _make_fake_gemini_response(regular_text=r"ans\u00"),
        _make_fake_gemini_response(regular_text=r"41 \u4f"),
    ]

    async def _chunks():
        for response in responses:
            yield response

    async def _fake_generate_content_stream(**kwargs):
        return _chunks()

    fake_client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(
                generate_content_stream=_fake_generate_content_stream
            )
        )
    )
    monkeypatch.setattr(gemini_module, "_get_gemini_client", lambda *args: fake_client)

    stream = await gemini_module.gemini_complete_if_cache(
        model="gemini-model",
        prompt="hello",
        stream=True,
        enable_cot=True,
        api_key="test-key",
    )

    assert [chunk async for chunk in stream] == [
        "<think>",
        "think ",
        "😀 ",
        r"\u4f",
        "</think>",
        "ans",
        "A ",
        r"\u4f",
    ]


def _make_nonstreaming_client(fake_response):
    async def _fake_generate_content(**kwargs):
        return fake_response

    return SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=_fake_generate_content)
        )
    )


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reported", "expected_extra"),
    [
        pytest.param(None, {"total_tokens": 1}, id="missing-counts"),
        pytest.param(
            0, {"total_tokens": 0, "cached_prompt_tokens": 0}, id="zero-counts"
        ),
    ],
)
async def test_gemini_returns_text_with_partial_token_counts(
    monkeypatch, request, reported, expected_extra
):
    """google-genai usage counts may be None; the tracker must get ints.

    A reported zero, total or cached, is kept; a missing total is summed and a
    missing cached count is left out, as in the OpenAI binding. A plain
    recording tracker stands in for ``TokenTracker``: the binding accepts any
    object with ``add_usage(dict)``, and ``TokenTracker`` drops a None cached
    count itself, so only the raw dict shows what the binding sent.
    """
    gemini_module = _load_gemini_module(monkeypatch, request)
    response = _make_fake_gemini_response(regular_text="answer", finish_reason="STOP")
    response.usage_metadata.candidates_token_count = None
    response.usage_metadata.total_token_count = reported
    response.usage_metadata.cached_content_token_count = reported
    fake_client = _make_nonstreaming_client(response)
    monkeypatch.setattr(gemini_module, "_get_gemini_client", lambda *args: fake_client)
    recorded: list[dict] = []
    tracker = SimpleNamespace(add_usage=recorded.append)

    result = await gemini_module.gemini_complete_if_cache(
        model="gemini-model",
        prompt="Question",
        api_key="test-key",
        token_tracker=tracker,
    )

    assert result == "answer"
    assert recorded == [{"prompt_tokens": 1, "completion_tokens": 0, **expected_extra}]


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("finish_reason", "raw_json", "truncated"),
    [
        pytest.param("MAX_TOKENS", '{"entities":[{"name":"Ali', True, id="max-tokens"),
        pytest.param("STOP", '{"entities":[]}', False, id="stop"),
    ],
)
async def test_gemini_finish_reason_sets_truncation_marker(
    monkeypatch, request, finish_reason, raw_json, truncated
):
    """MAX_TOKENS truncation is returned for salvage but flagged uncacheable."""
    gemini_module = _load_gemini_module(monkeypatch, request)
    from lightrag.utils import is_truncated_response

    fake_client = _make_nonstreaming_client(
        _make_fake_gemini_response(regular_text=raw_json, finish_reason=finish_reason)
    )
    monkeypatch.setattr(gemini_module, "_get_gemini_client", lambda *args: fake_client)

    result = await gemini_module.gemini_complete_if_cache(
        model="gemini-model",
        prompt="Extract entities",
        api_key="test-key",
    )

    assert result == raw_json
    assert is_truncated_response(result) is truncated


@pytest.mark.offline
@pytest.mark.asyncio
async def test_gemini_thinking_only_max_tokens_response_raises(monkeypatch, request):
    """The #3597 shape on Gemini: the whole budget went to the thought trace.

    ``final_text`` is non-empty at the pre-strip check (it is
    ``<think>...</think>``), so only a check AFTER ``remove_think_tags`` sees
    that nothing usable came back. It used to be returned as an empty —
    truncation-flagged — success.

    The usage is still recorded: the raise used to precede the token
    accounting, so the request that spent its whole output budget on
    reasoning was the one missing from usage reporting (Codex review on
    PR #3607). Such a request's output is almost all thinking, which Gemini
    reports in ``thoughts_token_count`` outside ``candidates_token_count`` and
    bills as output, so both count as completion tokens, also in a computed
    total.
    """
    from lightrag.exceptions import EmptyTruncatedResponseError

    gemini_module = _load_gemini_module(monkeypatch, request)

    calls = {"n": 0}
    fake_response = _make_fake_gemini_response(
        thought_text="let me carefully consider the entities",
        finish_reason="MAX_TOKENS",
    )
    fake_response.usage_metadata.thoughts_token_count = 300
    fake_response.usage_metadata.total_token_count = None

    async def _counting_generate_content(**kwargs):
        calls["n"] += 1
        return fake_response

    fake_client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=_counting_generate_content)
        )
    )
    monkeypatch.setattr(gemini_module, "_get_gemini_client", lambda *args: fake_client)
    tracked: list[dict] = []
    tracker = SimpleNamespace(add_usage=tracked.append)

    # DECORATED call, deliberately: token-limit exhaustion is deterministic,
    # so it must escape the retry predicate and fail after ONE request —
    # retrying re-buys the same full-budget generation to fail identically
    # (Codex review on PR #3607).
    with pytest.raises(EmptyTruncatedResponseError) as excinfo:
        await gemini_module.gemini_complete_if_cache(
            model="gemini-model",
            prompt="Extract entities",
            api_key="test-key",
            enable_cot=True,
            token_tracker=tracker,
        )

    assert calls["n"] == 1, "a deterministic token-limit failure must not retry"
    assert tracked == [
        {"prompt_tokens": 1, "completion_tokens": 302, "total_tokens": 303}
    ], "the exhausted request's tokens vanished from usage accounting"
    message = str(excinfo.value)
    assert "Received empty content from Gemini API" in message
    assert "finish_reason=MAX_TOKENS" in message
    assert "budget consumed by reasoning" in message
    assert "GEMINI_LLM_MAX_OUTPUT_TOKENS" in message


@pytest.mark.offline
@pytest.mark.asyncio
async def test_gemini_no_content_at_all_names_the_token_limit(monkeypatch, request):
    """The pre-existing empty-response raise now says WHY it was empty."""
    gemini_module = _load_gemini_module(monkeypatch, request)

    fake_client = _make_nonstreaming_client(
        _make_fake_gemini_response(finish_reason="MAX_TOKENS")
    )
    monkeypatch.setattr(gemini_module, "_get_gemini_client", lambda *args: fake_client)

    from lightrag.exceptions import EmptyTruncatedResponseError

    with pytest.raises(EmptyTruncatedResponseError) as excinfo:
        await gemini_module.gemini_complete_if_cache(
            model="gemini-model",
            prompt="Extract entities",
            api_key="test-key",
        )

    message = str(excinfo.value)
    assert "finish_reason=MAX_TOKENS" in message
    assert "candidates_token_count=2" in message
    assert "GEMINI_LLM_MAX_OUTPUT_TOKENS" in message


@pytest.mark.offline
@pytest.mark.asyncio
async def test_gemini_non_length_empty_response_stays_retryable(monkeypatch, request):
    """The split's other half: an empty response that ended normally is a
    sampling artifact a fresh attempt can fix, so it must keep raising the
    retryable type — NOT the fail-fast one."""
    from lightrag.exceptions import EmptyTruncatedResponseError

    gemini_module = _load_gemini_module(monkeypatch, request)

    fake_client = _make_nonstreaming_client(
        _make_fake_gemini_response(finish_reason="STOP")
    )
    monkeypatch.setattr(gemini_module, "_get_gemini_client", lambda *args: fake_client)

    # __wrapped__ to skip the retry loop's real backoff sleeps.
    with pytest.raises(gemini_module.InvalidResponseError) as excinfo:
        await gemini_module.gemini_complete_if_cache.__wrapped__(
            model="gemini-model",
            prompt="Extract entities",
            api_key="test-key",
        )

    assert not isinstance(excinfo.value, EmptyTruncatedResponseError)
    assert "model produced no output" in str(excinfo.value)


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize("enable_cot", [True, False])
async def test_gemini_thinking_only_stop_response_stays_retryable(
    monkeypatch, request, enable_cot
):
    """A reasoning-only response that ended normally raises the retryable
    type with enable_cot=True, exactly as it does with enable_cot=False: the
    <think> wrapper must not turn it into an empty "answer" that skips the
    retry (and that the query path would return as the result)."""
    from lightrag.exceptions import EmptyTruncatedResponseError

    gemini_module = _load_gemini_module(monkeypatch, request)

    fake_client = _make_nonstreaming_client(
        _make_fake_gemini_response(thought_text="thinking", finish_reason="STOP")
    )
    monkeypatch.setattr(gemini_module, "_get_gemini_client", lambda *args: fake_client)

    # __wrapped__ to skip the retry loop's real backoff sleeps.
    with pytest.raises(gemini_module.InvalidResponseError) as excinfo:
        await gemini_module.gemini_complete_if_cache.__wrapped__(
            model="gemini-model",
            prompt="Extract entities",
            api_key="test-key",
            enable_cot=enable_cot,
        )

    assert not isinstance(excinfo.value, EmptyTruncatedResponseError)
    assert "model returned reasoning-only output" in str(excinfo.value)


def _make_capturing_client(fake_response):
    captured: dict = {}

    async def _fake_generate_content(**kwargs):
        captured.update(kwargs)
        return fake_response

    client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=_fake_generate_content)
        )
    )
    return client, captured


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("generation_kwargs", "expected_max_output_tokens"),
    [
        pytest.param({"max_tokens": 37}, 37, id="generic-limit-maps-to-native-field"),
        pytest.param(
            {"max_tokens": 37, "generation_config": {"temperature": 0.2}},
            37,
            id="generic-limit-preserves-other-generation-config",
        ),
        pytest.param(
            {"generation_config": {"max_output_tokens": 41}},
            41,
            id="native-limit-is-preserved",
        ),
        pytest.param(
            {"max_tokens": 37, "generation_config": {"max_output_tokens": 41}},
            41,
            id="native-limit-takes-precedence",
        ),
    ],
)
async def test_gemini_max_tokens_alias_precedence(
    monkeypatch, request, generation_kwargs, expected_max_output_tokens
):
    """LightRAG's generic max_tokens kwarg reaches Gemini's native config field."""
    gemini_module = _load_gemini_module(monkeypatch, request)
    fake_client, captured = _make_capturing_client(
        _make_fake_gemini_response(regular_text="ok")
    )
    monkeypatch.setattr(gemini_module, "_get_gemini_client", lambda *args: fake_client)

    await gemini_module.gemini_complete_if_cache(
        model="gemini-model", prompt="hi", api_key="test-key", **generation_kwargs
    )

    assert captured["config"].kwargs.get("max_output_tokens") == (
        expected_max_output_tokens
    )
    if "temperature" in generation_kwargs.get("generation_config", {}):
        assert captured["config"].kwargs.get("temperature") == generation_kwargs[
            "generation_config"
        ].get("temperature")


@pytest.mark.offline
@pytest.mark.asyncio
async def test_gemini_no_max_tokens_builds_no_config(monkeypatch, request):
    """Absent both the generic and native knobs, no config object is sent at all."""
    gemini_module = _load_gemini_module(monkeypatch, request)
    fake_client, captured = _make_capturing_client(
        _make_fake_gemini_response(regular_text="ok")
    )
    monkeypatch.setattr(gemini_module, "_get_gemini_client", lambda *args: fake_client)

    await gemini_module.gemini_complete_if_cache(
        model="gemini-model", prompt="hi", api_key="test-key"
    )

    assert "config" not in captured


@pytest.mark.offline
def test_gemini_complete_retry_predicate_matches_native_genai_errors(
    monkeypatch, request
):
    """google-genai raises its own errors.ServerError/ClientError -- never
    google.api_core.exceptions, which this SDK's request path never touches.
    A predicate built only from google.api_core.exceptions types never
    matches a real transient failure, so a 503 gets zero retries instead of
    the intended three."""
    gemini_module = _load_gemini_module(monkeypatch, request)

    def _would_retry(exc):
        return gemini_module.gemini_complete_if_cache.retry.retry(_FakeRetryState(exc))

    # A transient 5xx from the real SDK must be retried.
    assert _would_retry(_FakeGeminiServerError(503)) is True
    # 429 rate-limiting surfaces as a ClientError (4xx) in this SDK's
    # scheme rather than a distinct "resource exhausted" type, but it is
    # still transient and worth a fresh attempt.
    assert _would_retry(_FakeGeminiClientError(429)) is True
    # An ordinary 4xx (bad request, bad auth) must NOT be retried.
    assert _would_retry(_FakeGeminiClientError(400)) is False


@pytest.mark.offline
def test_gemini_embed_retry_predicate_matches_native_genai_server_error(
    monkeypatch, request
):
    """Same misclassification as the completion path, in gemini_embed's
    own retry decorator."""
    gemini_module = _load_gemini_module(monkeypatch, request)

    would_retry = gemini_module.gemini_embed.func.retry.retry(
        _FakeRetryState(_FakeGeminiServerError(503))
    )

    assert would_retry is True


@pytest.mark.offline
@pytest.mark.asyncio
async def test_gemini_complete_exhausted_retries_reraise_the_sdk_error(
    monkeypatch, request
):
    """A 503 that persists through every attempt must surface as the SDK's
    own ServerError, not as tenacity.RetryError, so the document's error_msg
    keeps Gemini's message and callers can still catch the SDK error."""
    from tenacity import wait_none

    gemini_module = _load_gemini_module(monkeypatch, request)
    monkeypatch.setattr(
        gemini_module.gemini_complete_if_cache.retry, "wait", wait_none()
    )

    calls = {"n": 0}

    async def _failing_generate_content(**kwargs):
        calls["n"] += 1
        raise _FakeGeminiServerError(503)

    fake_client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=_failing_generate_content)
        )
    )
    monkeypatch.setattr(gemini_module, "_get_gemini_client", lambda *args: fake_client)

    with pytest.raises(_FakeGeminiServerError, match="503"):
        await gemini_module.gemini_complete_if_cache(
            model="gemini-model", prompt="hi", api_key="test-key"
        )

    assert calls["n"] == 3


@pytest.mark.offline
@pytest.mark.asyncio
async def test_gemini_embed_exhausted_retries_reraise_the_sdk_error(
    monkeypatch, request
):
    """Same exhausted-retry path in gemini_embed: a persistent 429 must reach
    the caller as the SDK's ClientError, not as tenacity.RetryError."""
    from tenacity import wait_none

    gemini_module = _load_gemini_module(monkeypatch, request)
    monkeypatch.setattr(gemini_module.gemini_embed.func.retry, "wait", wait_none())
    monkeypatch.setattr(
        gemini_module.types,
        "EmbedContentConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )

    calls = {"n": 0}

    async def _failing_embed_content(**kwargs):
        calls["n"] += 1
        raise _FakeGeminiClientError(429)

    fake_client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(embed_content=_failing_embed_content)
        )
    )
    monkeypatch.setattr(gemini_module, "_get_gemini_client", lambda *args: fake_client)

    with pytest.raises(_FakeGeminiClientError, match="429"):
        await gemini_module.gemini_embed.func(["hello"], api_key="test-key")

    assert calls["n"] == 3
