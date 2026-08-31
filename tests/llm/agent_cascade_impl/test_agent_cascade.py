from __future__ import annotations

from collections.abc import AsyncIterator
import json
from pathlib import Path

import pytest

from lightrag.llm.agent_cascade import (
    AllProvidersFailed,
    CascadeConfig,
    ClaudeCodeProvider,
    CodexCLIProvider,
    CommandResult,
    LLMAgentCascade,
    LLMTask,
    McpHubProvider,
    ProviderCallError,
    ProviderOutput,
)


class ScriptedProvider:
    def __init__(self, name, *outcomes, model="test-model"):
        self.name = name
        self.model = model
        self.outcomes = list(outcomes)
        self.calls = 0

    async def invoke(self, _task):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return ProviderOutput(outcome, self.name, self.model)


def _config(*providers, threshold=3):
    return CascadeConfig(
        provider_order=providers,
        circuit_failure_threshold=threshold,
    )


@pytest.mark.asyncio
async def test_falls_back_and_validates_with_parser():
    claude = ScriptedProvider("claude", ProviderCallError("quota", "limit"))
    codex = ScriptedProvider("codex", "not json")
    mcphub = ScriptedProvider("mcphub", '{"value": 3}')
    cascade = LLMAgentCascade(
        _config("claude", "codex", "mcphub"),
        providers={"claude": claude, "codex": codex, "mcphub": mcphub},
    )

    result = await cascade.acall(
        LLMTask("extract", idempotency_key="row-1"), parser=json.loads
    )

    assert result.value == {"value": 3}
    assert result.provider == "mcphub"
    assert result.idempotency_key == "row-1"
    assert result.degraded is True
    assert [attempt.error_kind for attempt in result.attempts] == [
        "quota",
        "invalid_output",
        None,
    ]


@pytest.mark.asyncio
async def test_safety_refusal_stops_without_fallback():
    claude = ScriptedProvider("claude", "I cannot assist with that request.")
    codex = ScriptedProvider("codex", "must not run")
    cascade = LLMAgentCascade(
        _config("claude", "codex"),
        providers={"claude": claude, "codex": codex},
    )

    with pytest.raises(AllProvidersFailed) as caught:
        await cascade.acall(LLMTask("extract"))

    assert caught.value.attempts[0].error_kind == "safety"
    assert codex.calls == 0


@pytest.mark.asyncio
async def test_refusal_text_inside_structured_data_is_not_a_safety_signal():
    content = '{"source_text": "I cannot assist with that request."}'
    claude = ScriptedProvider("claude", content)
    cascade = LLMAgentCascade(_config("claude"), providers={"claude": claude})

    result = await cascade.acall(LLMTask("extract"), parser=json.loads)

    assert result.value["source_text"].startswith("I cannot")


@pytest.mark.asyncio
async def test_circuit_breaker_skips_repeated_auth_failure():
    claude = ScriptedProvider(
        "claude",
        ProviderCallError("authentication", "bad login"),
        ProviderCallError("authentication", "bad login"),
    )
    codex = ScriptedProvider("codex", "one", "two", "three")
    cascade = LLMAgentCascade(
        _config("claude", "codex", threshold=2),
        providers={"claude": claude, "codex": codex},
    )

    await cascade.acall(LLMTask("first"))
    await cascade.acall(LLMTask("second"))
    third = await cascade.acall(LLMTask("third"))

    assert claude.calls == 2
    assert third.attempts[0].error_kind == "circuit_open"
    assert third.provider == "codex"


@pytest.mark.asyncio
async def test_attempt_error_redacts_openai_style_secret():
    claude = ScriptedProvider(
        "claude",
        ProviderCallError("authentication", "credential sk-example-secret-value"),
    )
    cascade = LLMAgentCascade(_config("claude"), providers={"claude": claude})

    with pytest.raises(AllProvidersFailed) as caught:
        await cascade.acall(LLMTask("extract"))

    assert caught.value.attempts[0].error == "credential [REDACTED]"


def test_config_from_env_controls_models_and_limits():
    config = CascadeConfig.from_env(
        {
            "LIGHTRAG_AGENT_PROVIDER_ORDER": "codex,mcphub",
            "LIGHTRAG_AGENT_CODEX_MODEL": "codex-model",
            "LIGHTRAG_AGENT_MCPHUB_MODEL": "gateway-model",
            "MCPHUB_BASE_URL": "https://gateway.example/v1/",
            "LIGHTRAG_AGENT_MAX_TOKENS": "2048",
            "LIGHTRAG_AGENT_COMMAND_TIMEOUT_SECONDS": "45",
            "LIGHTRAG_AGENT_CIRCUIT_FAILURE_THRESHOLD": "4",
            "LIGHTRAG_AGENT_CIRCUIT_COOLDOWN_SECONDS": "90",
            "LIGHTRAG_AGENT_CODEX_HOME": "/tmp/codex-home",
        }
    )

    assert config.provider_order == ("codex", "mcphub")
    assert config.codex_model == "codex-model"
    assert config.mcphub_model == "gateway-model"
    assert config.mcphub_base_url == "https://gateway.example/v1/"
    assert config.max_tokens == 2048
    assert config.command_timeout_seconds == 45.0
    assert config.circuit_failure_threshold == 4
    assert config.circuit_cooldown_seconds == 90.0
    assert config.codex_home == "/tmp/codex-home"


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"idempotency_key": ""}, "idempotency_key"),
        ({"requires_tools": 1}, "requires_tools"),
        ({"history_messages": ("bad",)}, "history_messages"),
    ],
)
def test_task_rejects_invalid_control_metadata(kwargs, error):
    with pytest.raises((TypeError, ValueError), match=error):
        LLMTask("extract", **kwargs)


@pytest.mark.asyncio
async def test_claude_cli_is_tool_free_and_uses_isolated_oauth_environment():
    calls = []

    async def runner(command, **kwargs):
        calls.append((command, kwargs))
        return CommandResult(0, '{"ok": true}\n', "")

    provider = ClaudeCodeProvider(
        model="sonnet",
        timeout_seconds=30,
        command_runner=runner,
        environment={
            "CLAUDE_CODE_OAUTH_TOKEN_1": "test-token",
            "ANTHROPIC_API_KEY": "must-not-shadow-oauth",
            "PATH": "/usr/bin",
        },
    )

    result = await provider.invoke(LLMTask("extract"))

    command, kwargs = calls[0]
    tools_index = command.index("--tools")
    assert command[tools_index + 1] == ""
    assert "--no-session-persistence" in command
    assert kwargs["environment"]["CLAUDE_CODE_OAUTH_TOKEN"] == "test-token"
    assert "CLAUDE_CODE_OAUTH_TOKEN_1" not in kwargs["environment"]
    assert "ANTHROPIC_API_KEY" not in kwargs["environment"]
    assert result.content == '{"ok": true}'
    assert result.metadata["token_slot"] == "CLAUDE_CODE_OAUTH_TOKEN_1"


@pytest.mark.asyncio
async def test_claude_cli_rotates_named_oauth_token_pool():
    selected = []

    async def runner(_command, **kwargs):
        selected.append(kwargs["environment"]["CLAUDE_CODE_OAUTH_TOKEN"])
        return CommandResult(0, "ok", "")

    provider = ClaudeCodeProvider(
        model="sonnet",
        timeout_seconds=30,
        command_runner=runner,
        environment={
            "CLAUDE_CODE_OAUTH_TOKEN": "token-0",
            "CLAUDE_CODE_OAUTH_TOKEN_1": "token-1",
        },
    )

    await provider.invoke(LLMTask("one"))
    await provider.invoke(LLMTask("two"))

    assert selected == ["token-0", "token-1"]


@pytest.mark.asyncio
async def test_claude_local_login_does_not_inherit_anthropic_api_key():
    environments = []

    async def runner(_command, **kwargs):
        environments.append(kwargs["environment"])
        return CommandResult(0, "ok", "")

    provider = ClaudeCodeProvider(
        model="sonnet",
        timeout_seconds=30,
        command_runner=runner,
        environment={"ANTHROPIC_API_KEY": "must-not-use", "PATH": "/usr/bin"},
    )

    result = await provider.invoke(LLMTask("one"))

    assert "ANTHROPIC_API_KEY" not in environments[0]
    assert result.metadata["auth"] == "local_login"


@pytest.mark.asyncio
async def test_codex_cli_uses_ephemeral_read_only_execution_and_codex_home(
    tmp_path: Path,
):
    calls = []

    async def runner(command, **kwargs):
        calls.append((command, kwargs))
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text("codex answer\n", encoding="utf-8")
        return CommandResult(0, "events are ignored", "")

    provider = CodexCLIProvider(
        model="codex-model",
        timeout_seconds=30,
        codex_home=str(tmp_path / "codex-home"),
        command_runner=runner,
        environment={"PATH": "/usr/bin"},
    )

    result = await provider.invoke(LLMTask("transform"))

    command, kwargs = calls[0]
    assert command[:2] == ["codex", "exec"]
    assert command[-1] == "-"
    assert "--ephemeral" in command
    assert "--ignore-user-config" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert kwargs["environment"]["CODEX_HOME"] == str(tmp_path / "codex-home")
    assert result.content == "codex answer"
    assert result.metadata["ephemeral"] is True


@pytest.mark.asyncio
async def test_cli_error_is_classified_for_fallback():
    async def runner(_command, **_kwargs):
        return CommandResult(1, "", "429 too many requests")

    provider = CodexCLIProvider(
        model="codex-model",
        timeout_seconds=30,
        command_runner=runner,
    )

    with pytest.raises(ProviderCallError) as caught:
        await provider.invoke(LLMTask("transform"))

    assert caught.value.kind == "rate_limit"
    assert "too many requests" not in str(caught.value)


@pytest.mark.asyncio
async def test_mcphub_uses_lightrag_openai_compatible_call_shape():
    calls = []

    async def completion_func(**kwargs):
        calls.append(kwargs)
        return "gateway answer"

    provider = McpHubProvider(
        model="gateway-model",
        base_url="https://gateway.example/v1/",
        max_tokens=123,
        timeout_seconds=30,
        completion_func=completion_func,
        environment={"MCPHUB_API_KEY": "test-key"},
    )

    result = await provider.invoke(
        LLMTask(
            "extract",
            system_prompt="schema only",
            history_messages=({"role": "assistant", "content": "prior"},),
        )
    )

    assert result.content == "gateway answer"
    assert calls[0]["base_url"] == "https://gateway.example/v1"
    assert calls[0]["api_key"] == "test-key"
    assert calls[0]["max_completion_tokens"] == 123
    assert calls[0]["history_messages"][0]["content"] == "prior"


@pytest.mark.asyncio
async def test_light_rag_callable_returns_text_or_one_chunk_stream():
    provider = ScriptedProvider("claude", "plain", "streamed")
    cascade = LLMAgentCascade(_config("claude"), providers={"claude": provider})

    text = await cascade("question", system_prompt="system")
    stream = await cascade("question", stream=True)

    assert text == "plain"
    assert isinstance(stream, AsyncIterator)
    assert [chunk async for chunk in stream] == ["streamed"]


@pytest.mark.asyncio
async def test_light_rag_callable_rejects_images_before_provider_call():
    provider = ScriptedProvider("claude", "must not run")
    cascade = LLMAgentCascade(_config("claude"), providers={"claude": provider})

    with pytest.raises(ValueError, match="text-only"):
        await cascade("question", image_inputs=[{"data": "..."}])

    assert provider.calls == 0


@pytest.mark.asyncio
async def test_light_rag_callable_does_not_coerce_requires_tools_strings():
    provider = ScriptedProvider("claude", "must not run")
    cascade = LLMAgentCascade(_config("claude"), providers={"claude": provider})

    with pytest.raises(TypeError, match="requires_tools"):
        await cascade("question", requires_tools="false")

    assert provider.calls == 0


def test_sync_call_runs_without_an_event_loop():
    provider = ScriptedProvider("claude", '{"ok": true}')
    cascade = LLMAgentCascade(_config("claude"), providers={"claude": provider})

    result = cascade.call(LLMTask("extract"), parser=json.loads)

    assert result.value == {"ok": True}


@pytest.mark.asyncio
async def test_sync_call_rejects_an_existing_event_loop():
    provider = ScriptedProvider("claude", "must not run")
    cascade = LLMAgentCascade(_config("claude"), providers={"claude": provider})

    with pytest.raises(RuntimeError, match="use acall"):
        cascade.call(LLMTask("extract"))

    assert provider.calls == 0
