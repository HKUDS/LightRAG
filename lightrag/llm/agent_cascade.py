"""Fail over bounded text tasks across local agent logins and an API gateway.

This module is intentionally separate from the regular API-key bindings.  It is
for trusted local batch jobs that already use Claude Code or Codex CLI login
state and want a final OpenAI-compatible gateway fallback.  Every provider is
lazy: importing :mod:`lightrag` does not require either CLI or the OpenAI SDK.

The cascade is also a LightRAG ``llm_model_func``.  It returns a string for a
normal call and a one-chunk async iterator when ``stream=True``.  Call
``acall()`` directly when structured attempt metadata or parser validation is
needed by an extraction pipeline.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time
from typing import Any, Generic, Literal, Protocol, TypeVar

from lightrag.utils import logger


ProviderName = Literal["claude", "codex", "mcphub"]
ErrorKind = Literal[
    "unavailable",
    "authentication",
    "quota",
    "rate_limit",
    "timeout",
    "transport",
    "server",
    "invalid_output",
    "capability",
    "safety",
    "request",
    "internal",
    "circuit_open",
]

DEFAULT_PROVIDER_ORDER: tuple[ProviderName, ...] = ("claude", "codex", "mcphub")
DEFAULT_FALLBACK_KINDS: frozenset[ErrorKind] = frozenset(
    {
        "unavailable",
        "authentication",
        "quota",
        "rate_limit",
        "timeout",
        "transport",
        "server",
        "invalid_output",
        "capability",
        "circuit_open",
    }
)
_CIRCUIT_FAILURE_KINDS: frozenset[ErrorKind] = frozenset(
    {
        "unavailable",
        "authentication",
        "quota",
        "rate_limit",
        "timeout",
        "transport",
        "server",
    }
)

_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)\b(?:mcp|sk-ant|sk-proj|sk)-[A-Za-z0-9._-]+"),
    re.compile(
        r"(?i)((?:access|refresh|oauth|api)[_-]?token|api[_-]?key)"
        r"\s*[:=]\s*['\"]?[^\s,'\"]+"
    ),
)
_SAFETY_REFUSAL_RE = re.compile(
    r"(?i)^\s*(?:"
    r"i (?:cannot|can't) (?:help|assist) with|"
    r"i(?:'m| am) unable to (?:help|assist)|"
    r"i must refuse|"
    r"cannot comply with (?:that|this|the) request|"
    r"against my safety policy|"
    r"not able to provide (?:that |this )?assistance"
    r")"
)

ParsedT = TypeVar("ParsedT")


def _redact_error(value: object, *, limit: int = 800) -> str:
    text = str(value).strip() or type(value).__name__
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            text = pattern.sub(lambda match: f"{match.group(1)}[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    return text[:limit]


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(name: str, value: object) -> float:
    if type(value) not in {int, float} or value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


@dataclass(frozen=True)
class CascadeConfig:
    """Provider order, models, and failure policy for one reusable cascade."""

    provider_order: tuple[ProviderName, ...] = DEFAULT_PROVIDER_ORDER
    claude_model: str = "claude-sonnet-4-6"
    codex_model: str = "gpt-5.6-luna"
    mcphub_model: str = "gpt-5.6-luna"
    mcphub_base_url: str = "https://api.mcphub.com/v1"
    max_tokens: int = 8192
    command_timeout_seconds: float = 900.0
    circuit_failure_threshold: int = 3
    circuit_cooldown_seconds: float = 300.0
    codex_home: str | None = None
    fallback_kinds: frozenset[ErrorKind] = DEFAULT_FALLBACK_KINDS

    def __post_init__(self) -> None:
        if not self.provider_order:
            raise ValueError("provider_order must not be empty")
        if len(set(self.provider_order)) != len(self.provider_order):
            raise ValueError("provider_order must not contain duplicates")
        unknown = set(self.provider_order) - set(DEFAULT_PROVIDER_ORDER)
        if unknown:
            raise ValueError(f"unknown providers: {', '.join(sorted(unknown))}")
        for name in ("claude_model", "codex_model", "mcphub_model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonblank string")
        if (
            not isinstance(self.mcphub_base_url, str)
            or not self.mcphub_base_url.strip()
        ):
            raise ValueError("mcphub_base_url must be a nonblank string")
        _positive_int("max_tokens", self.max_tokens)
        _positive_float("command_timeout_seconds", self.command_timeout_seconds)
        _positive_int("circuit_failure_threshold", self.circuit_failure_threshold)
        _positive_float("circuit_cooldown_seconds", self.circuit_cooldown_seconds)

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> "CascadeConfig":
        if environment is None:
            try:
                from dotenv import load_dotenv

                load_dotenv(dotenv_path=".env", override=False)
            except ImportError:
                pass
        env = environment if environment is not None else os.environ
        order = tuple(
            name.strip().casefold()
            for name in env.get(
                "LIGHTRAG_AGENT_PROVIDER_ORDER", "claude,codex,mcphub"
            ).split(",")
            if name.strip()
        )
        return cls(
            provider_order=order,  # type: ignore[arg-type]
            claude_model=env.get("LIGHTRAG_AGENT_CLAUDE_MODEL", "claude-sonnet-4-6"),
            codex_model=env.get("LIGHTRAG_AGENT_CODEX_MODEL", "gpt-5.6-luna"),
            mcphub_model=env.get("LIGHTRAG_AGENT_MCPHUB_MODEL", "gpt-5.6-luna"),
            mcphub_base_url=env.get("MCPHUB_BASE_URL", "https://api.mcphub.com/v1"),
            max_tokens=int(env.get("LIGHTRAG_AGENT_MAX_TOKENS", "8192")),
            command_timeout_seconds=float(
                env.get("LIGHTRAG_AGENT_COMMAND_TIMEOUT_SECONDS", "900")
            ),
            circuit_failure_threshold=int(
                env.get("LIGHTRAG_AGENT_CIRCUIT_FAILURE_THRESHOLD", "3")
            ),
            circuit_cooldown_seconds=float(
                env.get("LIGHTRAG_AGENT_CIRCUIT_COOLDOWN_SECONDS", "300")
            ),
            codex_home=env.get("LIGHTRAG_AGENT_CODEX_HOME") or None,
        )


@dataclass(frozen=True)
class LLMTask:
    prompt: str
    system_prompt: str = "You extract and transform data accurately."
    history_messages: tuple[Mapping[str, Any], ...] = ()
    task_type: str = "llm_data_task"
    idempotency_key: str | None = None
    requires_tools: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a nonblank string")
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise ValueError("system_prompt must be a nonblank string")
        if not isinstance(self.task_type, str) or not self.task_type.strip():
            raise ValueError("task_type must be a nonblank string")
        if self.idempotency_key is not None and (
            not isinstance(self.idempotency_key, str)
            or not self.idempotency_key.strip()
        ):
            raise ValueError("idempotency_key must be a nonblank string when set")
        if type(self.requires_tools) is not bool:
            raise TypeError("requires_tools must be a boolean")
        for message in self.history_messages:
            if not isinstance(message, Mapping):
                raise TypeError("history_messages entries must be mappings")

    def rendered_prompt(self, *, include_system: bool = True) -> str:
        sections = []
        if include_system:
            sections.append(f"System instructions:\n{self.system_prompt.strip()}")
        if self.history_messages:
            history = json.dumps(
                list(self.history_messages), ensure_ascii=False, default=str
            )
            sections.append(f"Conversation history (JSON):\n{history}")
        sections.append(f"Task:\n{self.prompt.strip()}")
        return "\n\n".join(sections)


@dataclass(frozen=True)
class ProviderOutput:
    content: str
    provider: ProviderName
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderAttempt:
    provider: ProviderName
    model: str
    status: Literal["succeeded", "failed", "skipped"]
    elapsed_ms: int
    error_kind: ErrorKind | None = None
    error: str | None = None


@dataclass(frozen=True)
class LLMResult(Generic[ParsedT]):
    value: ParsedT
    content: str
    provider: ProviderName
    model: str
    attempts: tuple[ProviderAttempt, ...]
    task_type: str
    idempotency_key: str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    degraded: bool = False


class ProviderCallError(RuntimeError):
    def __init__(self, kind: ErrorKind, message: str) -> None:
        self.kind = kind
        super().__init__(_redact_error(message))


class AllProvidersFailed(RuntimeError):
    def __init__(self, attempts: Sequence[ProviderAttempt]) -> None:
        self.attempts = tuple(attempts)
        summary = "; ".join(
            f"{attempt.provider}:{attempt.error_kind or attempt.status}"
            for attempt in self.attempts
        )
        super().__init__(f"all configured LLM providers failed ({summary})")


class LLMProvider(Protocol):
    name: ProviderName
    model: str

    async def invoke(self, task: LLMTask) -> ProviderOutput: ...


def _classify_error(error: object) -> ErrorKind:
    text = str(error).casefold()
    if any(token in text for token in ("safety", "content policy", "refusal")):
        return "safety"
    if any(
        token in text
        for token in (
            "unauthorized",
            "authentication",
            "not logged in",
            "invalid api key",
            "invalid_grant",
            "401",
            "403",
        )
    ):
        return "authentication"
    if any(
        token in text
        for token in (
            "quota",
            "usage limit",
            "credit exhausted",
            "payment required",
            "402",
        )
    ):
        return "quota"
    if any(token in text for token in ("rate limit", "too many requests", "429")):
        return "rate_limit"
    if any(token in text for token in ("timed out", "timeout")):
        return "timeout"
    if any(token in text for token in ("connection", "network", "transport", "dns")):
        return "transport"
    if re.search(r"\b5\d\d\b", text):
        return "server"
    if any(
        token in text
        for token in ("not configured", "not set", "not installed", "not found")
    ):
        return "unavailable"
    if any(token in text for token in ("bad request", "invalid request", "400")):
        return "request"
    return "internal"


def _looks_like_safety_refusal(content: str) -> bool:
    return _SAFETY_REFUSAL_RE.search(content[:2000]) is not None


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[..., Awaitable[CommandResult]]


async def _run_command(
    command: Sequence[str],
    *,
    input_text: str,
    timeout_seconds: float,
    cwd: str,
    environment: Mapping[str, str],
) -> CommandResult:
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=dict(environment),
        )
    except FileNotFoundError as exc:
        raise ProviderCallError(
            "unavailable", f"agent executable not found: {command[0]}"
        ) from exc

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(input_text.encode("utf-8")),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        if process.returncode is None:
            process.kill()
        await process.communicate()
        raise ProviderCallError(
            "timeout", f"agent command timed out after {timeout_seconds:g}s"
        ) from exc
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.communicate()
        raise

    return CommandResult(
        returncode=process.returncode or 0,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
    )


class ClaudeCodeProvider:
    """Run one tool-free Claude Code turn using local login or OAuth env state."""

    name: ProviderName = "claude"

    def __init__(
        self,
        *,
        model: str,
        timeout_seconds: float,
        executable: str = "claude",
        command_runner: CommandRunner | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.executable = executable
        self._command_runner = command_runner
        self._environment = environment
        self._token_lock = threading.Lock()
        self._next_token = 0

    def _child_environment(self) -> tuple[dict[str, str], str | None]:
        env = dict(self._environment if self._environment is not None else os.environ)
        # This adapter is for Claude Code login/OAuth use. Never let a caller's
        # unrelated API key silently change the billing/authentication path.
        env.pop("ANTHROPIC_API_KEY", None)
        token_names = ("CLAUDE_CODE_OAUTH_TOKEN",) + tuple(
            f"CLAUDE_CODE_OAUTH_TOKEN_{index}" for index in range(1, 6)
        )
        available = [(name, env.get(name, "")) for name in token_names if env.get(name)]
        selected_name: str | None = None
        if available:
            with self._token_lock:
                selected_name, token = available[self._next_token % len(available)]
                self._next_token += 1
            env["CLAUDE_CODE_OAUTH_TOKEN"] = token
            for name in token_names[1:]:
                env.pop(name, None)
        return env, selected_name

    async def invoke(self, task: LLMTask) -> ProviderOutput:
        if task.requires_tools:
            raise ProviderCallError(
                "capability", "the Claude cascade adapter intentionally disables tools"
            )
        runner = self._command_runner or _run_command
        if self._command_runner is None and shutil.which(self.executable) is None:
            raise ProviderCallError("unavailable", "Claude Code CLI is not installed")
        environment, selected_token = self._child_environment()
        command = [
            self.executable,
            "--print",
            "--output-format",
            "text",
            "--model",
            self.model,
            "--permission-mode",
            "dontAsk",
            "--tools",
            "",
            "--no-session-persistence",
            "--disable-slash-commands",
            "--system-prompt",
            task.system_prompt,
        ]
        with tempfile.TemporaryDirectory(prefix="lightrag-claude-") as workdir:
            try:
                result = await runner(
                    command,
                    input_text=task.rendered_prompt(include_system=False),
                    timeout_seconds=self.timeout_seconds,
                    cwd=workdir,
                    environment=environment,
                )
            except ProviderCallError:
                raise
            except Exception as exc:
                raise ProviderCallError(
                    _classify_error(exc),
                    f"Claude command failed: {type(exc).__name__}",
                ) from exc
        if result.returncode:
            detail = result.stderr or result.stdout or f"exit code {result.returncode}"
            raise ProviderCallError(
                _classify_error(detail),
                f"Claude command failed (exit code {result.returncode})",
            )
        content = result.stdout.strip()
        if not content:
            raise ProviderCallError("invalid_output", "Claude returned an empty output")
        metadata = {"auth": "local_login"}
        if selected_token:
            metadata["auth"] = "oauth_environment"
            metadata["token_slot"] = selected_token
        return ProviderOutput(content, self.name, self.model, metadata)


class CodexCLIProvider:
    """Run one ephemeral Codex CLI turn with its normal cached authentication."""

    name: ProviderName = "codex"

    def __init__(
        self,
        *,
        model: str,
        timeout_seconds: float,
        codex_home: str | None = None,
        executable: str = "codex",
        command_runner: CommandRunner | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.codex_home = codex_home
        self.executable = executable
        self._command_runner = command_runner
        self._environment = environment

    async def invoke(self, task: LLMTask) -> ProviderOutput:
        if task.requires_tools:
            raise ProviderCallError(
                "capability", "the Codex cascade adapter does not enable task tools"
            )
        runner = self._command_runner or _run_command
        if self._command_runner is None and shutil.which(self.executable) is None:
            raise ProviderCallError("unavailable", "Codex CLI is not installed")
        environment = dict(
            self._environment if self._environment is not None else os.environ
        )
        if self.codex_home:
            environment["CODEX_HOME"] = str(Path(self.codex_home).expanduser())

        with tempfile.TemporaryDirectory(prefix="lightrag-codex-") as workdir:
            output_path = Path(workdir) / "last-message.txt"
            command = [
                self.executable,
                "exec",
                "--model",
                self.model,
                "--sandbox",
                "read-only",
                "--cd",
                workdir,
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--color",
                "never",
                "--output-last-message",
                str(output_path),
                "-",
            ]
            try:
                result = await runner(
                    command,
                    input_text=task.rendered_prompt(),
                    timeout_seconds=self.timeout_seconds,
                    cwd=workdir,
                    environment=environment,
                )
            except ProviderCallError:
                raise
            except Exception as exc:
                raise ProviderCallError(
                    _classify_error(exc),
                    f"Codex command failed: {type(exc).__name__}",
                ) from exc
            if result.returncode:
                detail = (
                    result.stderr or result.stdout or f"exit code {result.returncode}"
                )
                raise ProviderCallError(
                    _classify_error(detail),
                    f"Codex command failed (exit code {result.returncode})",
                )
            content = (
                output_path.read_text(encoding="utf-8").strip()
                if output_path.exists()
                else ""
            )
        if not content:
            raise ProviderCallError("invalid_output", "Codex returned an empty output")
        return ProviderOutput(
            content,
            self.name,
            self.model,
            {"auth": "codex_cli", "ephemeral": True, "sandbox": "read-only"},
        )


class McpHubProvider:
    """Use LightRAG's OpenAI-compatible binding against an MCPHub endpoint."""

    name: ProviderName = "mcphub"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        max_tokens: int,
        timeout_seconds: float,
        api_key: str | None = None,
        completion_func: Callable[..., Awaitable[Any]] | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key
        self._completion_func = completion_func
        self._environment = environment

    async def invoke(self, task: LLMTask) -> ProviderOutput:
        if task.requires_tools:
            raise ProviderCallError(
                "capability", "MCPHub chat completion has no agent tool loop"
            )
        env = self._environment if self._environment is not None else os.environ
        api_key = self.api_key or env.get("MCPHUB_API_KEY") or env.get("MCP_KEY")
        if not api_key:
            raise ProviderCallError("unavailable", "MCPHUB_API_KEY is not configured")
        completion_func = self._completion_func
        if completion_func is None:
            try:
                from lightrag.llm.openai import openai_complete_if_cache
            except ImportError as exc:
                raise ProviderCallError(
                    "unavailable", "OpenAI provider dependencies are not installed"
                ) from exc
            completion_func = openai_complete_if_cache
        try:
            content = await completion_func(
                model=self.model,
                prompt=task.prompt,
                system_prompt=task.system_prompt,
                history_messages=list(task.history_messages),
                base_url=self.base_url,
                api_key=api_key,
                stream=False,
                timeout=int(self.timeout_seconds),
                max_completion_tokens=self.max_tokens,
            )
        except ProviderCallError:
            raise
        except Exception as exc:
            raise ProviderCallError(
                _classify_error(exc),
                f"MCPHub call failed: {type(exc).__name__}",
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderCallError("invalid_output", "MCPHub returned an empty output")
        return ProviderOutput(content.strip(), self.name, self.model)


@dataclass
class _CircuitState:
    failures: int = 0
    open_until: float = 0.0


class LLMAgentCascade:
    """Try configured providers in order and return the first accepted output."""

    def __init__(
        self,
        config: CascadeConfig | None = None,
        *,
        providers: Mapping[ProviderName, LLMProvider] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or CascadeConfig.from_env()
        self._clock = clock
        self._lock = threading.Lock()
        self._circuits = {name: _CircuitState() for name in self.config.provider_order}
        self._providers = (
            dict(providers)
            if providers is not None
            else {
                "claude": ClaudeCodeProvider(
                    model=self.config.claude_model,
                    timeout_seconds=self.config.command_timeout_seconds,
                ),
                "codex": CodexCLIProvider(
                    model=self.config.codex_model,
                    timeout_seconds=self.config.command_timeout_seconds,
                    codex_home=self.config.codex_home,
                ),
                "mcphub": McpHubProvider(
                    model=self.config.mcphub_model,
                    base_url=self.config.mcphub_base_url,
                    max_tokens=self.config.max_tokens,
                    timeout_seconds=self.config.command_timeout_seconds,
                ),
            }
        )

    def _circuit_is_open(self, provider: ProviderName) -> bool:
        with self._lock:
            state = self._circuits[provider]
            if state.open_until <= self._clock():
                if state.open_until:
                    state.failures = 0
                    state.open_until = 0.0
                return False
            return True

    def _record_success(self, provider: ProviderName) -> None:
        with self._lock:
            self._circuits[provider] = _CircuitState()

    def _record_failure(self, provider: ProviderName, kind: ErrorKind) -> None:
        if kind not in _CIRCUIT_FAILURE_KINDS:
            return
        with self._lock:
            state = self._circuits[provider]
            state.failures += 1
            if state.failures >= self.config.circuit_failure_threshold:
                state.open_until = self._clock() + self.config.circuit_cooldown_seconds

    async def acall(
        self,
        task: LLMTask,
        *,
        parser: Callable[[str], ParsedT] | None = None,
    ) -> LLMResult[ParsedT | str]:
        if not isinstance(task, LLMTask):
            raise TypeError("task must be an LLMTask")
        attempts: list[ProviderAttempt] = []
        for index, provider_name in enumerate(self.config.provider_order):
            provider = self._providers.get(provider_name)
            if provider is None:
                raise ValueError(f"provider implementation missing: {provider_name}")
            if self._circuit_is_open(provider_name):
                attempts.append(
                    ProviderAttempt(
                        provider=provider_name,
                        model=provider.model,
                        status="skipped",
                        elapsed_ms=0,
                        error_kind="circuit_open",
                        error="provider circuit is open",
                    )
                )
                continue
            started = self._clock()
            try:
                output = await provider.invoke(task)
                if not isinstance(output.content, str) or not output.content.strip():
                    raise ProviderCallError(
                        "invalid_output", f"{provider_name} returned an empty output"
                    )
                if _looks_like_safety_refusal(output.content):
                    raise ProviderCallError(
                        "safety", f"{provider_name} returned a safety refusal"
                    )
                try:
                    value = parser(output.content) if parser else output.content
                except Exception as exc:
                    raise ProviderCallError(
                        "invalid_output",
                        f"output validation failed: {type(exc).__name__}",
                    ) from exc
            except ProviderCallError as exc:
                elapsed_ms = max(0, int((self._clock() - started) * 1000))
                attempts.append(
                    ProviderAttempt(
                        provider=provider_name,
                        model=provider.model,
                        status="failed",
                        elapsed_ms=elapsed_ms,
                        error_kind=exc.kind,
                        error=_redact_error(exc),
                    )
                )
                self._record_failure(provider_name, exc.kind)
                if exc.kind not in self.config.fallback_kinds:
                    raise AllProvidersFailed(attempts) from exc
                continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                attempts.append(
                    ProviderAttempt(
                        provider=provider_name,
                        model=provider.model,
                        status="failed",
                        elapsed_ms=max(0, int((self._clock() - started) * 1000)),
                        error_kind="internal",
                        error=_redact_error(exc),
                    )
                )
                raise AllProvidersFailed(attempts) from exc
            self._record_success(provider_name)
            attempts.append(
                ProviderAttempt(
                    provider=provider_name,
                    model=provider.model,
                    status="succeeded",
                    elapsed_ms=max(0, int((self._clock() - started) * 1000)),
                )
            )
            return LLMResult(
                value=value,
                content=output.content,
                provider=provider_name,
                model=output.model,
                attempts=tuple(attempts),
                task_type=task.task_type,
                idempotency_key=task.idempotency_key,
                provider_metadata=dict(output.metadata),
                degraded=index > 0,
            )
        raise AllProvidersFailed(attempts)

    def call(
        self,
        task: LLMTask,
        *,
        parser: Callable[[str], ParsedT] | None = None,
    ) -> LLMResult[ParsedT | str]:
        """Synchronous convenience wrapper for scripts without an event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.acall(task, parser=parser))
        raise RuntimeError(
            "LLMAgentCascade.call() cannot run inside an event loop; use acall()"
        )

    async def __call__(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        stream: bool | None = None,
        **kwargs: Any,
    ) -> str | AsyncIterator[str]:
        """LightRAG ``llm_model_func`` compatibility entrypoint.

        Agent CLIs are final-response transports, so streaming is represented
        as one final chunk.  Image inputs and arbitrary tools are deliberately
        unsupported by this text-only batch adapter.
        """
        if kwargs.get("image_inputs"):
            raise ValueError("LLMAgentCascade is text-only and rejects image_inputs")
        task = LLMTask(
            prompt=prompt,
            system_prompt=system_prompt
            or "You are a helpful assistant. Follow the requested output format exactly.",
            history_messages=tuple(history_messages or ()),
            task_type=kwargs.get("task_type", "lightrag_llm_call"),
            idempotency_key=kwargs.get("idempotency_key"),
            requires_tools=kwargs.get("requires_tools", False),
        )
        result = await self.acall(task)
        if result.degraded:
            logger.warning(
                "LLM agent cascade degraded to %s after %d failed/skipped provider(s)",
                result.provider,
                len(result.attempts) - 1,
            )
        if stream:
            return _single_chunk(result.content)
        return result.content


async def _single_chunk(content: str) -> AsyncIterator[str]:
    yield content


__all__ = [
    "AllProvidersFailed",
    "CascadeConfig",
    "ClaudeCodeProvider",
    "CodexCLIProvider",
    "CommandResult",
    "LLMAgentCascade",
    "LLMResult",
    "LLMTask",
    "McpHubProvider",
    "ProviderAttempt",
    "ProviderCallError",
    "ProviderOutput",
]
