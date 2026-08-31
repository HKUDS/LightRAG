# Local Agent LLM Cascade

`lightrag.llm.agent_cascade` provides a bounded text-call interface for trusted
local workloads that want this provider order:

1. Claude Code through its local login or `CLAUDE_CODE_OAUTH_TOKEN`.
2. Codex CLI through its cached ChatGPT/API-key login.
3. An MCPHub OpenAI-compatible chat completion.

The cascade can be passed directly as a LightRAG `llm_model_func`, or called
with an output parser for large extraction and transformation jobs. A fallback
success is marked `degraded=True`, and every prior attempt is retained as
structured, credential-redacted metadata.

This adapter is for trusted local batch workers. It is not an API-server
authentication mechanism and must not be exposed to untrusted prompts or
public users.

## Prerequisites and authentication

Install the two agent CLIs and authenticate them before starting the worker:

```bash
claude --version
codex --version
codex login status
```

Codex officially supports ChatGPT login and API-key login for local workflows,
caches the login in `~/.codex/auth.json` or the operating-system credential
store, and refreshes active ChatGPT sessions during use. For unattended
enterprise automation, use a Codex access token or workload identity when your
workspace supports it. General OpenAI API services should continue to use a
Platform API key. See the [official Codex authentication
documentation](https://developers.openai.com/codex/auth/).

Claude can use its existing local login. A worker can instead provide one or
more OAuth environment variables:

```bash
export CLAUDE_CODE_OAUTH_TOKEN=...
export CLAUDE_CODE_OAUTH_TOKEN_1=...  # optional round-robin pool
export CLAUDE_CODE_OAUTH_TOKEN_2=...
```

When an OAuth token is selected, the Claude child process receives only that
token under the canonical name and does not inherit `ANTHROPIC_API_KEY`. Token
values are never placed in result metadata or attempt errors.

The final provider uses LightRAG's existing OpenAI-compatible binding:

```bash
uv sync --extra offline-llm
export MCPHUB_API_KEY=...
export MCPHUB_BASE_URL=https://api.mcphub.com/v1
```

`MCP_KEY` is accepted as a legacy key alias. Never commit any credential.

## Use with LightRAG

```python
from lightrag import LightRAG
from lightrag.llm.agent_cascade import LLMAgentCascade

cascade = LLMAgentCascade()

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=cascade,
    embedding_func=your_embedding_func,
)

await rag.initialize_storages()
try:
    await rag.ainsert("Text to index")
finally:
    await rag.finalize_storages()
```

The provider CLIs return final responses rather than token deltas. When
LightRAG calls the function with `stream=True`, the adapter returns a valid
async iterator containing one final chunk.

The cascade is text-only. It rejects `image_inputs`; use a vision-capable
LightRAG provider for the VLM role.

## Use for validated extraction

The parser is the acceptance gate. A provider that returns empty or
schema-invalid output is recorded as `invalid_output`, then the next provider
is attempted.

```python
import json

from lightrag.llm.agent_cascade import LLMAgentCascade, LLMTask

cascade = LLMAgentCascade()
result = await cascade.acall(
    LLMTask(
        prompt="Return one JSON object for this source record: ...",
        system_prompt="Extract observed fields only. Do not infer missing values.",
        task_type="catalog_record_extraction",
        idempotency_key="source-123",
    ),
    parser=json.loads,
)

print(result.value)
print(result.provider, result.degraded)
for attempt in result.attempts:
    print(attempt.provider, attempt.status, attempt.error_kind)
```

Synchronous scripts can use `cascade.call(task, parser=...)`. Inside notebooks,
services, or any running event loop, use `await cascade.acall(...)` instead.

For production structured output, pass a strict Pydantic
`model_validate_json` function or an equivalent schema parser.

## Fallback behavior

The default policy falls through on:

- a missing CLI, credential, or optional OpenAI dependency;
- authentication, quota, rate-limit, timeout, transport, and server failures;
- empty or parser-invalid output;
- a provider without a required capability.

It stops on request/programming errors and safety refusals. This prevents a bad
request from being hidden by another billable call and prevents fallback from
becoming a safety-policy bypass.

After three availability failures, the default circuit breaker skips that
provider for five minutes. Reuse one `LLMAgentCascade` across a batch so the
breaker state remains useful.

## Tool and process boundary

- Claude runs with an empty tool list, no session persistence, and slash
  commands disabled.
- Codex runs ephemerally in a new empty temporary directory with a read-only
  sandbox, user configuration and repository rules disabled. Codex remains an
  agent runtime; use this adapter only on a trusted machine and trusted input.
- MCPHub is one chat completion and has no agent tool loop.
- `requires_tools=True` is rejected by every default provider. Inject a custom
  provider instead of silently granting broader capabilities.

Each CLI gets a per-call child environment. Concurrent calls therefore do not
mutate process-global OAuth variables. Each call still starts a subprocess, so
batch controllers should enforce provider-specific concurrency and account
limits.

## Batch execution

For large jobs:

- reuse one cascade instance;
- checkpoint every record using `idempotency_key`;
- retry a record rather than an entire batch;
- persist `attempts` beside the accepted parsed output;
- use an external `asyncio.Semaphore` for provider/account concurrency;
- keep prompts idempotent because fallback replays the logical task;
- do not log raw prompts if records contain sensitive data.

The cascade does not persist trajectories or prompts itself. Claude is invoked
with `--no-session-persistence`, and Codex with `--ephemeral`.

## Environment variables

| Variable | Default | Meaning |
|---|---:|---|
| `LIGHTRAG_AGENT_PROVIDER_ORDER` | `claude,codex,mcphub` | Ordered, comma-separated providers. |
| `LIGHTRAG_AGENT_CLAUDE_MODEL` | `claude-sonnet-4-6` | Model passed to Claude Code. |
| `LIGHTRAG_AGENT_CODEX_MODEL` | `gpt-5.6-luna` | Model passed to `codex exec`. |
| `LIGHTRAG_AGENT_MCPHUB_MODEL` | `gpt-5.6-luna` | OpenAI-compatible fallback model. |
| `LIGHTRAG_AGENT_CODEX_HOME` | Codex default | Optional credential/config directory. |
| `LIGHTRAG_AGENT_MAX_TOKENS` | `8192` | MCPHub output-token ceiling. |
| `LIGHTRAG_AGENT_COMMAND_TIMEOUT_SECONDS` | `900` | Per-provider wall-clock timeout. |
| `LIGHTRAG_AGENT_CIRCUIT_FAILURE_THRESHOLD` | `3` | Failures before opening a provider circuit. |
| `LIGHTRAG_AGENT_CIRCUIT_COOLDOWN_SECONDS` | `300` | Open-circuit cooldown. |
| `MCPHUB_BASE_URL` | `https://api.mcphub.com/v1` | OpenAI-compatible fallback endpoint. |
| `MCPHUB_API_KEY` | none | MCPHub bearer credential. |
| `CLAUDE_CODE_OAUTH_TOKEN(_1.._5)` | local login | Optional Claude OAuth token pool. |

Callers can also construct `CascadeConfig` directly or inject provider
implementations for tests and custom runtimes.
