# Query-time extra LLM kwargs

Read this before changing how `kg_query` / `naive_query` / `extract_keywords_only`
in `lightrag/operate.py`, or `bypass` mode in `LightRAG.aquery_llm`
(`lightrag/lightrag.py`), obtain or call the role-wrapped LLM func, or before
adding a field to `QueryParam` that also needs to reach the LLM call.

## The problem this solves

A caller sometimes needs one query's outbound LLM call to carry data that is
scoped to that single request -- most commonly a caller identity (from its own
inbound request) forwarded as an HTTP header, for a downstream LLM proxy or
observability layer to attribute the call correctly.

The obvious approach -- set a `contextvars.ContextVar` just before calling
`aquery`, read it inside a custom `llm_model_func` -- does not work with
LightRAG's dispatch. Each role's LLM func (`extract` / `keyword` / `query` /
`vlm`, `lightrag/llm_roles.py`) is wrapped once by
`priority_limit_async_func_call` (`lightrag/utils.py`) in a fixed pool of
persistent `asyncio.Task` workers, created lazily on first use and never
recreated per call. A call is handed to a worker as a plain tuple on an
`asyncio.PriorityQueue`; the worker executes it inside *its own* task context,
captured when the worker task was created, long before this request existed.
`ContextVar.set()` calls made by the caller's task are invisible there --
correct `asyncio` behavior, not a bug in the queue.

## Why a `QueryParam` field instead

The queue hand-off carries plain positional/keyword arguments through
correctly; only `contextvars` state does not survive it. So instead of
bypassing the queue, `QueryParam.extra_llm_kwargs` (`lightrag/base.py`) lets a
caller attach arbitrary data as ordinary keyword arguments, merged in at each
of the three query-time call sites right before invoking `use_model_func`:

- `kg_query`'s main synthesis call
- `naive_query`'s main synthesis call
- `extract_keywords_only`'s keyword-extraction call

The data then rides the same tuple through the same queue as everything else,
keeping the queue's concurrency limiting, timeout handling and priority
ordering intact. This is purely additive: the field defaults to `None`, and an
empty dict is spread in as a no-op for every existing caller.

A fourth site, `bypass` mode's direct LLM call inside `aquery_llm`
(`lightrag/lightrag.py`), also has a `QueryParam` in scope and gets the same
treatment, even though it skips `operate.py` entirely.

## Collision behavior

Each call site literal already sets some keywords itself (`system_prompt`,
`history_messages`, `enable_cot`, `stream`, or `response_format` on the
keyword-extraction call). A key in `extra_llm_kwargs` that repeats one of
those lands twice in the *same* call expression -- `f(stream=x, **{"stream":
y})` -- which Python rejects with `TypeError: ...got multiple values for
keyword argument '...'` before the call is even made. This is deliberate and
requires no code of its own: it is plain Python call syntax, so it cannot
drift out of sync with the call site's own keyword list the way a hand-maintained
reserved-name check could.

Kwargs bound at an earlier layer -- `hashing_kv` and a role's
`RoleLLMConfig.kwargs` (bound onto the raw func via `functools.partial` in
`_wrap_llm_role_func`, `lightrag/llm_roles.py`) or `_priority` (bound onto the
wrapped func at each `operate.py` call site) -- behave differently.
`functools.partial.__call__` lets the call's keywords *override* the bound
ones rather than erroring on the duplicate, so a colliding key there is
silently overridden instead of raising. `_timeout` / `_queue_timeout` are not
bound by anything upstream and simply pass through as `wait_func`'s own
per-call timeout override -- a real, intentional escape hatch of the queue
wrapper, not a collision.

## Rejected alternative

Extending `RoleLLMConfig` (`lightrag/llm_roles.py`) to carry this instead was
considered and rejected: `RoleLLMConfig` is deliberately per-role, set at
`LightRAG` construction or through `update_llm_role_config`, and every value
in it is static across every call that role handles. Per-request data needs a
new value on every single call, which only a `QueryParam` field -- constructed
fresh per query -- can express.

## Known limitation

This does not help the ingestion-time `extract` / `vlm` roles, which run
per-document-chunk with no `QueryParam` in scope, or any direct use of
`llm_model_func` outside a `QueryParam`-carrying entry point. It only covers
the four query-time call sites named above.
