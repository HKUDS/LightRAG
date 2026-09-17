# Query-time extra provider kwargs

Read this before changing how `kg_query` / `naive_query` / `extract_keywords_only`
/ `_perform_kg_search` in `lightrag/operate.py`, `bypass` mode in
`LightRAG.aquery_llm` (`lightrag/lightrag.py`), or `apply_rerank_if_enabled` /
`process_chunks_unified` in `lightrag/utils.py` obtain or call a queue-wrapped
provider func, or before adding a field to `QueryParam` that also needs to
reach one of those calls.

## The problem this solves

A caller sometimes needs one query's outbound provider call to carry data
scoped to that single request -- most commonly a caller identity (from its own
inbound request) forwarded as an HTTP header, for a downstream proxy or
observability layer to attribute the call correctly. A single query can make
three kinds of provider call: an LLM completion, an embedding (to vector-search
for the query itself, and for keywords in local/global/hybrid/mix mode), and a
rerank call. All three go through the same dispatch mechanism and hit the same
problem.

The obvious approach -- set a `contextvars.ContextVar` just before calling
`aquery`, read it inside the custom `llm_model_func` / `embedding_func` /
`rerank_model_func` -- does not work with LightRAG's dispatch. Each of these is
wrapped once by `priority_limit_async_func_call` (`lightrag/utils.py`) in a
fixed pool of persistent `asyncio.Task` workers, created lazily on first use
and never recreated per call. A call is handed to a worker as a plain tuple on
an `asyncio.PriorityQueue`; the worker executes it inside *its own* task
context, captured when the worker task was created, long before this request
existed. `ContextVar.set()` calls made by the caller's task are invisible
there -- correct `asyncio` behavior, not a bug in the queue.

## Why `QueryParam` fields instead

The queue hand-off carries plain positional/keyword arguments through
correctly; only `contextvars` state does not survive it. So instead of
bypassing the queue, three `QueryParam` fields (`lightrag/base.py`) let a
caller attach arbitrary data as ordinary keyword arguments, merged in right
before the call each one covers:

- `extra_llm_kwargs` -- `kg_query`'s and `naive_query`'s main synthesis calls,
  `extract_keywords_only`'s keyword-extraction call, and `bypass` mode's
  direct call in `aquery_llm` (four sites).
- `extra_embedding_kwargs` -- `_perform_kg_search`'s batched query/keyword
  embedding pre-compute (covers local/global/hybrid/mix mode), and a second,
  parallel pre-compute added to `naive_query` for the same purpose.
- `extra_rerank_kwargs` -- the rerank call inside `apply_rerank_if_enabled`,
  reached through `process_chunks_unified` from both `kg_query` and
  `naive_query`.

Three fields rather than one: the LLM, embedding and rerank functions are
different callables with different signatures, often different providers
entirely: forcing one caller-supplied dict onto all three risks a key that is
valid for one colliding with, or being silently rejected by, another.

The data then rides the same tuple through the same queue as everything else,
keeping the queue's concurrency limiting, timeout handling and priority
ordering intact. This is purely additive: every field defaults to `None`, and
an empty dict is spread in as a no-op for every existing caller.

## Collision behavior: `extra_llm_kwargs`

Each LLM call site literal already sets some keywords itself (`system_prompt`,
`history_messages`, `enable_cot`, `stream`, or `response_format` on the
keyword-extraction call). A key in `extra_llm_kwargs` that repeats one of
those lands twice in the *same* call expression -- `f(stream=x, **{"stream":
y})` -- which Python rejects with `TypeError: ...got multiple values for
keyword argument '...'` before the call is even made. This is deliberate and
requires no code of its own: it is plain Python call syntax, so it cannot
drift out of sync with the call site's own keyword list the way a
hand-maintained reserved-name check could.

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

## Failure behavior: `extra_embedding_kwargs` / `extra_rerank_kwargs`

Both call sites these cover already catch every exception and degrade
gracefully by design: the embedding pre-compute falls back to a backend's
on-demand embedding, and the rerank call falls back to the unreranked chunk
order. That is the wrong default the moment a caller has asked for per-call
data to be forwarded -- silently retrying without it defeats the point, and
for an attribution use case can be worse than failing outright.

So setting either field disables that call's graceful degradation: any
exception, not only one caused by the extra kwargs, propagates instead of
being swallowed. This is a plain `if extra_kwargs: raise` in each `except`
block, not a duplicate-keyword collision like `extra_llm_kwargs` gets for
free, because neither call site rejects a colliding keyword the way a literal
call expression does -- both build their kwargs as a single dict before the
call.

## Rejected alternative

Extending `RoleLLMConfig` (`lightrag/llm_roles.py`) to carry this instead was
considered and rejected: `RoleLLMConfig` is deliberately per-role, set at
`LightRAG` construction or through `update_llm_role_config`, and every value
in it is static across every call that role handles. Per-request data needs a
new value on every single call, which only a `QueryParam` field -- constructed
fresh per query -- can express. The same reasoning ruled out a single shared
field across all three kinds of call, above.

## Known limitation

This covers query-time calls only. It does not help the ingestion-time
`extract` / `vlm` LLM roles or document-chunk embedding, all of which run
per-document-chunk with no `QueryParam` in scope, or any direct provider call
outside a `QueryParam`-carrying entry point. `pick_by_vector_similarity`
(`lightrag/utils.py`)'s own embedding fallback is not separately wired: in the
normal flow it only re-embeds when `_perform_kg_search` did not already supply
an embedding, so `extra_embedding_kwargs` reaches it transitively whenever it
would otherwise fire.
