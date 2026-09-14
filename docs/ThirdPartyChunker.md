# Third-party chunkers

An installed package can publish a legacy six-argument chunker without modifying LightRAG. The Server discovers registrations at startup and injects one selected callback. This implements [#3868](https://github.com/HKUDS/LightRAG/issues/3868); it does not add `C(name=...)`, context-aware callbacks, identity enforcement, multiple active chunkers, or a transform hook for F/R/V/P.

## Publish an import-cheap registration

In your package's `pyproject.toml`:

```toml
[project.entry-points."lightrag.chunkers"]
acme = "acme_chunker.plugin:register"
```

Keep both the package `__init__.py` and registration module free of implementation imports:

```python
# acme_chunker/plugin.py
from lightrag.chunker.registry import ChunkerSpec, register_chunker

def register():
    register_chunker(ChunkerSpec(
        name="acme",
        impl="acme_chunker.implementation:chunk",
        version="1",
        description="Organization-specific document splitting",
        executor_safe=False,
    ))
```

The zero-argument function may register multiple specs. LightRAG stores their metadata without importing implementations; only the selected `impl` is resolved. Plugin registration is trusted installed Python code, not a sandbox: authors must not eagerly import implementations, perform network work, or initialize process/thread resources there.

Names must match `[a-z0-9][a-z0-9_-]{0,63}`. Reserved names are `fixed_token`, `recursive_character`, `semantic_vector`, `paragraph_semantic`, and `f/r/v/p/c`. `version` is an opaque string for diagnostics, not compatibility enforcement. A one-line description is required. Optional spec fields can be extended without changing the entry-point contract.

Discovery is once per process; Gunicorn preload resolves before forking, and workers inherit the same registration/selection snapshot. Keep implementation imports preload-safe too: allocate loop-bound/network resources inside the invocation, not at import. Restart the deployment after installing/updating packages or changing selection. Registration order is deterministic by distribution and entry-point origin. A duplicate logs both origins at ERROR; a selected duplicate fails startup, while an unselected duplicate retains the last registration. A failing provider is skipped with its origin logged, and its partial registrations are discarded; selection of a name it failed to provide then fails as unknown.

## Implement the existing callback contract

```python
def chunk(tokenizer, content, split_by_character, split_by_character_only,
          chunk_overlap_token_size, chunk_token_size):
    # Return your algorithm's ordered list of dictionaries, each containing
    # tokens (int), content (str), and chunk_order_index (int).
    ...
```

Both synchronous and async callbacks are supported, and receive exactly these six positional arguments (no private `_emit_source_span` keyword). Default invocation is on the event loop, including synchronous factories returning awaitables. Exceptions fail the document rather than selecting a different algorithm. Existing downstream chunk validation and embedding-size handling still apply; custom text is not eligible for source-span sidecar backfill.

For CPU-bound, synchronous, thread-safe code, set `executor_safe=True` to use LightRAG's existing bounded chunking executor. Such code must not need the running event loop or return awaitables. An async function/async callable object with that flag is rejected at startup. Leave it false for loop-dependent implementations, or offload explicitly inside an async callback. The declaration is an author promise, not a sandbox or a proof of thread safety.

## Select in the Server

Install the package into the same environment as LightRAG, then set:

```dotenv
CUSTOM_CHUNKER=acme
```

Or use `lightrag-server --custom-chunker acme` / `lightrag-gunicorn --custom-chunker acme` (CLI wins over `.env`). The value is a bare registered name. Python import paths, file paths and code are not configuration values; HTTP payloads cannot select an implementation either.

Startup fails for an unknown/ambiguous selection, import or attribute errors, a non-callable implementation, or an introspectable callable that cannot accept six positional arguments. Native callables without an inspectable signature remain allowed; their output/runtime contract still applies.

The selected callback is injected as `LightRAG(chunking_func=...)`, so it serves **both explicit C and no-selector inserts** instance-wide. F/R/V/P continue to select built-ins and retain the existing bypass diagnostic. Use `chunking.strategy="custom"` in text APIs or `C` in filename/routing hints for explicit selection. Unset means no constructor override: caller-present C remains 422, and persisted/background C still warns and uses exact fixed-token fallback. Merely installing a plugin does not select it.

## SDK embedding

```python
from lightrag import LightRAG
from lightrag.chunker.plugins import load_and_resolve_chunker

callback = load_and_resolve_chunker("acme")
rag = LightRAG(chunking_func=callback)  # Add your normal storage/LLM config.
# await rag.initialize_storages() before inserting; finalize when finished.
```

SDK users may also call `register_chunker(spec, origin="my-application")` directly. An unset resolution returns `None`: omit the constructor argument in that case, rather than passing `chunking_func=None`.

For discovery without selection, call `load_third_party_chunkers()`, then use `registered_chunker_names()` or `resolve_chunker(name)` from the registry. This discovery-only API logs failures without claiming that a selection has been validated.

## Diagnostics and reprocessing

One startup INFO line lists registered names, versions, descriptions, origins and the selection, explicitly noting the no-selector impact. Discovery failures are ERRORs and do not break an unrelated valid selection. Each Server failure line includes the outcome after selection validation: the named selected chunker remains available for C/no-selector inserts, startup aborted because selection failed, or the unset configuration preserves existing C admission/fallback behavior. A successful startup validation does not guarantee a plugin's later runtime/output correctness.

`doc_status.metadata.custom_chunker` stores the most recent attempted identity as `{"name": "acme", "version": "1", "authoritative": false}` separately from `chunk_opts`. The existing `chunk_method` strings are unchanged. An attempt without a registered callback records null name/version when replacing a previous observation.

This observation survives reset only for comparison. A persisted C document whose recorded name/version differs from the current configuration emits one drift WARNING per processing attempt, then runs with the **current** callback (or normal warned fallback if it was removed). The fallback warning is separate and keeps its existing per-document/per-attempt cadence. No identity/version string gates processing or chooses a callback. Unchanged identity strings cannot establish unchanged implementation; manage package versions and deployment reproducibility separately. Changing configuration does not automatically reprocess already-processed documents.
