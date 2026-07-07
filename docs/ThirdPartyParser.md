# Third-Party Parser Engine Development and Registration Guide

LightRAG's parsing layer dispatches all parsing engines through a unified `BaseParser` contract plus the central engine registry (`lightrag/parser/registry.py`). Built-in engines (`native` / `legacy` / `mineru` / `docling` / `paddleocr_vl`) and third-party engines use the exact same dispatch path: both pipeline workers and the debug CLI drive parsing through `get_parser(engine).parse(ParseContext(...))`, with **no special cases for built-in engines**. Therefore, a third-party package only needs to do two things:

1. **Implement** a `BaseParser` subclass;
2. **Register** a `ParserSpec` (automatic discovery through the `lightrag.parsers` entry point is recommended).

Once that is done, the engine automatically gets: a dedicated (or shared) parsing concurrency pool, three engine-selection methods (filename hints / `LIGHTRAG_PARSER` routing rules / API `parse_engine` parameter), suffix capability validation, and single-file debug support through `python -m lightrag.parser.cli --engine <name>`.

> For architecture background, see RFC #3197; for the sidecar file format, see `docs/LightRAGSidecarFormat.md`; for CLI usage, see `docs/ParserDebugCLI.md`.

---

## 1. Dispatch Flow Overview

```
Upload/scan -> enqueue(PENDING_PARSE, parse_engine=<engine>)
    -> pipeline selects a concurrency pool by ParserSpec.queue_group
    -> parse worker: get_parser(engine).parse(ParseContext(rag, doc_id, file_path, content_data))
        +- success -> ParseResult -> enter analyze / chunk / KG pipeline
        +- exception -> this document's doc_status=FAILED (only this document is affected)
```

For a single document, all engine responsibilities converge into one `parse(ctx)` call: parsing, persisting `full_docs`, archiving the source file, and returning a structured result.

## 2. Implementing a Parser

### 2.1 Contract (`lightrag/parser/base.py`)

```python
class MyParser(BaseParser):
    engine_name = "myengine"          # Must match ParserSpec.engine_name

    async def parse(self, ctx: ParseContext) -> ParseResult: ...
```

`ParseContext` provides:

| Member | Description |
|---|---|
| `ctx.rag` | LightRAG instance (used for `_persist_parsed_full_docs`, etc.) |
| `ctx.doc_id` / `ctx.file_path` / `ctx.content_data` | Document identifier, normalized file path, and `full_docs` row |
| `ctx.resolve(engine_name)` | Returns `ResolvedSource(source_path, document_name, parsed_dir)`: resolves the on-disk source file path, normalized document name, and derived `__parsed__/<base>.parsed/` artifact directory |
| `ctx.archive_source(path)` | After parsing succeeds and `full_docs` synchronization is complete, archives the source file into `__parsed__/` |

`ParseResult` fields: `doc_id` / `file_path` / `parse_format` (`"raw"` or `"lightrag"`) / `content` / `blocks_path` (`""` when there is no sidecar) / `parse_engine` / `parse_stage_skipped` (skipped scenarios such as cache hits) / `parse_warnings` (non-fatal warnings, persisted to `doc_status.metadata`).

### 2.2 Three Implementation Paths (Choose a Base Class by Engine Type)

**A. Plain-text engine (no sidecar) - inherit `BaseParser` directly**

See `lightrag/parser/legacy/parser.py` (`LegacyParser`). Core skeleton:

```python
class MyTextParser(BaseParser):
    engine_name = "myengine"

    async def parse(self, ctx: ParseContext) -> ParseResult:
        rs = ctx.resolve(self.engine_name)
        source = rs.source_path
        if not source.is_file():
            raise FileNotFoundError(f"myengine source not found: {source}")

        text = await asyncio.to_thread(my_extract, source)  # Run CPU work in a thread
        if not text.strip():
            raise ValueError(f"extracted no usable text from {ctx.file_path}")

        await ctx.rag._persist_parsed_full_docs(ctx.doc_id, {
            "content": text,
            "file_path": ctx.file_path,
            "parse_format": FULL_DOCS_FORMAT_RAW,
            "parse_engine": self.engine_name,
            "update_time": int(time.time()),
        })
        await ctx.archive_source(str(source))
        return ParseResult(
            doc_id=ctx.doc_id, file_path=ctx.file_path,
            parse_format=FULL_DOCS_FORMAT_RAW, content=text,
            blocks_path="", parse_engine=self.engine_name,
        )
```

**B. Local parser that produces a sidecar - inherit `NativeParserBase`** (`lightrag/parser/native_base.py`)

The template fixes the complete flow of "pre-clean artifact directory (with rollback) -> extract in a thread -> build IR -> write sidecar -> persist -> archive". You only need to implement two hooks:

```python
class MyNativeParser(NativeParserBase):
    engine_name = "myengine"

    def extract(self, source, *, parsed_dir, asset_dir, base_name):
        """Synchronous, runs in a thread; returns (blocks, warnings, metadata).
        Assets such as images can be written to asset_dir before write_sidecar."""

    def build_ir(self, blocks, *, document_name, asset_dir_name, metadata) -> IRDoc:
        """blocks -> IRDoc (handed to the shared sidecar writer)."""
```

Optional overrides: `validate_source` (by default only requires the file to exist), `surface_warnings` (maps extraction warnings to `parse_warnings`). Reference implementation: `lightrag/parser/docx/parser.py`.

**C. External parsing service (download raw bundle + cache) - inherit `ExternalParserBase`** (`lightrag/parser/external/_base.py`)

The template fixes the flow of "raw cache-hit check -> if missed, clear the directory and download again -> build IR -> write sidecar -> persist -> archive". Implement three hooks plus two class attributes:

```python
class MyExternalParser(ExternalParserBase):
    engine_name = "myengine"
    raw_dir_suffix = ".myengine_raw"          # Raw bundle directory suffix (starts with .)
    force_reparse_env = "LIGHTRAG_FORCE_REPARSE_MYENGINE"

    def is_bundle_valid(self, raw_dir, source_path) -> bool: ...   # Cache-hit check
    async def download_into(self, raw_dir, source_path, *, upload_name): ...
    def build_ir(self, raw_dir, document_name) -> IRDoc: ...
```

Optional override: `validate_ir` (post-build validation, for example failing on zero blocks). Reference implementations: `lightrag/parser/external/mineru/parser.py`, `.../docling/parser.py`, and `.../paddleocr_vl/parser.py`.

### 2.3 Failure Semantics (Important)

- If `parse(ctx)` **raises any exception, only that document is marked FAILED**. The error message is written to `doc_status.error_msg`, and other documents in the batch are not affected.
- If parsing produces **empty content, raise an exception** instead of returning an empty string. Otherwise, a zero-knowledge document would silently enter chunking (all built-in engines follow this convention).
- Before calling the engine, the worker performs suffix guarding: if a `PENDING_PARSE` document's suffix is not included in that engine's `ParserSpec.suffixes`, the document is FAILED directly and the engine code is never called.

## 3. Declaring `ParserSpec` (Capability Metadata)

```python
from lightrag.parser.registry import ParserSpec, register_parser

register_parser(ParserSpec(
    engine_name="myengine",
    impl="my_pkg.parser:MyParser",       # "module:Class", lazy-loaded by get_parser
    suffixes=frozenset({"pdf", "foo"}),  # Lowercase, no dot
    queue_group="myengine",              # See concurrency model below
    concurrency=int(os.getenv("MAX_PARALLEL_PARSE_MYENGINE", "2")),
    # Required only by external-service engines (routing skips this engine when no endpoint is configured):
    endpoint_configured=lambda: bool(os.getenv("MYENGINE_ENDPOINT", "").strip()),
    endpoint_requirement=lambda: "MYENGINE_ENDPOINT",
))
```

| Field | Required | Description |
|---|---|---|
| `engine_name` | yes | Registry key; also the engine name used by `--engine`, filename hints, and `LIGHTRAG_PARSER`. **Registering the same name as an existing engine overwrites the previous registration** (including built-in engines). Avoid colliding with `native/legacy/mineru/docling/paddleocr_vl` unless you intentionally want to replace that implementation. |
| `impl` | yes | `"module:Class"` string. The registry imports it only when a document is actually parsed. **The implementation must never be imported early during registration** (capability queries must stay import-cheap; this is a registry design invariant). |
| `suffixes` | yes | File extensions the engine can handle (lowercase, no dot). Used for routing validation and worker-side suffix guarding. |
| `queue_group` | | Concurrency-pool group. Defaults to `"native"` (sharing the native pool). Use a unique group name for a dedicated pool. |
| `concurrency` | | Number of workers for this group (only needed for the group's sole owner). Environment-variable overrides are **baked by the registration code at registration time** (as in the `int(os.getenv(...))` example above); the registered value is authoritative. |
| `endpoint_configured` / `endpoint_requirement` | | Zero-argument closures (read env only, no network calls). The former returns whether the external service required by this engine is configured; the latter returns the configuration item name to show users when it is missing. Local engines do not need these fields (available by default). |
| `user_selectable` | | Defaults to `True`. `False` means an internal format handler (such as `reuse` / `passthrough`) that is not shown as a selectable engine. |

### Concurrency Model

- For each batch, the pipeline creates **one queue plus one worker group for every `queue_group`**;
- Worker count for a group: built-in groups (`native` / `mineru` / `docling` / `paddleocr_vl`) are determined by LightRAG instance fields `max_parallel_parse_*` (constructor overrides are supported); a third-party dedicated group uses the `concurrency` value from the group's sole owner spec; **only one** spec in a group may declare `concurrency`, otherwise batch startup fails;
- When using `queue_group="native"` to share the built-in pool, `concurrency` does not take effect (pool size is determined by `max_parallel_parse_native`; ignored spec-level `concurrency` values are recorded as warning logs at batch startup). Lightweight local engines (such as `legacy`) fit this mode, while external-service engines should usually use a dedicated group so slow requests do not block local parsing.

## 4. Registration: Automatic Discovery via Entry Point (Recommended)

LightRAG automatically discovers third-party engines through the `lightrag.parsers` entry-point group (`lightrag/parser/plugins.py`). A third-party package only needs to:

**1. Declare the entry point in its own `pyproject.toml`:**

```toml
[project.entry-points."lightrag.parsers"]
myengine = "my_pkg.lightrag_plugin:register"
```

**2. Provide a zero-argument registration function:**

```python
# my_pkg/lightrag_plugin.py - keep imports cheap: do not import parser implementations here
import os
from lightrag.parser.registry import ParserSpec, register_parser

def register() -> None:
    register_parser(ParserSpec(
        engine_name="myengine",
        impl="my_pkg.parser:MyParser",   # Implementation is lazy-loaded
        suffixes=frozenset({"foo"}),
        queue_group="myengine",
        concurrency=int(os.getenv("MAX_PARALLEL_PARSE_MYENGINE", "2")),
    ))
```

After `pip install my-pkg`, it works immediately without changing LightRAG code:

- **API Server**: `create_app()` calls `load_third_party_parsers()` **before** validating `LIGHTRAG_PARSER` routing rules, so routing rules can directly reference third-party engine names (for example, `LIGHTRAG_PARSER="foo:myengine"`). File-type guarding for uploads and scans is **derived entirely from the registry plus routing at runtime**. The criterion is "can this file route to an engine that supports it": a bare suffix (no hint) requires a `LIGHTRAG_PARSER` rule that routes it to your engine (otherwise the default `legacy` engine cannot handle it, so the file is rejected instead of being accepted and later FAILED during parsing); uploads with a filename hint (for example, `report.[myengine].foo`) pass without a rule; unique suffixes of engines whose endpoints are not configured do not participate (the same rule used by built-in mineru/docling image formats). **Practical recommendation: when publishing a third-party engine, tell users to configure the matching `LIGHTRAG_PARSER="foo:myengine"` rule in deployment docs**, so bare-filename uploads and directory scans work automatically;
- **Debug CLI**: `python -m lightrag.parser.cli sample.foo --engine myengine` works directly (`main()` loads plugins before building `--engine` choices). For engines without sidecars (`blocks_path=""`), the CLI prints a plain-text summary instead of a block summary; engines inheriting `ExternalParserBase` automatically get raw cache display and `--force-reparse` support;
- **Embedded library usage** (using the `LightRAG` class directly without going through the server or CLI): call this once before building the pipeline:

```python
from lightrag.parser.plugins import load_third_party_parsers
load_third_party_parsers()   # Idempotent within the process
```

Loading semantics: idempotent per process (repeated calls are no-ops); **if a single plugin raises an exception, it is only logged and skipped**. It does not affect other plugins or built-in engines, and it does not block server startup. However, that engine will be unavailable, so watch for `[parser-plugins]` lines in startup logs.

> If you do not want to publish a package, you can skip entry points and directly call `register_parser(...)` in your own startup script before starting/calling LightRAG. The registry is an in-process module singleton, so the effect is the same, just without "install and it works" behavior.

## 5. Routing: Making Documents Use Your Engine

Engine selection priority (`lightrag/parser/routing.py`):

1. **Filename hint**: `report.[myengine].foo` (processing options are allowed, such as `report.[myengine-iet].foo`);
2. **`LIGHTRAG_PARSER` rules**: for example, `LIGHTRAG_PARSER="foo:myengine,pdf:mineru"` (matched by suffix glob; the first match wins);
3. **Default**: `legacy`.

When API upload explicitly passes `parse_engine="myengine"`, that engine is fixed directly (stored in the `PENDING_PARSE` row and honored verbatim by the worker; unsupported suffixes are FAILED instead of silently falling back). Engines that register `endpoint_configured` are skipped by routing when the endpoint is not configured (hint/rule validation also shows the `endpoint_requirement` prompt).

## 6. Testing Recommendations

- **Unit-test the engine itself**: bypass the CLI and directly call `get_parser("myengine").parse(ParseContext(fake_rag, doc_id, file_path, content_data))`. `fake_rag` only needs to provide `_persist_parsed_full_docs` / `_resolve_source_file_for_parser` / `full_docs` / `doc_status` (see `build_debug_rag()` in `lightrag/parser/debug.py`, or the minimal `_FakeRag` in `tests/parser/test_legacy_parser.py`);
- **The registry is a module-level singleton**: after calling `register_parser` in a test, clean up with `finally: registry._REGISTRY.pop("myengine", None)` (see `tests/parser/test_registry.py`);
- **Entry-point loading logic**: see `tests/parser/test_plugins.py` (monkeypatch `lightrag.parser.plugins.entry_points` to inject a fake entry point; the `plugins._loaded` flag must be reset).
