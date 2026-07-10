"""Shared template for native (local, in-process) parser engines.

``NativeParserBase.parse`` fixes the common local-parse flow once:

    resolve + validate source → compute parsed_dir/asset_dir
    → pre-clean (rmtree parsed_dir + mkdir + mkdir asset_dir, with rollback)
    → extract() in a thread → build_ir() → write_sidecar(clean_parsed_dir=False)
    → persist full_docs (lightrag) → archive source

Subclasses implement ``extract`` (sync, runs in a thread) and ``build_ir``.
Currently only :class:`NativeDocxParser`; xlsx/pptx/md land later as new
subclasses implementing the same two hooks.
"""

from __future__ import annotations

import asyncio
import contextvars
import shutil
import threading
import time
from abc import abstractmethod
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightrag.constants import FULL_DOCS_FORMAT_LIGHTRAG
from lightrag.parser.base import BaseParser, ParseContext, ParseResult
from lightrag.utils import logger

if TYPE_CHECKING:
    from lightrag.sidecar.ir import IRDoc


@dataclass(frozen=True)
class NativeExtractRuntime:
    """Per-parse runtime handed to :meth:`NativeParserBase.extract`.

    Bundles the per-file state the async template resolves BEFORE entering the
    worker thread: the decoded ``parse_engine`` params, an optional synchronous
    LLM callable (built only when :meth:`NativeParserBase.wants_llm_bridge`
    says the params need one), and the cancellation event the bridge polls.
    The three travel together — a subclass that consumes none of them simply
    ignores the argument.
    """

    engine_params: Mapping[str, Any] = field(default_factory=dict)
    llm_invoke: Callable[..., str] | None = None
    cancel_event: threading.Event | None = None


class NativeParserBase(BaseParser):
    """Base for engines that parse a file locally into a sidecar."""

    # ``write_sidecar`` block_drawing_path_style; docx keeps the legacy
    # "basename_only" shape for byte-equivalence.
    sidecar_path_style: str = "with_prefix"
    # Prefix used in the "empty content" error message.
    empty_content_label: str = "Native"

    # --- engine-private hooks ------------------------------------------------
    def validate_source(self, source: Path, file_path: str) -> None:
        """Validate the resolved source (default: must be an existing file)."""
        if not (source.exists() and source.is_file()):
            raise FileNotFoundError(
                f"{self.engine_name} source file not found: {source}"
            )

    @abstractmethod
    def extract(
        self,
        source: Path,
        *,
        parsed_dir: Path,
        asset_dir: Path,
        base_name: str,
        runtime: NativeExtractRuntime | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        """Extract ``(blocks, warnings, metadata)`` (sync; runs in a thread).

        ``parsed_dir`` and ``asset_dir`` are pre-created by the template; the
        hook may write side artifacts (e.g. image bytes) into ``asset_dir``
        before :func:`write_sidecar` runs with ``clean_parsed_dir=False``.
        ``runtime`` carries the decoded per-file engine params (and, when
        those request it, a synchronous LLM bridge); engines without tunable
        params ignore it.
        """
        ...

    def wants_llm_bridge(self, engine_params: Mapping[str, Any]) -> bool:
        """Whether these engine params require the synchronous LLM bridge.

        The base stays engine-agnostic — only a concrete suffix implementation
        knows which of its params imply LLM work (docx overrides this for
        ``smart_heading``).
        """
        return False

    def _build_llm_submit(
        self, ctx: ParseContext
    ) -> tuple[Callable[..., Coroutine[Any, Any, str]] | None, list, bool]:
        """Build the loop-side async LLM entry for the bridge.

        Returns ``(submit, cache_keys_collector, i4_cache_disabled)``.
        ``submit`` is ``None`` when the rag stand-in has no LLM surface
        (debug CLI / golden tests without injection) — the algorithm layer
        hard-fails later only if it actually needs the LLM (a short-document
        gate may skip it entirely). ``i4_cache_disabled`` flags the I4
        determinism waiver so parse() can surface it as a parse warning
        (§2.3.5 channel a), not just a log line.

        Uses the EXTRACT role func under the dedicated ``smartheading`` cache
        namespace: title-block judgment is its own semantics, and hits must
        never collide with entity-extraction prompts. Collected cache keys
        exist solely so ``adelete_by_doc_id(delete_llm_cache=True)`` can
        purge parse-stage LLM cache (there is no chunk to carry an
        ``llm_cache_list`` at parse time).
        """
        rag = ctx.rag
        build_config = getattr(rag, "_build_global_config", None)
        if build_config is None:
            return None, [], False
        global_config = build_config()
        llm_func = (global_config.get("role_llm_funcs") or {}).get("extract")
        if llm_func is None:
            return None, [], False

        from lightrag.utils import get_llm_cache_identity, use_llm_func_with_cache

        i4_cache_disabled = not global_config.get(
            "enable_llm_cache_for_entity_extract", True
        )
        if i4_cache_disabled:
            # I4 (deterministic re-parse) relies on LLM cache hits; without
            # them repeated parses may differ. Documented waiver, not an error.
            logger.warning(
                "[%s] enable_llm_cache_for_entity_extract is off: smart_heading "
                "LLM judgments will not be cached, so repeated parses of the "
                "same file may produce different results (I4 waiver)",
                self.engine_name,
            )

        identity = get_llm_cache_identity(global_config, "extract")
        llm_response_cache = getattr(rag, "llm_response_cache", None)
        collector: list = []

        async def _submit(prompt: str, *, system_prompt: str | None = None) -> str:
            content, _timestamp = await use_llm_func_with_cache(
                prompt,
                llm_func,
                llm_response_cache=llm_response_cache,
                system_prompt=system_prompt,
                cache_type="smartheading",
                cache_keys_collector=collector,
                llm_cache_identity=identity,
            )
            return content

        return _submit, collector, i4_cache_disabled

    @abstractmethod
    def build_ir(
        self,
        blocks: list[dict[str, Any]],
        *,
        document_name: str,
        asset_dir_name: str,
        metadata: dict[str, Any],
    ) -> "IRDoc": ...

    def surface_warnings(
        self, warnings: dict[str, Any], source: Path
    ) -> dict[str, Any] | None:
        """Map parser warnings to the ``parse_warnings`` result field (opt)."""
        return None

    # --- template ------------------------------------------------------------
    async def parse(self, ctx: ParseContext) -> ParseResult:
        from lightrag.parser.routing import decode_parse_engine, encode_parse_engine
        from lightrag.sidecar import write_sidecar
        from lightrag.utils_pipeline import (
            make_lightrag_doc_content,
            sidecar_uri_for,
        )

        # Per-file engine params ride the stored ``parse_engine`` directive
        # (e.g. ``native(smart_heading=true)``). A malformed/corrupt directive
        # fails this doc loudly rather than silently parsing with no params
        # (same contract as the external engines).
        _engine, engine_params, decode_errs = decode_parse_engine(
            ctx.content_data.get("parse_engine")
            if isinstance(ctx.content_data, dict)
            else None
        )
        if decode_errs:
            raise ValueError(
                f"{self.engine_name}: invalid parse_engine for doc_id={ctx.doc_id}: "
                + "; ".join(decode_errs)
            )
        # A directive naming a DIFFERENT engine reaching this parser means a
        # corrupt/misrouted row — fail loudly instead of silently re-branding
        # foreign params as our own on persist (review, native_base cross-check).
        if _engine and _engine != self.engine_name:
            raise ValueError(
                f"{self.engine_name}: parse_engine names a different engine "
                f"{_engine!r} for doc_id={ctx.doc_id}"
            )
        engine_params = engine_params or {}

        # Per-parse cancel event, polled by the LLM bridge between waits. The
        # rag-level shutdown event (when present) covers finalize_storages
        # while an extract is still in flight.
        cancel_event = threading.Event()
        llm_invoke = None
        smartheading_cache_keys: list = []
        i4_cache_disabled = False
        if self.wants_llm_bridge(engine_params):
            submit, smartheading_cache_keys, i4_cache_disabled = self._build_llm_submit(
                ctx
            )
            if submit is not None:
                from lightrag.parser.llm_bridge import (
                    LLMBridgePipelineCancelled,
                    LLMBridgeShutdown,
                    SyncLLMBridge,
                )

                shutdown_event = getattr(ctx.rag, "_parser_shutdown_event", None)
                llm_invoke = SyncLLMBridge(
                    asyncio.get_running_loop(),
                    submit,
                    cancel_events=(
                        cancel_event,
                        (
                            ctx.pipeline_cancel_event,
                            LLMBridgePipelineCancelled,
                        ),
                        (shutdown_event, LLMBridgeShutdown),
                    ),
                )
        runtime = NativeExtractRuntime(
            engine_params=engine_params,
            llm_invoke=llm_invoke,
            cancel_event=cancel_event,
        )

        rs = ctx.resolve(self.engine_name)
        source = rs.source_path
        self.validate_source(source, ctx.file_path)

        document_name = rs.document_name
        base_name = Path(document_name).stem or document_name
        parsed_dir = rs.parsed_dir
        asset_dir = parsed_dir / f"{base_name}.blocks.assets"

        def _extract_sync():
            # Pre-clean parsed_dir and pre-create asset_dir so the extractor
            # can write image bytes BEFORE write_sidecar (clean_parsed_dir=False
            # then keeps them). parsed_artifact_dir_for returns a unique dir per
            # source, so this rmtree only clobbers a prior attempt's artifacts.
            if parsed_dir.exists():
                shutil.rmtree(parsed_dir)
            parsed_dir.mkdir(parents=True, exist_ok=True)
            asset_dir.mkdir(parents=True, exist_ok=True)
            return self.extract(
                source,
                parsed_dir=parsed_dir,
                asset_dir=asset_dir,
                base_name=base_name,
                runtime=runtime,
            )

        # Prefer the rag-owned parser executor (its pool size tracks
        # max_parallel_parse_native and an LLM wait can hold a thread for
        # seconds — the process-default to_thread pool must not be starved).
        # Debug/golden rag stand-ins lack it and fall back to to_thread;
        # copy_context() preserves the contextvars propagation to_thread does.
        executor_getter = getattr(ctx.rag, "_get_parse_native_executor", None)
        try:
            if executor_getter is not None:
                (
                    blocks,
                    warnings,
                    metadata,
                ) = await asyncio.get_running_loop().run_in_executor(
                    executor_getter(), contextvars.copy_context().run, _extract_sync
                )
            else:
                blocks, warnings, metadata = await asyncio.to_thread(_extract_sync)
            # A15 (§2.3.5 channel a): the I4 determinism waiver is a
            # warning-grade event — surface it via parse_warnings →
            # doc_status, not only the process log.
            if llm_invoke is not None and i4_cache_disabled:
                warnings["smart_i4_cache_disabled"] = 1
        except BaseException:
            # Unblock a bridge poller promptly (idempotent), THEN roll back
            # the pre-created (possibly partial) dirs. The worker thread may
            # briefly outlive the rmtree; the pre-clean at the next parse
            # attempt sweeps any late writes.
            cancel_event.set()
            if parsed_dir.exists():
                shutil.rmtree(parsed_dir, ignore_errors=True)
            raise
        if not blocks:
            if parsed_dir.exists():
                shutil.rmtree(parsed_dir, ignore_errors=True)
            raise ValueError(
                f"{self.empty_content_label} parser returned empty content "
                f"for {ctx.file_path}"
            )

        parse_warnings = self.surface_warnings(warnings, source)
        ir = self.build_ir(
            blocks,
            document_name=document_name,
            asset_dir_name=asset_dir.name,
            metadata=metadata,
        )
        parsed_data = write_sidecar(
            ir,
            parsed_dir=parsed_dir,
            doc_id=ctx.doc_id,
            engine=self.engine_name,
            clean_parsed_dir=False,  # asset dir pre-populated above
            block_drawing_path_style=self.sidecar_path_style,
        )

        await ctx.rag._persist_parsed_full_docs(
            ctx.doc_id,
            {
                "content": make_lightrag_doc_content(parsed_data["content"]),
                "file_path": ctx.file_path,
                "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
                "sidecar_location": sidecar_uri_for(parsed_dir),
                # Re-encode the engine + params so the persisted directive keeps
                # the per-file params (the `{**existing, **record}` merge in
                # _persist_parsed_full_docs would otherwise revert it to the
                # bare engine name). No params encodes back to the bare name.
                "parse_engine": encode_parse_engine(
                    self.engine_name, engine_params or None
                ),
                "update_time": int(time.time()),
            },
        )
        await ctx.archive_source(str(source))
        return ParseResult(
            doc_id=ctx.doc_id,
            file_path=ctx.file_path,
            parse_format=FULL_DOCS_FORMAT_LIGHTRAG,
            content=parsed_data["content"],
            blocks_path=parsed_data["blocks_path"],
            parse_engine=self.engine_name,
            parse_warnings=parse_warnings,
            smartheading_llm_cache_ids=(
                list(dict.fromkeys(smartheading_cache_keys)) or None
            ),
        )
