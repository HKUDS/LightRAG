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
    says the params need one), and the cancellation events any blocking work
    in the worker thread should poll. They travel together — a subclass that
    consumes none of them simply ignores the argument.

    ``cancel_events`` is the full set, in the ``(event, exception_type)`` shape
    :func:`~lightrag.parser.llm_bridge.normalize_cancel_events` accepts, so a
    consumer stays agnostic about which source fired: the per-parse event
    (first entry), the pipeline's (``/documents/cancel_pipeline``), and the
    rag shutdown event.
    """

    engine_params: Mapping[str, Any] = field(default_factory=dict)
    llm_invoke: Callable[..., str] | None = None
    cancel_events: tuple = ()


class NativeParserBase(BaseParser):
    """Base for engines that parse a file locally into a sidecar."""

    # ``write_sidecar`` block_drawing_path_style. All native engines use
    # the spec shape ``<base>.blocks.assets/<filename>`` (a non-empty
    # ``path`` always points inside ``*.blocks.assets/``).
    sidecar_path_style: str = "with_prefix"
    # Prefix used in the "empty content" error message.
    empty_content_label: str = "Native"

    # --- engine-private hooks ------------------------------------------------
    def validate_source(self, source: Path, file_path: str) -> None:
        """Validate the resolved source (default: must be an existing file).

        Runs inline in :meth:`parse`, i.e. ON THE EVENT LOOP, so it must stay
        stat-level. Anything that reads the file belongs in
        :meth:`validate_source_blocking`.
        """
        if not (source.exists() and source.is_file()):
            raise FileNotFoundError(
                f"{self.engine_name} source file not found: {source}"
            )

    def validate_source_blocking(self, source: Path, file_path: str) -> None:
        """Validate what cannot be checked without reading the source.

        Runs at the top of ``_extract_sync`` — inside the parser executor, and
        BEFORE the ``parsed_dir`` cleanup — so that:

        * a check whose cost scales with the file (opening the archive of a
          .docx with a huge central directory takes seconds) cannot stall the
          event loop and every unrelated request with it, and
        * a refusal has not already deleted the previous attempt's artifacts.

        Default is a no-op; engines that need a read-level gate override it.
        """
        return None

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
        determinism waiver so parse() can surface it as a parse warning,
        not just a log line.

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

    def finalize_parse_warnings(
        self,
        warnings: dict[str, Any],
        metadata: dict[str, Any],
        *,
        parsed_dir: Path,
        base_name: str,
        source: Path,
        i4_cache_disabled: bool,
    ) -> dict[str, Any] | None:
        """Map raw parser warnings to the doc_status ``parse_warnings`` field.

        Runs after ``extract`` (so it sees the full raw warnings dict). Engines
        may override to divert a subset of warnings to sidecar audit artifacts
        and return only the remainder for doc_status (see
        :class:`~lightrag.parser.docx.parser.NativeDocxParser`). The base stays
        engine-agnostic and simply surfaces everything via ``surface_warnings``.
        """
        return self.surface_warnings(warnings, source)

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
        from lightrag.parser.llm_bridge import (
            LLMBridgePipelineCancelled,
            LLMBridgeShutdown,
        )

        cancel_event = threading.Event()
        # Captured ONCE here, deliberately: _shutdown_parser_executor() sets the
        # current event and then replaces the attribute with a fresh, unset one,
        # so an in-flight parse must keep its reference to the event that was
        # live when it started. Re-reading ctx.rag._parser_shutdown_event later
        # would observe the replacement and miss the shutdown entirely.
        shutdown_event = getattr(ctx.rag, "_parser_shutdown_event", None)
        # Built once and shared by every consumer in the worker thread — the
        # LLM bridge and, since GHSA-25c3-j78v-83qx, the native markdown image
        # downloader. Before that the pipeline event reached the bridge only,
        # so a cancel was invisible to a document whose parse was stuck in a
        # network read.
        cancel_events = (
            cancel_event,
            (ctx.pipeline_cancel_event, LLMBridgePipelineCancelled),
            (shutdown_event, LLMBridgeShutdown),
        )
        llm_invoke = None
        smartheading_cache_keys: list = []
        i4_cache_disabled = False
        if self.wants_llm_bridge(engine_params):
            submit, smartheading_cache_keys, i4_cache_disabled = self._build_llm_submit(
                ctx
            )
            if submit is not None:
                from lightrag.parser.llm_bridge import SyncLLMBridge

                llm_invoke = SyncLLMBridge(
                    asyncio.get_running_loop(),
                    submit,
                    cancel_events=cancel_events,
                )
        runtime = NativeExtractRuntime(
            engine_params=engine_params,
            llm_invoke=llm_invoke,
            cancel_events=cancel_events,
        )

        rs = ctx.resolve(self.engine_name)
        source = rs.source_path
        self.validate_source(source, ctx.file_path)

        document_name = rs.document_name
        base_name = Path(document_name).stem or document_name
        parsed_dir = rs.parsed_dir
        asset_dir = parsed_dir / f"{base_name}.blocks.assets"

        # Whether _extract_sync got as far as replacing parsed_dir. Written in
        # the executor thread, read here after the await. It exists because the
        # two failure modes need opposite handling: once the directory is being
        # replaced a failure must roll it back, but a refusal that happens
        # BEFORE that must leave the previous attempt's sidecar alone.
        #
        # For normal completion the future's completion is the happens-before
        # edge, but a CANCELLED future completes while the worker thread is
        # still running — so the checkpoint's {read events, set flag} and the
        # coroutine's {set cancel_event, read flag} race. cleanup_lock makes
        # each of those a critical section: the coroutine either sees
        # cleanup_started before the worker commits to the rmtree (and rolls
        # back) or the worker sees the cancel event at the checkpoint (and
        # never starts). Both sections are pure in-memory work, so holding the
        # lock never blocks on I/O.
        cleanup_started = False
        cleanup_lock = threading.Lock()

        def _extract_sync():
            nonlocal cleanup_started
            # Read-level validation runs here, not in parse(): it opens the
            # source, and on the event loop that cost is paid by every other
            # request.
            self.validate_source_blocking(source, ctx.file_path)
            # Cancellation checkpoint, and the last moment one is useful.
            # Cancelling run_in_executor cancels only the future — THIS thread
            # keeps running, and everything below destroys parsed_dir and
            # rebuilds it. Ordinary extraction polls nothing, so without this
            # a cancel arriving during validation lets an abandoned worker
            # delete a complete sidecar from an earlier parse and leave a
            # partial one behind, after the coroutine has already returned.
            # The source is gone from INPUT_DIR by then, so that is not
            # recoverable.
            #
            # Raise LLMBridgePipelineCancelled, NOT asyncio.CancelledError:
            # the pipeline-level cancel (_watch_pipeline_cancellation) only
            # SETS ctx.pipeline_cancel_event, it never cancels the worker task,
            # so a CancelledError raised here would be set on a live, un-
            # cancelled future and re-raised into _parse_worker as a
            # BaseException that matches neither of its except clauses — the
            # doc would strand in PARSING and, if every worker died this way,
            # wedge the batch on q.join() with busy=True. LLMBridgePipelineCancelled
            # is the repo-wide "blocking parse wait was cancelled" signal that
            # _parse_worker catches to mark the doc cancelled. In the task-
            # cancel case the future is already cancelled, so awaiting it raises
            # CancelledError regardless of what the worker raised — one uniform
            # raise covers both.
            #
            # The rag-level shutdown event is checked here too, and raises the
            # distinct LLMBridgeShutdown. _shutdown_parser_executor() sets it
            # and then calls executor.shutdown(wait=False), i.e. it does NOT
            # wait for a running extract — its contract is that in-flight work
            # exits via the event. Without this branch a worker still inside
            # validate_source_blocking when finalize_storages() runs would walk
            # on to rmtree parsed_dir and extract into storages being torn down.
            # Shutdown stays a generic parse failure for audit (the parse worker
            # catches only PipelineCancelledException / LLMBridgePipelineCancelled
            # as "cancelled"), which is the documented intent.
            pipeline_cancelled = ctx.pipeline_cancel_event
            with cleanup_lock:
                if shutdown_event is not None and shutdown_event.is_set():
                    raise LLMBridgeShutdown(
                        f"parser executor shut down before extraction: {ctx.file_path}"
                    )
                if cancel_event.is_set() or (
                    pipeline_cancelled is not None and pipeline_cancelled.is_set()
                ):
                    raise LLMBridgePipelineCancelled(
                        f"parse cancelled before extraction: {ctx.file_path}"
                    )
                cleanup_started = True
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
        except BaseException:
            # Unblock a bridge poller promptly (idempotent), THEN roll back
            # the pre-created (possibly partial) dirs. The worker thread may
            # briefly outlive the rmtree; the pre-clean at the next parse
            # attempt sweeps any late writes.
            #
            # Only roll back what this attempt started building. A refusal from
            # validate_source_blocking lands here having touched nothing, and
            # deleting parsed_dir for it would destroy a COMPLETE sidecar from
            # an earlier successful parse — one a persisted sidecar_location
            # still points at, which the reuse path then cannot resolve.
            #
            # Take cleanup_lock around {set cancel_event, read cleanup_started}
            # so it interlocks with the worker's checkpoint (see cleanup_lock
            # above): on a cancelled future the worker is still running, and
            # without this the coroutine could read cleanup_started=False and
            # skip the rollback while the worker goes on to set the flag and
            # rmtree the prior complete sidecar. Snapshot the flag inside the
            # lock, do the I/O outside it.
            with cleanup_lock:
                cancel_event.set()
                should_roll_back = cleanup_started
            if should_roll_back and parsed_dir.exists():
                shutil.rmtree(parsed_dir, ignore_errors=True)
            raise
        if not blocks:
            if parsed_dir.exists():
                shutil.rmtree(parsed_dir, ignore_errors=True)
            raise ValueError(
                f"{self.empty_content_label} parser returned empty content "
                f"for {ctx.file_path}"
            )

        # Map raw warnings → doc_status parse_warnings. Runs after extract() so
        # the hook sees the complete warnings dict; docx overrides it to divert
        # its smart-heading diagnostics (incl. the I4 waiver, which the hook —
        # not the base — records) into the sidecar smart_audit.json instead.
        parse_warnings = self.finalize_parse_warnings(
            warnings,
            metadata,
            parsed_dir=parsed_dir,
            base_name=base_name,
            source=source,
            i4_cache_disabled=i4_cache_disabled,
        )
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
