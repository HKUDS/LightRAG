"""Operator-visible reporting of token-limit truncation (issue #3601).

A truncated extraction response used to be visible only as a provider-level
WARNING carrying no document or chunk identity, so a run could index a short —
or empty — knowledge graph and still report success. The reporting contract
pinned here:

- every truncated response is logged with chunk + file identity (server log,
  unbounded, one line per event);
- ``pipeline_status`` gets the FIRST event immediately (a long run must not
  hide the condition until it ends) plus exactly ONE aggregate at the end of
  the stage — never one line per chunk, because ``history_messages`` is a
  bounded ring and a document whose budget is too small truncates on all of
  them;
- the counts reach the caller's document-scoped tally, which the pipeline
  stamps into ``doc_status.metadata``;
- a parse failure caused by truncation says so, instead of leaving the
  operator to correlate two unrelated-looking warnings by chunk id.

Both the initial extraction and the gleaning round are covered, with the
extraction cache both enabled and disabled: the cache-disabled branch is the
one that used to drop the marker entirely.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest

import lightrag.operate as operate
from lightrag.operate import extract_entities
from lightrag.utils import (
    Tokenizer,
    TokenizerInterface,
    TokenLimitTruncationTally,
    TruncatedResponse,
)

pytestmark = pytest.mark.offline


_EXTRACTION_RESULT = (
    "(entity<|#|>TEST_ENTITY<|#|>CONCEPT<|#|>A test entity)<|COMPLETE|>"
)


class _DummyTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


class _FakeCache:
    """Extraction cache double, so both cache branches can be exercised."""

    def __init__(self):
        self.global_config = {"enable_llm_cache_for_entity_extract": True}
        self._store = {}

    async def get_by_id(self, key):
        return self._store.get(key)

    async def upsert(self, entries):
        self._store.update(entries)


def _make_global_config(*, gleaning: int = 0) -> dict:
    extract_func = AsyncMock(return_value=_EXTRACTION_RESULT)
    return {
        "llm_model_func": extract_func,
        "role_llm_funcs": {
            "extract": extract_func,
            "keyword": extract_func,
            "query": extract_func,
            "vlm": extract_func,
        },
        "entity_extract_max_gleaning": gleaning,
        "entity_extract_max_records": 100,
        "entity_extract_max_entities": 40,
        "addon_params": {},
        "tokenizer": Tokenizer("dummy", _DummyTokenizer()),
        "llm_model_max_async": 1,
    }


def _make_chunks(count: int = 1, file_path: str = "reports/annual.md") -> dict:
    content = "Test content."
    return {
        f"chunk-{index:03d}": {
            "tokens": len(content),
            "content": content,
            "full_doc_id": "doc-001",
            "chunk_order_index": index,
            "file_path": file_path,
        }
        for index in range(1, count + 1)
    }


@pytest.fixture
def _high_token_limit(monkeypatch):
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "999999")


@pytest.fixture
def _propagate_lightrag_logger(monkeypatch):
    """``lightrag.utils.logger`` disables propagation; restore it so caplog
    can see WARNING records emitted from inside ``lightrag.operate``."""
    monkeypatch.setattr(logging.getLogger("lightrag"), "propagate", True)


@pytest.fixture
def status_messages(monkeypatch) -> list[str]:
    """Capture what extraction publishes to ``pipeline_status``.

    ``PipelineStatusLogger`` is instantiated inside ``extract_entities``, so
    patching the class in ``operate``'s namespace is the available seam.
    """
    captured: list[str] = []

    class _CaptureLogger:
        def __init__(self, pipeline_status):
            self.pipeline_status = pipeline_status

        def log(self, *messages):
            captured.extend(messages)

    monkeypatch.setattr(operate, "PipelineStatusLogger", _CaptureLogger)
    return captured


def _truncation_lines(messages: list[str]) -> list[str]:
    return [m for m in messages if "token-limit truncation" in m.lower()]


@pytest.mark.parametrize("cache_enabled", [False, True])
async def test_first_truncation_is_published_with_chunk_and_document_identity(
    _high_token_limit, status_messages, cache_enabled
):
    """The condition must be visible without reading the server log.

    Parametrized over the cache: with ``enable_llm_cache_for_entity_extract``
    off, extraction takes the branch that never inspected the marker at all
    (issue #3601 gap 3).
    """
    config = _make_global_config()
    config["role_llm_funcs"]["extract"].return_value = TruncatedResponse(
        _EXTRACTION_RESULT
    )

    tally = TokenLimitTruncationTally()
    await extract_entities(
        chunks=_make_chunks(),
        global_config=config,
        llm_response_cache=_FakeCache() if cache_enabled else None,
        truncation_tally=tally,
    )

    first, aggregate = _truncation_lines(status_messages)
    assert "chunk-001" in first
    assert "reports/annual.md" in first
    assert first.startswith("Warning:")
    assert aggregate.startswith("Warning:")
    assert "1 of 1 chunks" in aggregate
    assert tally.as_metadata()["stages"] == {"initial": 1}


async def test_extraction_cancelled_mid_wait_still_feeds_the_document_tally(
    _high_token_limit,
):
    """Codex review (PR #3607): both explicit publish call sites sat after
    the ``asyncio.wait`` on the chunk tasks, so an external cancellation
    landing on that wait bypassed them — a truncated response that chunk 1
    had already recorded never reached the caller's document tally, and
    ``ainsert_custom_chunks``'s failure bookkeeping persisted FAILED with no
    ``llm_truncation``. The hand-off now lives in a ``finally``."""
    import asyncio

    config = _make_global_config()  # llm_model_max_async=1 serializes chunks
    b_entered = asyncio.Event()
    release_b = asyncio.Event()
    calls = {"n": 0}

    async def _llm(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # Chunk 1: truncated but parseable — recorded, then the task
            # completes and releases the semaphore.
            return TruncatedResponse(_EXTRACTION_RESULT)
        # Chunk 2 starts only after chunk 1 fully finished (semaphore of 1);
        # park it so the parent is deterministically stuck on asyncio.wait.
        b_entered.set()
        await release_b.wait()
        return _EXTRACTION_RESULT

    config["role_llm_funcs"]["extract"].side_effect = _llm

    tally = TokenLimitTruncationTally()
    extract_task = asyncio.create_task(
        extract_entities(
            chunks=_make_chunks(2),
            global_config=config,
            truncation_tally=tally,
        )
    )
    await b_entered.wait()
    extract_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await extract_task
    # Unblock the orphaned chunk-2 task so it finishes cleanly.
    release_b.set()
    await asyncio.sleep(0)

    payload = tally.as_metadata()
    assert payload is not None, (
        "the cancellation on asyncio.wait discarded the truncation chunk 1 "
        "had already recorded — the FAILED bookkeeping would persist nothing"
    )
    assert payload["stages"] == {"initial": 1}
    assert payload["samples"] == ["chunk-001"]


async def test_gleaning_truncation_is_reported_and_counted(
    _high_token_limit, status_messages
):
    """Gleaning is a second LLM call and truncates independently of the first."""
    config = _make_global_config(gleaning=1)
    config["role_llm_funcs"]["extract"].return_value = TruncatedResponse(
        _EXTRACTION_RESULT
    )

    tally = TokenLimitTruncationTally()
    await extract_entities(
        chunks=_make_chunks(),
        global_config=config,
        truncation_tally=tally,
    )

    payload = tally.as_metadata()
    assert payload["stages"] == {"initial": 1, "gleaning": 1}
    # Two responses truncated, but only one chunk is affected.
    assert (payload["events"], payload["affected"]) == (2, 1)
    assert payload["samples"] == ["chunk-001"]

    aggregate = _truncation_lines(status_messages)[-1]
    assert "initial: 1, gleaning: 1" in aggregate


async def test_status_lines_do_not_grow_with_the_number_of_truncated_chunks(
    _high_token_limit, status_messages
):
    """The whole point of aggregating: a fully-truncated document must not
    flood the bounded ``history_messages`` ring with one line per chunk."""
    config = _make_global_config(gleaning=1)
    config["role_llm_funcs"]["extract"].return_value = TruncatedResponse(
        _EXTRACTION_RESULT
    )

    tally = TokenLimitTruncationTally()
    await extract_entities(
        chunks=_make_chunks(count=6),
        global_config=config,
        truncation_tally=tally,
    )

    lines = _truncation_lines(status_messages)
    assert len(lines) == 2, f"expected first + aggregate only, got {lines}"
    assert "6 of 6 chunks" in lines[-1]
    # 6 chunks x (initial + gleaning) all truncated.
    assert tally.events == 12
    assert tally.affected == 6


async def test_every_occurrence_still_reaches_the_server_log(
    _high_token_limit, status_messages, caplog, _propagate_lightrag_logger
):
    """Aggregation trims the status ring, not the log: the per-chunk detail
    an operator needs for diagnosis stays one line per event."""
    config = _make_global_config()
    config["role_llm_funcs"]["extract"].return_value = TruncatedResponse(
        _EXTRACTION_RESULT
    )

    with caplog.at_level("WARNING", logger="lightrag"):
        await extract_entities(chunks=_make_chunks(count=3), global_config=config)

    per_chunk = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Token-limit truncation during initial")
    ]
    assert len(per_chunk) == 3
    assert {"chunk-001", "chunk-002", "chunk-003"} == {
        message.split("chunk ")[1].split(" ")[0] for message in per_chunk
    }


async def test_a_clean_run_reports_nothing(_high_token_limit, status_messages):
    config = _make_global_config(gleaning=1)

    tally = TokenLimitTruncationTally()
    await extract_entities(
        chunks=_make_chunks(count=3),
        global_config=config,
        truncation_tally=tally,
    )

    assert _truncation_lines(status_messages) == []
    assert not tally
    assert tally.as_metadata_extra() == {}


async def test_truncation_is_reported_when_extraction_fails_midway(
    _high_token_limit, status_messages
):
    """A run that truncates and then dies must still hand up what it saw —
    truncation is frequently the reason the run failed."""
    calls = {"n": 0}

    async def _flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return TruncatedResponse(_EXTRACTION_RESULT)
        raise RuntimeError("provider exploded")

    config = _make_global_config()
    config["role_llm_funcs"]["extract"] = _flaky

    tally = TokenLimitTruncationTally()
    with pytest.raises(Exception, match="provider exploded"):
        await extract_entities(
            chunks=_make_chunks(count=2),
            global_config=config,
            truncation_tally=tally,
        )

    assert len(_truncation_lines(status_messages)) == 2
    assert tally.events == 1


async def test_extraction_reports_without_a_caller_supplied_tally(
    _high_token_limit, status_messages
):
    """``truncation_tally`` is optional: direct callers (tests, custom
    wiring) still get the status reporting."""
    config = _make_global_config()
    config["role_llm_funcs"]["extract"].return_value = TruncatedResponse(
        _EXTRACTION_RESULT
    )

    await extract_entities(chunks=_make_chunks(), global_config=config)

    assert len(_truncation_lines(status_messages)) == 2


class TestSummaryStage:
    """Description summaries are LLM calls too.

    A too-small output budget cuts them off exactly like extraction, and the
    partial text becomes the entity's/relation's persisted description with no
    trace anywhere. They belong in the same per-document record.
    """

    @staticmethod
    def _summary_config(summary_return) -> dict:
        return {
            "role_llm_funcs": {"extract": AsyncMock(return_value=summary_return)},
            "addon_params": {},
            "_resolved_summary_language": "English",
            "summary_length_recommended": 100,
            "summary_context_size": 10_000,
            "tokenizer": Tokenizer("dummy", _DummyTokenizer()),
        }

    async def test_truncated_summary_is_recorded_against_its_subject(self):
        """Also pins the check ORDER: ``sanitize_text_for_encoding`` rebuilds a
        plain str, so a check placed after it would record nothing."""
        tally = TokenLimitTruncationTally()

        summary = await operate._summarize_descriptions(
            "Entity",
            "TEST_ENTITY",
            ["first description", "second description"],
            self._summary_config(TruncatedResponse("partial summ")),
            llm_response_cache=None,
            truncation_tally=tally,
        )

        assert summary == "partial summ"
        assert tally.as_metadata()["stages"] == {"summary": 1}
        assert tally.as_metadata()["samples"] == ["Entity:TEST_ENTITY"]

    async def test_complete_summary_records_nothing(self):
        tally = TokenLimitTruncationTally()

        await operate._summarize_descriptions(
            "Entity",
            "TEST_ENTITY",
            ["first description", "second description"],
            self._summary_config("a complete summary"),
            llm_response_cache=None,
            truncation_tally=tally,
        )

        assert not tally

    async def _run_merge(self, graph, pipeline_status, document_tally):
        """Drive the whole merge chain: merge_nodes_and_edges →
        _merge_nodes_then_upsert → _handle_entity_relation_summary →
        _summarize_descriptions, with ALICE's summary truncated."""
        import asyncio

        from lightrag.kg.shared_storage import initialize_share_data
        from lightrag.operate import merge_nodes_and_edges

        initialize_share_data()

        config = self._summary_config(TruncatedResponse("partial summ"))
        config.update(
            {
                "summary_max_tokens": 1_000_000,
                # 2 descriptions and a threshold of 2 forces the LLM summary.
                "force_llm_summary_on_merge": 2,
                "source_ids_limit_method": operate.SOURCE_IDS_LIMIT_METHOD_KEEP,
                "max_source_ids_per_entity": 10_000,
                "max_source_ids_per_relation": 10_000,
                "max_file_paths": 100,
                "file_path_more_placeholder": "...",
            }
        )

        def _node(description: str, source: str) -> dict:
            return {
                "entity_name": "ALICE",
                "entity_type": "person",
                "description": description,
                "source_id": source,
                "file_path": "d.txt",
                "timestamp": 1,
            }

        chunk_results = [({"ALICE": [_node("first", "c1"), _node("second", "c2")]}, {})]

        await merge_nodes_and_edges(
            chunk_results,
            graph,
            _MemVdb(),
            _MemVdb(),
            config,
            full_entities_storage=_MemKV(),
            full_relations_storage=_MemKV(),
            doc_id="doc-001",
            pipeline_status=pipeline_status,
            pipeline_status_lock=asyncio.Lock(),
            entity_chunks_storage=_MemKV(),
            relation_chunks_storage=_MemKV(),
            truncation_tally=document_tally,
        )

    async def test_merge_publishes_one_aggregate_and_feeds_the_document_tally(self):
        pipeline_status = {"history_messages": [], "latest_message": ""}
        document_tally = TokenLimitTruncationTally()

        await self._run_merge(_MemGraph(), pipeline_status, document_tally)

        lines = _truncation_lines(pipeline_status["history_messages"])
        assert len(lines) == 1, f"expected one aggregate line, got {lines}"
        assert lines[0].startswith("Warning:")
        assert "1 description summary" in lines[0]
        assert document_tally.as_metadata()["stages"] == {"summary": 1}

    async def test_a_failing_merge_still_reports_what_it_truncated(self):
        """A truncation recorded before the merge dies must not be lost.

        The summary is produced (and recorded) in step 8 of the entity merge;
        the graph write that fails comes after. The exception then leaves
        merge_nodes_and_edges through wait_tasks_with_drain, skipping the tail
        — so without a failure-path publish the FAILED document's metadata
        would claim extraction was the only stage that truncated.
        """

        class _FailingGraph(_MemGraph):
            async def upsert_node(self, name, node_data):
                raise RuntimeError("graph write exploded")

        pipeline_status = {"history_messages": [], "latest_message": ""}
        document_tally = TokenLimitTruncationTally()

        with pytest.raises(Exception, match="graph write exploded"):
            await self._run_merge(_FailingGraph(), pipeline_status, document_tally)

        payload = document_tally.as_metadata()
        assert payload is not None, (
            "the merge recorded a truncated summary but the failure path "
            "never handed it to the document-scoped tally"
        )
        assert payload["stages"] == {"summary": 1}
        lines = _truncation_lines(pipeline_status["history_messages"])
        assert len(lines) == 1, f"expected one aggregate line, got {lines}"

    async def test_a_cancellation_between_phases_still_feeds_the_document_tally(
        self, monkeypatch
    ):
        """Codex review (PR #3607): the summary tally used to be absorbed
        only by the per-phase ``wait_tasks_with_drain`` wrappers and the
        success tail. A cancellation landing on an inter-phase await point
        (simulated here at the Phase 2 status-line write) escaped both — the
        document tally never learned of a summary that was already truncated
        AND merged into the graph, so the custom-chunk failure bookkeeping
        would recompute the journal from the incomplete tally and overwrite
        the write-ahead snapshot that did carry the event."""
        import asyncio

        pipeline_status = {"history_messages": [], "latest_message": ""}
        document_tally = TokenLimitTruncationTally()

        orig_append = operate.append_pipeline_history

        def cancel_at_phase2(status, message):
            if str(message).startswith("Phase 2"):
                raise asyncio.CancelledError()
            return orig_append(status, message)

        monkeypatch.setattr(operate, "append_pipeline_history", cancel_at_phase2)

        with pytest.raises(asyncio.CancelledError):
            await self._run_merge(_MemGraph(), pipeline_status, document_tally)

        payload = document_tally.as_metadata()
        assert payload is not None, (
            "the cancellation between phases discarded the truncated summary "
            "that phase 1 had already merged into the graph"
        )
        assert payload["stages"] == {"summary": 1}


class _MemGraph:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: dict = {}

    async def get_node(self, name):
        return self.nodes.get(name)

    async def has_node(self, name):
        return name in self.nodes

    async def upsert_node(self, name, node_data):
        self.nodes[name] = dict(node_data)

    async def has_edge(self, src, tgt):
        return (src, tgt) in self.edges

    async def get_edge(self, src, tgt):
        return self.edges.get((src, tgt))

    async def upsert_edge(self, src, tgt, edge_data):
        self.edges[(src, tgt)] = dict(edge_data)


class _MemVdb:
    async def upsert(self, data):
        pass

    async def delete(self, ids):
        pass


class _MemKV:
    def __init__(self):
        self.data: dict = {}

    async def get_by_id(self, key):
        return self.data.get(key)

    async def get_by_ids(self, keys):
        return [self.data.get(key) for key in keys]

    async def upsert(self, data):
        self.data.update(data)

    async def index_done_callback(self):
        pass


class TestParseFailureAttribution:
    """A failed parse must name truncation as the cause when it is the cause.

    ``TruncatedResponse`` survives ``remove_think_tags``, so the parsers can
    distinguish "the model ignored the output contract" from "the model was cut
    off" — only the second is fixed by raising the output budget.
    """

    async def test_text_mode_undelimited_result_names_truncation(
        self, caplog, _propagate_lightrag_logger
    ):
        with caplog.at_level("WARNING", logger="lightrag"):
            await operate._process_extraction_result(
                TruncatedResponse("(entity<|#|>A<|#|>C<|#|>d)"),
                "chunk-001",
                1,
                "d.txt",
                tuple_delimiter="<|#|>",
                completion_delimiter="<|COMPLETE|>",
            )

        messages = [
            record.getMessage()
            for record in caplog.records
            if "Complete delimiter can not be found" in record.getMessage()
        ]
        assert messages, "expected the undelimited-result warning"
        assert "truncated by the model's output token limit" in messages[0]

    async def test_text_mode_stays_silent_about_a_cause_it_cannot_prove(
        self, caplog, _propagate_lightrag_logger
    ):
        with caplog.at_level("WARNING", logger="lightrag"):
            await operate._process_extraction_result(
                "(entity<|#|>A<|#|>C<|#|>d)",
                "chunk-001",
                1,
                "d.txt",
                tuple_delimiter="<|#|>",
                completion_delimiter="<|COMPLETE|>",
            )

        messages = [
            record.getMessage()
            for record in caplog.records
            if "Complete delimiter can not be found" in record.getMessage()
        ]
        assert messages, "expected the undelimited-result warning"
        assert "truncated" not in messages[0]

    async def test_json_mode_unparseable_result_names_truncation(
        self, caplog, _propagate_lightrag_logger
    ):
        # Prose with no recoverable object at all: json_repair salvages a
        # half-written entity list, so a cut-off *preamble* is what actually
        # reaches the unparseable branch.
        with caplog.at_level("WARNING", logger="lightrag"):
            await operate._process_json_extraction_result(
                TruncatedResponse("Sure — here is the analysis of the"),
                "chunk-001",
                1,
                "d.txt",
            )

        messages = [
            record.getMessage()
            for record in caplog.records
            if "empty or unrecoverable" in record.getMessage()
        ]
        assert messages, "expected the unparseable-result warning"
        assert "truncated by the model's output token limit" in messages[0]
