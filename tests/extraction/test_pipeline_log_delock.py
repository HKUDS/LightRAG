"""Proof that the hot pipeline-status LOG sites in ``operate`` no longer acquire
``pipeline_status_lock``, while the KEEP cancellation-check blocks in the same
functions still do.

``extract_entities`` is the ideal witness: it contains BOTH a converted per-chunk
log site (was ``async with pipeline_status_lock`` around latest/append, now the
operation-scoped ``PipelineStatusLogger``) AND cancellation-check KEEP blocks
that legitimately take the lock. Because of the latter, a "raise-on-enter" lock
CANNOT be used to prove the log site is de-locked (it would blow up at the
cancel check) — so we use two complementary techniques, exactly as the design
requires:

- raise-on-enter lock → proves the cancel-check KEEP blocks still take the lock;
- inject a capture logger class (monkeypatch ``operate.PipelineStatusLogger``
  BEFORE calling ``extract_entities`` — the instance is constructed inside the
  function body, so patching the class in ``operate``'s namespace is the only
  seam) → proves the per-chunk log routes through the lock-free logger rather
  than the lock.
"""

from unittest.mock import AsyncMock

import pytest

import lightrag.operate as operate
from lightrag.operate import extract_entities
from lightrag.utils import Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline


class _DummyTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


_EXTRACTION_RESULT = (
    "(entity<|#|>TEST_ENTITY<|#|>CONCEPT<|#|>A test entity)<|COMPLETE|>"
)


def _make_global_config() -> dict:
    extract_func = AsyncMock(return_value=_EXTRACTION_RESULT)
    return {
        "llm_model_func": extract_func,
        "role_llm_funcs": {
            "extract": extract_func,
            "keyword": extract_func,
            "query": extract_func,
            "vlm": extract_func,
        },
        "entity_extract_max_gleaning": 0,
        "entity_extract_max_records": 100,
        "entity_extract_max_entities": 40,
        "addon_params": {},
        "tokenizer": Tokenizer("dummy", _DummyTokenizer()),
        "llm_model_max_async": 1,
    }


def _make_chunks() -> dict[str, dict]:
    content = "Test content."
    return {
        "chunk-001": {
            "tokens": len(content),
            "content": content,
            "full_doc_id": "doc-001",
            "chunk_order_index": 0,
        }
    }


class _TrackingLock:
    """Async lock double that counts acquisitions but never blocks."""

    def __init__(self):
        self.enter_count = 0

    async def __aenter__(self):
        self.enter_count += 1
        return self

    async def __aexit__(self, *exc):
        return False


class _RaiseOnEnterLock:
    async def __aenter__(self):
        raise AssertionError("pipeline_status_lock acquired")

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def _high_token_limit(monkeypatch):
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "999999")


async def test_cancel_check_keep_block_still_takes_the_lock(
    monkeypatch, _high_token_limit
):
    """raise-on-enter lock: the KEEP cancellation check (start of extraction) must
    still acquire the lock — so it raises. This is exactly why raise-on-enter alone
    cannot prove the LOG site is de-locked in a mixed function."""
    status = {"latest_message": "", "history_messages": []}
    with pytest.raises(AssertionError, match="pipeline_status_lock acquired"):
        await extract_entities(
            chunks=_make_chunks(),
            global_config=_make_global_config(),
            pipeline_status=status,
            pipeline_status_lock=_RaiseOnEnterLock(),
        )


async def test_per_chunk_log_routes_through_logger_not_the_lock(
    monkeypatch, _high_token_limit
):
    """Inject a capture logger class: the per-chunk extraction log must go
    through the lock-free ``PipelineStatusLogger`` (proving that hot site is
    de-locked), while the lock is still taken by the cancellation-check KEEP
    blocks.

    The class must be patched in ``operate``'s namespace (it is bound there by
    ``from ... import``, so patching ``lightrag.kg.shared_storage`` would not
    reach it) and BEFORE the call (the instance is constructed inside
    ``extract_entities``). Capture state lives in this test's closure so
    nothing leaks between tests."""
    captured: list[tuple] = []
    created: list[object] = []

    class _CaptureLogger:
        def __init__(self, pipeline_status):
            self.pipeline_status = pipeline_status
            created.append(self)

        def log(self, *messages):
            captured.append(messages)

        def append(self, *messages):
            captured.append(messages)

    monkeypatch.setattr(operate, "PipelineStatusLogger", _CaptureLogger)

    status = {"latest_message": "", "history_messages": []}
    lock = _TrackingLock()
    await extract_entities(
        chunks=_make_chunks(),
        global_config=_make_global_config(),
        pipeline_status=status,
        pipeline_status_lock=lock,
    )

    # The per-chunk "Chunk N of M extracted ..." log went through the logger,
    # and the logger was built for THIS operation's pipeline_status.
    flat = [m for group in captured for m in group]
    assert any("extracted" in m for m in flat), flat
    assert created and created[0].pipeline_status is status
    # The lock is taken EXACTLY by the two KEEP cancellation checks for a single
    # chunk (extraction-entry + per-chunk), and NOT by the log site. Re-wrapping
    # the per-chunk log back in ``async with pipeline_status_lock`` would push
    # this to 3 — so the exact count guards against that regression.
    assert lock.enter_count == 2


async def test_logs_written_even_without_pipeline_status_lock(_high_token_limit):
    """Pins semantic delta #1: the old per-chunk site required
    ``pipeline_status_lock`` and raised TypeError with ``None``
    (``async with None``); the logger writes regardless of the lock. The KEEP
    cancel checks skip themselves when the lock is None."""
    status = {"latest_message": "", "history_messages": []}
    await extract_entities(
        chunks=_make_chunks(),
        global_config=_make_global_config(),
        pipeline_status=status,
        pipeline_status_lock=None,
    )
    assert any("extracted" in m for m in status["history_messages"])
    assert "extracted" in status["latest_message"]
