"""GHSA-32jh-39m7-8x84: persisted ``sentence_split_regex`` must never be used.

Rejecting the field on ``/documents/text`` closes the ingress for NEW requests,
but ``chunk_options`` is snapshotted into ``full_docs`` at *enqueue* time —
before chunking runs. A build that still accepted the field therefore persisted
the attacker's pattern to storage, and the worker that froze on it left the
document in ``PROCESSING``, which is an auto-resume status. Without a scrub at
the processing trust boundary, upgrading would reload the stored pattern and
freeze the worker again on every restart: a boot loop no restart clears.

These tests pin the scrub itself (``apply_trusted_sentence_split_regex``), the
audit line it must leave when it discards a value, and — end to end through the
real ``"V"`` dispatch — the trust-boundary property it exists to guarantee.
"""

import asyncio
import logging
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.constants import DEFAULT_SENTENCE_SPLIT_REGEX
from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface, logger
from lightrag.utils_pipeline import apply_trusted_sentence_split_regex

pytestmark = pytest.mark.offline

# The advisory's payload: 40 `a`s partition 2**39 ways and the trailing anchor
# forces the engine to reject every one.
REDOS = "(a+)+$"


def _addon(regex: str | None) -> dict:
    sv: dict = {"buffer_size": 1}
    if regex is not None:
        sv["sentence_split_regex"] = regex
    return {"chunker": {"semantic_vector": sv}}


@contextmanager
def _capturable_lightrag_logger():
    # The lightrag logger sets propagate=False, so caplog's root handler never
    # sees its records. Re-enable it for the duration of the assertion.
    previous = logger.propagate
    logger.propagate = True
    try:
        yield
    finally:
        logger.propagate = previous


def test_poisoned_snapshot_regex_is_discarded():
    # The exact upgrade scenario: a snapshot written by a pre-fix build.
    poisoned = {"buffer_size": 1, "sentence_split_regex": REDOS}
    out = apply_trusted_sentence_split_regex(poisoned, _addon(r"(?<=[.?!])\s+"))
    assert out["sentence_split_regex"] == r"(?<=[.?!])\s+"
    assert REDOS not in out.values()
    # Unrelated snapshot params still win from the snapshot.
    assert out["buffer_size"] == 1


def test_scrub_does_not_mutate_the_snapshot():
    # The caller passes the persisted dict; mutating it in place would write the
    # scrubbed value back into the doc's stored options on the next persist.
    poisoned = {"sentence_split_regex": REDOS}
    apply_trusted_sentence_split_regex(poisoned, _addon(None))
    assert poisoned == {"sentence_split_regex": REDOS}


def test_live_env_regex_wins_over_snapshot_even_when_benign():
    # Not a "reject bad patterns" filter — the snapshot value is unconditionally
    # replaced, so provenance (API vs SDK) never has to be reconstructed.
    snapshot = {"sentence_split_regex": r"(?<=[.?!])\s+"}
    out = apply_trusted_sentence_split_regex(snapshot, _addon(r"(?<=[。！？])"))
    assert out["sentence_split_regex"] == r"(?<=[。！？])"


def test_missing_live_value_omits_the_key_entirely():
    # With nothing configured the key must be absent, so the chunker applies its
    # own default — never the persisted pattern.
    out = apply_trusted_sentence_split_regex(
        {"sentence_split_regex": REDOS}, _addon(None)
    )
    assert "sentence_split_regex" not in out


def test_empty_addon_params_falls_back_to_env_default():
    # A LightRAG instance whose addon_params were never populated must still get
    # the trusted default rather than the snapshot value.
    for addon in ({}, None, {"chunker": {}}):
        out = apply_trusted_sentence_split_regex({"sentence_split_regex": REDOS}, addon)
        assert out.get("sentence_split_regex") == DEFAULT_SENTENCE_SPLIT_REGEX


def test_discarding_a_snapshot_value_leaves_an_audit_line(caplog):
    # Dropping a persisted value is a decision with consequences — either an
    # attack was disarmed or an SDK caller's pattern was silently overridden.
    # Neither is discoverable from the "Chunking V: ..." line, which prints the
    # value AFTER replacement.
    with _capturable_lightrag_logger(), caplog.at_level(logging.WARNING, "lightrag"):
        apply_trusted_sentence_split_regex(
            {"sentence_split_regex": REDOS},
            _addon(r"(?<=[.?!])\s+"),
            doc_id="doc-poisoned",
        )
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("doc-poisoned" in m and REDOS in m for m in messages), (
        f"expected an audit line naming the doc and the discarded pattern; got {messages!r}"
    )


def test_no_audit_noise_when_the_snapshot_already_matches(caplog):
    # The common case (snapshot written by this build == live config) must stay
    # silent, or the warning is worthless as an attack signal.
    with _capturable_lightrag_logger(), caplog.at_level(logging.WARNING, "lightrag"):
        apply_trusted_sentence_split_regex(
            {"sentence_split_regex": r"(?<=[.?!])\s+"},
            _addon(r"(?<=[.?!])\s+"),
            doc_id="doc-clean",
        )
        # Absent from the snapshot is equally unremarkable.
        apply_trusted_sentence_split_regex({"buffer_size": 1}, _addon(r"(?<=[.?!])\s+"))
    assert [
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    ] == []


def test_audit_line_bounds_the_untrusted_pattern(caplog):
    # The discarded pattern is attacker-controlled text going into an operator's
    # log; it must be quoted (repr) and length-bounded.
    with _capturable_lightrag_logger(), caplog.at_level(logging.WARNING, "lightrag"):
        apply_trusted_sentence_split_regex(
            {"sentence_split_regex": "z" * 5000},
            _addon(r"(?<=[.?!])\s+"),
            doc_id="doc-long",
        )
    (message,) = [
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    ]
    assert "z" * 5000 not in message
    assert len(message) < 600


class _SimpleTokenizerImpl(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.full((len(texts), 32), 0.1, dtype=np.float32)


async def _mock_llm(prompt, **kwargs):
    return '{"name":"x","summary":"s","detail_description":"d"}'


def _new_rag(tmp_path: Path) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"regex-scrub-{tmp_path.name}",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=32, max_token_size=4096, func=_mock_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
    )


def test_v_dispatch_hands_the_chunker_the_trusted_regex(tmp_path, monkeypatch):
    """End-to-end wiring: a poisoned snapshot must not reach the chunker.

    Drives the real ``"V"`` branch of ``process_single_document`` with a
    caller-supplied ``chunk_options`` carrying the advisory payload — the same
    shape a pre-fix build persisted into ``full_docs``. The chunker itself is
    stubbed, so the assertion is on the kwargs it *received*: a wiring bug that
    computed the scrubbed dict but splatted the original would pass a
    source-level check and fail here.
    """
    monkeypatch.delenv("CHUNK_V_SENTENCE_SPLIT_REGEX", raising=False)

    import lightrag.chunker as chunker_pkg

    captured: dict = {}

    async def _v_spy(
        tokenizer, content, chunk_token_size, *, embedding_func=None, **kwargs
    ):
        captured["kwargs"] = dict(kwargs)
        return [{"tokens": 5, "content": "stub", "chunk_order_index": 0}]

    monkeypatch.setattr(chunker_pkg, "chunking_by_semantic_vector", _v_spy)

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "Poisoned body for the semantic-vector chunker.",
                ids=["doc-redos-snapshot"],
                file_paths="poisoned.[native-V].txt",
                track_id="track-v-scrub",
                process_options="V",
                chunk_options={
                    "semantic_vector": {
                        "buffer_size": 2,
                        "sentence_split_regex": REDOS,
                    }
                },
            )
            # The snapshot really does carry the payload — otherwise this test
            # would pass for the wrong reason.
            row = await rag.full_docs.get_by_id("doc-redos-snapshot")
            assert (
                row["chunk_options"]["semantic_vector"]["sentence_split_regex"] == REDOS
            )
            await rag.apipeline_process_enqueue_documents()
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())

    kwargs = captured.get("kwargs")
    assert kwargs is not None, "the V chunker was never dispatched"
    assert kwargs["sentence_split_regex"] == DEFAULT_SENTENCE_SPLIT_REGEX
    # Unrelated snapshot params still win from the snapshot.
    assert kwargs["buffer_size"] == 2
