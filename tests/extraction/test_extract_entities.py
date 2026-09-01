"""Tests for entity extraction gleaning token limit guard."""

import json
import logging
import re
from unittest.mock import AsyncMock

import pytest

from lightrag.utils import Tokenizer, TokenizerInterface


@pytest.fixture
def _propagate_lightrag_logger(monkeypatch):
    """``lightrag.utils.logger`` sets ``propagate = False`` to avoid noisy
    test output; restore propagation locally so ``caplog`` can capture
    WARNING records emitted from inside ``lightrag.operate``."""
    monkeypatch.setattr(logging.getLogger("lightrag"), "propagate", True)


class DummyTokenizer(TokenizerInterface):
    """Simple 1:1 character-to-token mapping for testing."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def _make_global_config(
    entity_extract_max_gleaning: int = 1,
) -> dict:
    """Build a minimal global_config dict for extract_entities."""
    tokenizer = Tokenizer("dummy", DummyTokenizer())
    extract_func = AsyncMock(return_value="")
    return {
        "llm_model_func": extract_func,
        "role_llm_funcs": {
            "extract": extract_func,
            "keyword": extract_func,
            "query": extract_func,
            "vlm": extract_func,
        },
        "entity_extract_max_gleaning": entity_extract_max_gleaning,
        "entity_extract_max_records": 100,
        "entity_extract_max_entities": 40,
        "addon_params": {},
        "tokenizer": tokenizer,
        "llm_model_max_async": 1,
    }


# Minimal valid extraction result that _process_extraction_result can parse
_EXTRACTION_RESULT = (
    "(entity<|#|>TEST_ENTITY<|#|>CONCEPT<|#|>A test entity)<|COMPLETE|>"
)


def _make_chunks(content: str = "Test content.") -> dict[str, dict]:
    return {
        "chunk-001": {
            "tokens": len(content),
            "content": content,
            "full_doc_id": "doc-001",
            "chunk_order_index": 0,
        }
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_gleaning_skipped_when_tokens_exceed_limit(
    monkeypatch, caplog, _propagate_lightrag_logger
):
    """Gleaning must be skipped (with a WARNING) when the projected
    gleaning input — system + history(user+assistant) + continue prompt —
    exceeds ``MAX_EXTRACT_INPUT_TOKENS``.  This prevents
    ``context_length_exceeded`` errors from the LLM provider on the second
    round when the initial response was long.
    """
    from lightrag.operate import extract_entities

    # 10 tokens cannot fit any realistic prompt — guard must trip.
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "10")

    global_config = _make_global_config(entity_extract_max_gleaning=1)
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _EXTRACTION_RESULT

    with caplog.at_level("WARNING", logger="lightrag"):
        await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    # Only the initial extraction round ran; gleaning was skipped.
    assert llm_func.await_count == 1

    warnings_emitted = [
        rec.getMessage()
        for rec in caplog.records
        if rec.levelname == "WARNING"
        and rec.getMessage().startswith("Gleaning stopped for chunk chunk-001:")
    ]
    assert warnings_emitted, (
        "expected a WARNING log explaining gleaning was skipped due to "
        "token limit; got: "
        f"{[r.getMessage() for r in caplog.records]}"
    )
    # Message must surface both the measured token count and the limit so
    # operators can size MAX_EXTRACT_INPUT_TOKENS appropriately.
    msg = warnings_emitted[0]
    assert "exceeded limit (10)" in msg
    assert "Input tokens (" in msg


@pytest.mark.offline
@pytest.mark.asyncio
async def test_gleaning_proceeds_when_tokens_within_limit(monkeypatch):
    """Gleaning runs normally when the projected input fits the cap."""
    from lightrag.operate import extract_entities

    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "999999")

    global_config = _make_global_config(entity_extract_max_gleaning=1)
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _EXTRACTION_RESULT

    await extract_entities(
        chunks=_make_chunks(),
        global_config=global_config,
    )

    # Both rounds run: initial extraction + one gleaning pass.
    assert llm_func.await_count == 2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_no_gleaning_when_max_gleaning_zero(monkeypatch):
    """``entity_extract_max_gleaning=0`` disables gleaning regardless of
    token budget — the guard is downstream of the feature flag."""
    from lightrag.operate import extract_entities

    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "999999")

    global_config = _make_global_config(entity_extract_max_gleaning=0)
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _EXTRACTION_RESULT

    await extract_entities(
        chunks=_make_chunks(),
        global_config=global_config,
    )

    assert llm_func.await_count == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_gleaning_guard_disabled_when_max_tokens_zero(monkeypatch):
    """Setting ``MAX_EXTRACT_INPUT_TOKENS=0`` opts out of the guard so
    gleaning always runs regardless of input size — useful for callers
    whose provider has no hard input ceiling."""
    from lightrag.operate import extract_entities

    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "0")

    global_config = _make_global_config(entity_extract_max_gleaning=1)
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _EXTRACTION_RESULT

    await extract_entities(
        chunks=_make_chunks(),
        global_config=global_config,
    )

    # Guard disabled → gleaning still runs even with tight projected input.
    assert llm_func.await_count == 2


class CountingTokenizer(DummyTokenizer):
    """DummyTokenizer that records every string passed to encode()."""

    def __init__(self):
        self.encoded: list[str] = []

    def encode(self, content: str):
        self.encoded.append(content)
        return super().encode(content)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_gleaning_guard_encodes_system_prompt_once(monkeypatch):
    """The gleaning token guard must not re-encode the chunk-invariant
    system prompt (which embeds the multi-thousand-token examples block)
    for every chunk — it is constant for the whole extraction run, so it
    is encoded exactly once regardless of chunk count."""
    from lightrag.operate import extract_entities

    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "999999")

    inner = CountingTokenizer()
    global_config = _make_global_config(entity_extract_max_gleaning=1)
    global_config["tokenizer"] = Tokenizer("counting", inner)
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _EXTRACTION_RESULT

    chunks = {
        f"chunk-00{i}": {
            "tokens": 16,
            "content": f"Test content number {i}.",
            "full_doc_id": "doc-001",
            "chunk_order_index": i - 1,
        }
        for i in (1, 2, 3)
    }

    await extract_entities(
        chunks=chunks,
        global_config=global_config,
    )

    # All 3 chunks ran initial extraction + gleaning.
    assert llm_func.await_count == 6

    # The system prompt is the longest string fed to the tokenizer; it must
    # be encoded exactly once for the whole run, not once per chunk.
    longest = max(inner.encoded, key=len)
    assert sum(1 for s in inner.encoded if s == longest) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_gleaning_json_mode_binds_continue_prompt(monkeypatch):
    """JSON mode + gleaning must reach the gleaning LLM call: the closure's
    text-mode branch assigns entity_continue_extraction_user_prompt, which
    makes the name local to the whole closure (Python scoping is static), so
    the JSON branch must bind it explicitly. Regression guard for the
    UnboundLocalError that otherwise surfaced at the gleaning call."""
    from lightrag.operate import extract_entities

    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "999999")

    global_config = _make_global_config(entity_extract_max_gleaning=1)
    global_config["entity_extraction_use_json"] = True
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = "{}"

    # Must not raise UnboundLocalError.
    await extract_entities(
        chunks=_make_chunks(),
        global_config=global_config,
    )

    # Both rounds ran: initial extraction + one gleaning pass.
    assert llm_func.await_count == 2


# --- Additional coverage for the invariant-hoisting optimization ------------
#
# The tests above pin that the system prompt is encoded once per run (text
# mode) and that JSON mode reaches the gleaning call at all. The three below
# pin the two properties nothing else asserts: that hoisting MOVED the
# continue prompt's tokens into the invariant rather than dropping them from
# the guard, and that the precompute stays behind the guard's own condition.


def _make_counting_config(
    entity_extract_max_gleaning: int = 1,
    use_json: bool = False,
) -> tuple[dict, CountingTokenizer]:
    """global_config whose tokenizer records every encoded string."""
    inner = CountingTokenizer()
    global_config = _make_global_config(
        entity_extract_max_gleaning=entity_extract_max_gleaning
    )
    global_config["tokenizer"] = Tokenizer("counting", inner)
    global_config["entity_extraction_use_json"] = use_json
    return global_config, inner


def _make_multi_chunks(count: int = 3) -> dict[str, dict]:
    return {
        f"chunk-{i:03d}": {
            "tokens": 16,
            "content": f"Test content number {i}.",
            "full_doc_id": "doc-001",
            "chunk_order_index": i - 1,
        }
        for i in range(1, count + 1)
    }


# Minimal valid JSON-mode extraction result that
# _process_json_extraction_result can parse.
_JSON_EXTRACTION_RESULT = json.dumps(
    {
        "entities": [
            {
                "name": "TEST_ENTITY",
                "type": "CONCEPT",
                "description": "A test entity",
            }
        ],
        "relationships": [],
    }
)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_json_mode_encodes_both_invariants_once_per_run(monkeypatch):
    """In JSON mode BOTH hoisted strings are chunk-invariant — the system
    prompt and the continue prompt — so each is encoded once for the whole
    run regardless of chunk count."""
    from lightrag.operate import extract_entities
    from lightrag.prompt import PROMPTS

    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "999999")

    global_config, inner = _make_counting_config(
        entity_extract_max_gleaning=1, use_json=True
    )
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _JSON_EXTRACTION_RESULT

    await extract_entities(
        chunks=_make_multi_chunks(3),
        global_config=global_config,
    )

    # All 3 chunks ran initial extraction + gleaning.
    assert llm_func.await_count == 6

    for template_key in (
        "entity_extraction_json_system_prompt",
        "entity_continue_extraction_json_user_prompt",
    ):
        # Match on the template's leading literal text so the assertion does
        # not depend on how the prompt is rendered.
        prefix = PROMPTS[template_key].split("{", 1)[0]
        assert prefix, f"{template_key} must start with literal text"
        matches = [s for s in inner.encoded if s.startswith(prefix)]
        assert len(matches) == 1, (
            f"{template_key} was encoded {len(matches)} times for a 3-chunk "
            "run; it is chunk-invariant and must be encoded once"
        )


async def _projected_gleaning_tokens(
    monkeypatch, caplog, use_json: bool, continue_prompt_padding: str = ""
) -> int:
    """Run extraction under a cap of 10 tokens (which no realistic prompt
    fits) and return the token count the guard reported in its WARNING.
    ``continue_prompt_padding`` is appended verbatim to the mode's continue
    prompt template, so a caller can measure how the projection responds to a
    known change in that prompt's length."""
    from lightrag.operate import extract_entities
    from lightrag.prompt import PROMPTS

    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "10")
    continue_key = (
        "entity_continue_extraction_json_user_prompt"
        if use_json
        else "entity_continue_extraction_user_prompt"
    )
    if continue_prompt_padding:
        monkeypatch.setitem(
            PROMPTS,
            continue_key,
            PROMPTS[continue_key] + continue_prompt_padding,
        )

    global_config, _ = _make_counting_config(
        entity_extract_max_gleaning=1, use_json=use_json
    )
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _JSON_EXTRACTION_RESULT if use_json else _EXTRACTION_RESULT

    caplog.clear()
    with caplog.at_level("WARNING", logger="lightrag"):
        await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    # Only the initial extraction round ran; gleaning was skipped.
    assert llm_func.await_count == 1

    reported = [
        re.search(r"Input tokens \((\d+)\)", rec.getMessage())
        for rec in caplog.records
        if rec.getMessage().startswith("Gleaning stopped for chunk chunk-001:")
    ]
    assert reported and reported[0], (
        "expected a gleaning-stopped WARNING carrying the projected token "
        f"count; got: {[r.getMessage() for r in caplog.records]}"
    )
    return int(reported[0].group(1))


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize("use_json", [False, True], ids=["text_mode", "json_mode"])
async def test_guard_still_counts_the_continue_prompt(
    monkeypatch, caplog, _propagate_lightrag_logger, use_json
):
    """The guard must keep counting the continue prompt in both modes.

    Hoisting moved the JSON continue prompt's tokens from a per-chunk encode
    into the precomputed invariant, and left the text-mode one per-chunk.  If
    either were dropped rather than moved, the guard would under-count and
    let an oversized gleaning payload reach the provider.  Lengthening the
    continue prompt by a known amount must move the projection by exactly
    that amount (the test tokenizer maps one character to one token).
    """
    baseline = await _projected_gleaning_tokens(monkeypatch, caplog, use_json)
    padding = "X" * 64
    padded = await _projected_gleaning_tokens(
        monkeypatch, caplog, use_json, continue_prompt_padding=padding
    )

    assert padded - baseline == len(padding), (
        f"projected gleaning input moved by {padded - baseline} tokens when "
        f"the continue prompt grew by {len(padding)}; the guard is no longer "
        "counting the continue prompt"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_invariant_tokens_not_encoded_when_guard_inactive(monkeypatch):
    """Precomputing the invariant token count must stay behind the same
    condition that gates the guard, so a run that never consults it does not
    pay for the encode.  With gleaning off, the system prompt is never fed to
    the tokenizer."""
    from lightrag.operate import extract_entities
    from lightrag.prompt import PROMPTS

    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "999999")

    global_config, inner = _make_counting_config(entity_extract_max_gleaning=0)
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _EXTRACTION_RESULT

    await extract_entities(
        chunks=_make_chunks(),
        global_config=global_config,
    )

    assert llm_func.await_count == 1

    prefix = PROMPTS["entity_extraction_system_prompt"].split("{", 1)[0]
    assert not [s for s in inner.encoded if s.startswith(prefix)], (
        "system prompt was encoded even though the gleaning guard never runs"
    )
