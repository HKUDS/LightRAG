"""The per-request token memo and one-shot graph serialization stay invisible.

The fitter's encode-verification contract (every returned candidate is
re-encoded, PR #3565) must survive memoization: a memo hit returns the exact
count of the exact text, so every fit decision, probe sequence and survivor
set is unchanged. All assertions are count-based; none measures wall time.
"""

import json
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightrag import operate
from lightrag.base import QueryParam
from lightrag.prompt import PROMPTS
from lightrag.utils import TiktokenTokenizer

from tests.llm.test_kg_total_token_budget import (
    QUERY,
    TEMPLATE,
    _config,
    _entity,
    _prompt_tokens,
    _relation,
    _search,
)

pytestmark = pytest.mark.offline

# The baseline measurement classifies prompt-sized encodes as texts over 1KB.
_PROMPT_CHARS = 1024


class _CountingTokenizer(TiktokenTokenizer):
    """Real tiktoken encoding that records every encode call's exact text."""

    def __init__(self):
        super().__init__()
        self.encoded: list[str] = []

    def encode(self, content):
        self.encoded.append(content)
        return super().encode(content)


def _context_entity(name, description="short evidence"):
    return {
        "entity": name,
        "type": "person",
        "description": description,
        "created_at": "2026-01-01 00:00:00",
        "file_path": f"{name}.md",
    }


def _context_relation(source, target, description="short relationship"):
    return {
        "entity1": source,
        "entity2": target,
        "description": description,
        "created_at": "2026-01-01 00:00:00",
        "file_path": f"{source}-{target}.md",
    }


def _graph_records():
    entities = [_context_entity(f"E{i}", f"d{i} " * 40) for i in range(6)]
    relations = [
        _context_relation(f"E{i}", f"E{i + 1}", f"r{i} " * 40) for i in range(6)
    ]
    return entities, relations


def _chunks():
    return [
        {
            "chunk_id": f"c{i}",
            "content": f"chunk evidence {i} " * 30,
            "file_path": f"c{i}.md",
        }
        for i in range(3)
    ]


def _fit(entities, relations, chunks, limit, tokenizer, memo, template=TEMPLATE):
    param = QueryParam(max_total_tokens=limit, enable_rerank=False)
    return operate._fit_kg_context_to_total_budget(
        entities,
        relations,
        chunks,
        QUERY,
        param,
        _config(tokenizer),
        template,
        memo,
    )


def _prompt_texts(texts):
    return [text for text in texts if len(text) > _PROMPT_CHARS]


def _matrix_records(shape):
    if shape == "few-short":
        return (
            [_context_entity("E0"), _context_entity("E1")],
            [_context_relation("E0", "E1"), _context_relation("E1", "E2")],
            [],
        )
    if shape == "many-long":
        return (
            [_context_entity(f"E{i}", f"evidence {i} " * 300) for i in range(8)],
            [
                _context_relation(f"E{i}", f"E{i + 1}", f"relation {i} " * 300)
                for i in range(8)
            ],
            [],
        )
    if shape == "oversized-entity":
        return (
            [_context_entity("Huge", "e" * 5000)],
            [
                _context_relation("E0", "E1"),
                _context_relation("E1", "E2"),
                _context_relation("E2", "E3"),
            ],
            [],
        )
    if shape == "oversized-relation":
        return (
            [_context_entity("E0"), _context_entity("E1"), _context_entity("E2")],
            [_context_relation("Huge", "Target", "r" * 5000)],
            [],
        )
    if shape == "few-short-chunks":
        return (
            [_context_entity("E0"), _context_entity("E1")],
            [_context_relation("E0", "E1"), _context_relation("E1", "E2")],
            [
                {
                    "chunk_id": f"c{i}",
                    "content": f"chunk {i} " * 100,
                    "file_path": f"c{i}.md",
                }
                for i in range(4)
            ],
        )
    return [], [], []


@pytest.mark.parametrize(
    ("limit", "shape"),
    [
        pytest.param(260, "few-short", id="tight-few-short"),
        pytest.param(2500, "many-long", id="tight-many-long"),
        pytest.param(100000, "few-short", id="loose-few-short"),
        pytest.param(100000, "many-long", id="loose-many-long"),
        pytest.param(200, "oversized-entity", id="tight-oversized-entity"),
        pytest.param(260, "oversized-relation", id="tight-oversized-relation"),
        pytest.param(116, "empty", id="empty-exact-required-budget"),
        pytest.param(650, "few-short-chunks", id="tight-chunk-walk-down"),
        pytest.param(1, "few-short", id="exhausted"),
    ],
)
def test_memoized_fit_is_identical_to_direct_fit(limit, shape):
    """One memo per request never changes a fit outcome (real tiktoken)."""
    entities, relations, chunks = _matrix_records(shape)
    tokenizer = TiktokenTokenizer()
    direct = _fit(entities, relations, chunks, limit, tokenizer, None)
    memoized = _fit(entities, relations, chunks, limit, tokenizer, {})
    assert memoized == direct

    inputs = (entities, relations, chunks)
    if limit == 1:
        assert memoized is None
    elif shape == "empty":
        assert memoized is not None
        assert memoized[:3] == ([], [], [])
    elif limit == 100000:
        assert tuple(len(part) for part in memoized[:3]) == tuple(
            len(part) for part in inputs
        )
    else:
        assert any(
            len(kept) < len(source) for kept, source in zip(memoized[:3], inputs)
        )
    if shape == "few-short-chunks":
        assert 0 < len(memoized[2]) < len(chunks)


def test_exact_boundary_hit_keeps_the_surviving_prefix():
    """A memo hit at the exact budget boundary returns the identical count.

    The limit equals the one-entity prompt's direct count, so the binary
    search's re-verification of the winning prefix is a memo hit on the
    boundary: a drifted hit count walks the survivor down to nothing.
    """
    entities = [_context_entity("E0"), _context_entity("E1")]
    tokenizer = TiktokenTokenizer()
    context = PROMPTS["kg_query_context"].format(
        entities_str=json.dumps(entities[0], ensure_ascii=False),
        relations_str="",
        text_chunks_str="",
        reference_list_str="",
    )
    limit = _prompt_tokens(
        context, QueryParam(enable_rerank=False), _config(tokenizer)
    )
    direct = _fit(entities, [], [], limit, tokenizer, None)
    memoized = _fit(entities, [], [], limit, tokenizer, {})
    assert [record["entity"] for record in direct[0]] == ["E0"]
    assert memoized == direct


def test_shared_memo_encodes_each_distinct_prompt_once_across_the_three_fits():
    """The pipeline's 3-call sequence re-encodes nothing the memo already saw.

    Measured on this scenario (default template, limit 1800): the stock
    sequence makes 25 prompt-sized encodes of 11 distinct prompts (14 repeats)
    plus 3 query encodes; the shared memo makes 11 + 1. The delta below is
    derived from the recorded texts, not hardcoded.
    """
    entities, relations = _graph_records()
    chunks = _chunks()

    def sequence(shared):
        tokenizer = _CountingTokenizer()
        memo = {} if shared else None
        first = _fit(entities, relations, [], 1800, tokenizer, memo, template=None)
        second = _fit(first[0], first[1], [], 1800, tokenizer, memo, template=None)
        third = _fit(second[0], second[1], chunks, 1800, tokenizer, memo, template=None)
        return tokenizer.encoded, (first, second, third)

    memo_texts, memo_results = sequence(True)
    stock_texts, stock_results = sequence(False)
    assert memo_results == stock_results

    memo_prompts = _prompt_texts(memo_texts)
    stock_prompts = _prompt_texts(stock_texts)
    repeats = len(stock_prompts) - len(set(stock_prompts))
    assert repeats > 0
    assert len(memo_prompts) == len(set(memo_prompts)) == len(set(stock_prompts))
    assert len(memo_texts) == len(set(stock_texts))
    query_repeats = stock_texts.count(QUERY) - memo_texts.count(QUERY)
    assert memo_texts.count(QUERY) == 1
    assert stock_texts.count(QUERY) == 3
    assert len(stock_texts) - len(memo_texts) == repeats + query_repeats


@pytest.mark.parametrize("extra_chars", [1, 17, 400])
def test_lengthened_description_is_reencoded_and_memo_never_drifts(extra_chars):
    """The memo keys on exact text: a changed record misses, and every stored
    count equals a fresh direct encode of the same text."""
    entities, relations = _graph_records()
    tokenizer = _CountingTokenizer()
    memo = {}
    _fit(entities, relations, [], 1800, tokenizer, memo, template=None)
    keys_before = set(memo)

    lengthened = [
        {**entities[0], "description": entities[0]["description"] + "x" * extra_chars},
        *entities[1:],
    ]
    encoded_before = len(tokenizer.encoded)
    memoized = _fit(lengthened, relations, [], 1800, tokenizer, memo, template=None)
    new_texts = tokenizer.encoded[encoded_before:]
    assert any("x" * extra_chars in text for text in new_texts)
    assert set(memo) - keys_before

    direct = _fit(
        lengthened, relations, [], 1800, _CountingTokenizer(), None, template=None
    )
    assert memoized == direct

    oracle = TiktokenTokenizer()
    drift = [
        (text[:40], count, len(oracle.encode(text)))
        for text, count in memo.items()
        if count != len(oracle.encode(text))
    ]
    assert drift == []


@pytest.mark.parametrize(
    ("limit", "expect_many_probes"),
    [
        pytest.param(1200, True, id="tight-eight-probes"),
        pytest.param(100000, False, id="loose-two-probes"),
    ],
)
def test_graph_records_serialize_once_per_call_regardless_of_probes(
    limit, expect_many_probes, monkeypatch
):
    """Pre-serialization replaced the stock per-probe json.dumps (one per
    record per render) with one per record per fitter call: the dumps count
    stays at the record count while the probe count varies. Reverting render
    to per-record dumps makes the tight case count probes x records."""
    entities, relations = _graph_records()
    tokenizer = _CountingTokenizer()
    serialized = []
    real_dumps = json.dumps

    def counting_dumps(obj, *args, **kwargs):
        serialized.append(real_dumps(obj, *args, **kwargs))
        return serialized[-1]

    monkeypatch.setattr(operate.json, "dumps", counting_dumps)
    result = _fit(entities, relations, [], limit, tokenizer, {}, template=None)

    assert result is not None
    assert len(serialized) == len(entities) + len(relations)
    assert Counter(serialized) == Counter(
        real_dumps(record, ensure_ascii=False) for record in entities + relations
    )
    probes = len(_prompt_texts(tokenizer.encoded))
    if expect_many_probes:
        assert probes > 2
    else:
        assert probes == 2


def test_exhausted_budget_serializes_no_records(monkeypatch):
    """The empty-context rejection path pays no record serialization: the
    one-shot dumps are gated on the same probe that gates the stock shape's
    first per-record dumps. Hoisting the serialization above the empty probe
    makes this test red with one dumps per record."""
    entities, relations = _graph_records()
    tokenizer = _CountingTokenizer()
    serialized = []
    real_dumps = json.dumps

    def counting_dumps(obj, *args, **kwargs):
        serialized.append(real_dumps(obj, *args, **kwargs))
        return serialized[-1]

    monkeypatch.setattr(operate.json, "dumps", counting_dumps)
    result = _fit(entities, relations, [], 1, tokenizer, {})

    assert result is None
    assert serialized == []


def test_memo_none_reencodes_repeated_prompts_within_one_call():
    """The unused path pays nothing new: with memo=None every probe encodes
    directly, including the binary search's re-verification of the winning
    prefix, exactly as before the memo existed."""
    entities, relations = _graph_records()
    stock_tokenizer = _CountingTokenizer()
    direct = _fit(entities, relations, [], 1800, stock_tokenizer, None, template=None)
    memo_tokenizer = _CountingTokenizer()
    memoized = _fit(entities, relations, [], 1800, memo_tokenizer, {}, template=None)
    assert memoized == direct

    stock_prompts = _prompt_texts(stock_tokenizer.encoded)
    memo_prompts = _prompt_texts(memo_tokenizer.encoded)
    repeats = len(stock_prompts) - len(set(stock_prompts))
    assert repeats > 0
    assert len(memo_prompts) == len(set(memo_prompts))
    assert len(stock_tokenizer.encoded) == len(memo_tokenizer.encoded) + repeats
    assert stock_tokenizer.encoded.count(QUERY) == 1


async def test_request_pipeline_shares_one_memo_across_its_three_fits(monkeypatch):
    """Wiring: _build_query_context creates one memo and threads it through
    stage 2 and both stage-4 fits. Dropping the memo on the way in reproduces
    the stock encode count; the wired run eliminates exactly the fitter's
    repeated texts and probes the same distinct prompts."""
    entities = [_entity(f"E{i}", f"d{i} " * 40) for i in range(6)]
    relations = [_relation(f"E{i}", f"E{i + 1}", f"r{i} " * 40) for i in range(6)]
    chunks = _chunks()
    real_fitter = operate._fit_kg_context_to_total_budget

    async def run(drop_memo):
        tokenizer = _CountingTokenizer()
        param = QueryParam(
            mode="mix",
            max_total_tokens=1800,
            max_entity_tokens=6000,
            max_relation_tokens=8000,
            enable_rerank=False,
        )
        fitter_texts = []

        def observed(*args):
            start = len(tokenizer.encoded)
            result = real_fitter(*args[:7]) if drop_memo else real_fitter(*args)
            fitter_texts.extend(tokenizer.encoded[start:])
            return result

        monkeypatch.setattr(
            operate,
            "_perform_kg_search",
            AsyncMock(return_value=_search(entities, relations, chunks)),
        )
        monkeypatch.setattr(
            operate, "_merge_all_chunks", AsyncMock(return_value=chunks)
        )
        monkeypatch.setattr(operate, "_fit_kg_context_to_total_budget", observed)
        result = await operate._build_query_context(
            QUERY,
            "Alice",
            "knows",
            None,
            None,
            None,
            SimpleNamespace(global_config=_config(tokenizer)),
            param,
        )
        return tokenizer.encoded, fitter_texts, result

    wired_texts, wired_fit, wired = await run(False)
    stock_texts, stock_fit, stock = await run(True)
    assert wired.context == stock.context
    assert wired.raw_data == stock.raw_data

    repeats = len(stock_fit) - len(set(stock_fit))
    assert repeats > 0
    assert len(wired_fit) == len(set(wired_fit))
    assert set(wired_fit) == set(stock_fit)
    assert len(wired_texts) == len(stock_texts) - repeats
