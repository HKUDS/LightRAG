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

from tests.operate.test_kg_total_token_budget import (
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
    limit = _prompt_tokens(context, QueryParam(enable_rerank=False), _config(tokenizer))
    direct = _fit(entities, [], [], limit, tokenizer, None)
    memoized = _fit(entities, [], [], limit, tokenizer, {})
    assert [record["entity"] for record in direct[0]] == ["E0"]
    assert memoized == direct


def test_lengthened_description_is_reencoded_and_memo_never_drifts():
    """The memo keys on exact text: a changed record misses, and every stored
    count equals a fresh direct encode of the same text."""
    extra_chars = 17
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
    ("limit", "fits"),
    [
        pytest.param(1200, True, id="tight-many-probes"),
        pytest.param(1, False, id="exhausted"),
    ],
)
def test_graph_records_serialize_once_per_call_and_only_after_empty_probe(
    limit, fits, monkeypatch
):
    """Each graph record is serialized once per fitter call, however many
    probes render it, and not at all when the empty-context probe already
    exhausts the budget. Restoring per-probe dumps in render makes the tight
    case count probes x records; hoisting the serialization above the empty
    probe makes the exhausted case dump every record."""
    entities, relations = _graph_records()
    tokenizer = _CountingTokenizer()
    serialized = []
    real_dumps = json.dumps

    def counting_dumps(obj, *args, **kwargs):
        serialized.append(real_dumps(obj, *args, **kwargs))
        return serialized[-1]

    monkeypatch.setattr(operate.json, "dumps", counting_dumps)
    result = _fit(entities, relations, [], limit, tokenizer, {}, template=None)

    if not fits:
        assert result is None
        assert serialized == []
        return
    assert result is not None
    assert Counter(serialized) == Counter(
        real_dumps(record, ensure_ascii=False) for record in entities + relations
    )
    assert len(_prompt_texts(tokenizer.encoded)) > 2


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
