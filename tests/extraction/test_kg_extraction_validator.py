"""Tests for the optional per-chunk extraction-quality hook (#3691).

``LightRAG.kg_extraction_validator`` is the last word on what the LLM
extracted from a chunk: whatever it drops never reaches
``merge_nodes_and_edges``, the vector stores, or any ``source_id`` chain.
``None`` (the default) must leave the pipeline byte-identical.
"""

from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface


class DummyTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


_CHUNK_CONTENTS = {
    "chunk-alpha": "Alpha content about Alice.",
    "chunk-bravo": "Bravo content about Bob.",
}


def _extraction_result(name: str) -> str:
    return f"(entity<|#|>{name}<|#|>CONCEPT<|#|>Description of {name})<|COMPLETE|>"


def _extraction_result_with_relation(name: str, other: str) -> str:
    """One entity record plus a relation to ``other``, which may or may not
    have an entity record of its own — both shapes occur in real output."""
    return (
        f"entity<|#|>{name}<|#|>CONCEPT<|#|>Description of {name}"
        f"\nrelation<|#|>{name}<|#|>{other}<|#|>related<|#|>{name} relates to {other}."
        "\n<|COMPLETE|>"
    )


async def _fake_extract(prompt: str, *args, **kwargs) -> str:
    for key, content in _CHUNK_CONTENTS.items():
        if content in prompt:
            return _extraction_result(key.split("-", 1)[1].upper())
    raise AssertionError(f"prompt carried no known chunk content: {prompt[:200]!r}")


def _make_chunks() -> dict[str, dict]:
    return {
        key: {
            "tokens": len(content),
            "content": content,
            "full_doc_id": "doc-001",
            "chunk_order_index": index,
            "file_path": f"{key}.md",
        }
        for index, (key, content) in enumerate(_CHUNK_CONTENTS.items())
    }


def _make_global_config(extract_func, validator=None, max_async: int = 3) -> dict:
    tokenizer = Tokenizer("dummy", DummyTokenizer())
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
        "tokenizer": tokenizer,
        "llm_model_max_async": max_async,
        "kg_extraction_validator": validator,
    }


def _collect_names(chunk_results) -> list[str]:
    return [name for maybe_nodes, _ in chunk_results for name in maybe_nodes]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_no_validator_leaves_results_unchanged():
    from lightrag.operate import extract_entities

    chunk_results = await extract_entities(
        chunks=_make_chunks(),
        global_config=_make_global_config(AsyncMock(side_effect=_fake_extract)),
    )
    assert sorted(_collect_names(chunk_results)) == ["ALPHA", "BRAVO"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_sync_validator_receives_chunk_key_text_and_shapes():
    from lightrag.operate import extract_entities

    seen: list[tuple[str, str, list[str], list[tuple[str, str]]]] = []

    def validator(chunk_key, chunk_text, maybe_nodes, maybe_edges):
        seen.append((chunk_key, chunk_text, list(maybe_nodes), list(maybe_edges)))
        return maybe_nodes, maybe_edges

    await extract_entities(
        chunks=_make_chunks(),
        global_config=_make_global_config(
            AsyncMock(side_effect=_fake_extract), validator=validator
        ),
    )

    assert sorted(item[0] for item in seen) == sorted(_CHUNK_CONTENTS)
    by_key = {item[0]: item for item in seen}
    assert by_key["chunk-alpha"][1] == _CHUNK_CONTENTS["chunk-alpha"]
    # The shapes the hook receives are exactly the merge inputs.
    assert by_key["chunk-alpha"][2] == ["ALPHA"]
    assert by_key["chunk-alpha"][3] == []


@pytest.mark.offline
@pytest.mark.asyncio
async def test_sync_validator_filters_entities():
    from lightrag.operate import extract_entities

    def drop_alpha(chunk_key, chunk_text, maybe_nodes, maybe_edges):
        maybe_nodes.pop("ALPHA", None)
        return maybe_nodes, maybe_edges

    chunk_results = await extract_entities(
        chunks=_make_chunks(),
        global_config=_make_global_config(
            AsyncMock(side_effect=_fake_extract), validator=drop_alpha
        ),
    )
    assert _collect_names(chunk_results) == ["BRAVO"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_async_validator_is_awaited():
    from lightrag.operate import extract_entities

    async def drop_everything(chunk_key, chunk_text, maybe_nodes, maybe_edges):
        return {}, {}

    chunk_results = await extract_entities(
        chunks=_make_chunks(),
        global_config=_make_global_config(
            AsyncMock(side_effect=_fake_extract), validator=drop_everything
        ),
    )
    assert _collect_names(chunk_results) == []
    # Empty results still produce one entry per chunk — the merge treats an
    # empty extraction as "nothing from this chunk", not a failure.
    assert len(chunk_results) == len(_CHUNK_CONTENTS)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_hook_runs_after_extraction_before_return():
    """The hook runs after extraction: a validator that inspects maybe_nodes
    sees fully merge-shaped records, ``source_id`` already stamped."""
    from lightrag.operate import extract_entities

    received: dict[str, dict] = {}

    def capture(chunk_key, chunk_text, maybe_nodes, maybe_edges):
        received[chunk_key] = {
            name: [record["source_id"] for record in records]
            for name, records in maybe_nodes.items()
        }
        return maybe_nodes, maybe_edges

    chunks = _make_chunks()
    await extract_entities(
        chunks=chunks,
        global_config=_make_global_config(
            AsyncMock(side_effect=_fake_extract), validator=capture
        ),
    )

    for chunk_key in chunks:
        for name, source_ids in received[chunk_key].items():
            assert source_ids == [chunk_key], (
                "the hook must see the merge-shaped records, source_id included"
            )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_hook_runs_before_multimodal_injection():
    """The hook filters LLM output only, and runs BEFORE the core-generated
    multimodal entity is injected.

    Two things would break if the order were reversed: the hook would be
    handed an entity it never extracted (a grounding rule deletes it), and an
    entity it rejects would come back through the injected
    ``(multimodal_entity, rejected_entity)`` edge when the merge upserts the
    edge's endpoints.
    """
    from lightrag.operate import extract_entities

    chunks = _make_chunks()
    chunks["chunk-alpha"]["sidecar"] = {"type": "table", "id": "table-1"}

    seen: dict[str, list[str]] = {}

    def drop_alpha(chunk_key, chunk_text, maybe_nodes, maybe_edges):
        seen[chunk_key] = sorted(maybe_nodes)
        maybe_nodes.pop("ALPHA", None)
        return maybe_nodes, maybe_edges

    chunk_results = await extract_entities(
        chunks=chunks,
        global_config=_make_global_config(
            AsyncMock(side_effect=_fake_extract), validator=drop_alpha
        ),
    )

    # The hook saw the LLM's entity, never the multimodal one.
    assert seen["chunk-alpha"] == ["ALPHA"]

    # chunk_results is order-aligned with chunks — see test_chunk_results_ordering.
    nodes_by_chunk = {}
    edges_by_chunk = {}
    for chunk_key, (maybe_nodes, maybe_edges) in zip(chunks, chunk_results):
        nodes_by_chunk[chunk_key] = sorted(maybe_nodes)
        edges_by_chunk[chunk_key] = sorted(maybe_edges)

    # The multimodal entity is still injected — the hook does not suppress it.
    assert nodes_by_chunk["chunk-alpha"] == ["table-1"]
    # ...and it carries no edge to the entity the hook rejected, so the merge
    # cannot resurrect ALPHA through an endpoint upsert.
    assert edges_by_chunk["chunk-alpha"] == []


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "returned",
    [
        None,
        ({"ALPHA": []},),
        {"maybe_nodes": {}, "maybe_edges": {}},
        ({}, {}, {}),
        ({}, ["not-a-dict"]),
    ],
    ids=["none", "one-tuple", "two-key-dict", "three-tuple", "non-dict-member"],
)
async def test_validator_returning_a_non_pair_raises_type_error(returned):
    """A malformed return value fails the chunk with a named TypeError.

    ``{"maybe_nodes": ..., "maybe_edges": ...}`` is the dangerous one: a bare
    unpack accepts it and silently binds the two KEY STRINGS, which only
    surfaces much later inside the merge as an unrelated-looking failure.
    """
    from lightrag.operate import extract_entities

    def bad_validator(chunk_key, chunk_text, maybe_nodes, maybe_edges):
        return returned

    with pytest.raises(TypeError, match="kg_extraction_validator must return"):
        await extract_entities(
            chunks=_make_chunks(),
            global_config=_make_global_config(
                AsyncMock(side_effect=_fake_extract), validator=bad_validator
            ),
        )


class _Auditor:
    """A stateful validator — the issue's own "write audit logs" use case."""

    def __init__(self):
        self.seen: list[str] = []

    def validate(self, chunk_key, chunk_text, maybe_nodes, maybe_edges):
        self.seen.append(chunk_key)
        return maybe_nodes, maybe_edges


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.full((len(texts), 32), 0.1, dtype=np.float32)


async def _mock_llm(prompt, **kwargs):
    return ""


def _new_rag(tmp_path: Path, validator) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"kgvalidator-{tmp_path.name}",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=32, max_token_size=4096, func=_mock_embedding
        ),
        tokenizer=Tokenizer("dummy", DummyTokenizer()),
        kg_extraction_validator=validator,
    )


@pytest.mark.offline
def test_stateful_validator_keeps_its_identity_through_global_config(tmp_path):
    """The wiring test the pure-``extract_entities`` cases cannot cover.

    ``_build_global_config`` starts from ``asdict(self)``, which deep-copies
    every non-dataclass field. A plain function is copied atomically and looks
    fine; a bound method is NOT — its ``__self__`` is deep-copied, so a
    stateful validator would filter against a throwaway copy per document and
    lose everything it collected, silently.
    """
    auditor = _Auditor()
    rag = _new_rag(tmp_path, auditor.validate)

    config = rag._build_global_config()
    assert config["kg_extraction_validator"].__self__ is auditor

    # Two builds (two documents) must not hand out two different objects.
    assert (
        rag._build_global_config()["kg_extraction_validator"].__self__
        is rag._build_global_config()["kg_extraction_validator"].__self__
    )


@pytest.mark.offline
def test_default_validator_is_none_in_global_config(tmp_path):
    rag = _new_rag(tmp_path, None)
    assert rag._build_global_config()["kg_extraction_validator"] is None


@pytest.mark.offline
@pytest.mark.asyncio
async def test_relations_incident_to_a_rejected_entity_are_pruned():
    """Dropping an entity but leaving its relation must not resurrect it.

    ``_merge_edges_then_upsert`` materializes a missing endpoint as an
    UNKNOWN-typed node, so an orphaned relation would put the rejected entity
    back into the graph, the entity vector store, and its source_id chain —
    defeating the hook entirely through the one mistake a validator is most
    likely to make.
    """
    from lightrag.operate import extract_entities

    async def extract(prompt: str, *args, **kwargs) -> str:
        # ALPHA has an entity record AND a relation; BRAVO only an entity.
        if _CHUNK_CONTENTS["chunk-alpha"] in prompt:
            return (
                "entity<|#|>ALPHA<|#|>CONCEPT<|#|>Description of ALPHA"
                "\nentity<|#|>BRAVO<|#|>CONCEPT<|#|>Description of BRAVO"
                "\nrelation<|#|>ALPHA<|#|>BRAVO<|#|>related<|#|>ALPHA relates to BRAVO."
                "\n<|COMPLETE|>"
            )
        return _extraction_result("BRAVO")

    def drop_alpha(chunk_key, chunk_text, maybe_nodes, maybe_edges):
        # Deliberately leaves the (ALPHA, BRAVO) relation behind.
        maybe_nodes.pop("ALPHA", None)
        return maybe_nodes, maybe_edges

    chunk_results = await extract_entities(
        chunks=_make_chunks(),
        global_config=_make_global_config(
            AsyncMock(side_effect=extract), validator=drop_alpha
        ),
    )

    surviving_edges = [key for _, maybe_edges in chunk_results for key in maybe_edges]
    assert all("ALPHA" not in key for key in surviving_edges), (
        f"a relation incident to the rejected entity survived: {surviving_edges}"
    )
    assert "ALPHA" not in _collect_names(chunk_results)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_relations_to_never_extracted_endpoints_are_kept():
    """The counterpart, and the reason the pruning is scoped to REMOVED names.

    An extracted relation whose endpoint has no entity record of its own is
    normal output, not an inconsistency — relation processing materializing
    that endpoint is designed behaviour (see ``collect_kg_merge_candidates``).
    Pruning every dangling edge instead of only the hook-rejected ones would
    silently delete legitimate relations for anyone with a hook installed,
    a pass-through validator included.
    """
    from lightrag.operate import extract_entities

    async def extract(prompt: str, *args, **kwargs) -> str:
        if _CHUNK_CONTENTS["chunk-alpha"] in prompt:
            # CHARLIE is an endpoint that was never extracted as an entity.
            return _extraction_result_with_relation("ALPHA", "CHARLIE")
        return _extraction_result("BRAVO")

    def passthrough(chunk_key, chunk_text, maybe_nodes, maybe_edges):
        return maybe_nodes, maybe_edges

    chunk_results = await extract_entities(
        chunks=_make_chunks(),
        global_config=_make_global_config(
            AsyncMock(side_effect=extract), validator=passthrough
        ),
    )

    surviving_edges = [key for _, maybe_edges in chunk_results for key in maybe_edges]
    assert ("ALPHA", "CHARLIE") in surviving_edges, (
        "a baseline relation to a never-extracted endpoint must not be pruned"
    )
