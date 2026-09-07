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
async def test_core_does_not_prune_edges_for_the_validator():
    """Endpoint filtering belongs to the rule, and core must not fake it.

    A rule that deletes an entity but leaves a relation pointing at it undoes
    its own deletion — ``_merge_edges_then_upsert`` materializes the missing
    endpoint as an UNKNOWN-typed node. Core deliberately does not clean that
    up: the same omission also lets a name through in a chunk where it occurs
    only as an endpoint, which core cannot see, so covering one case and not
    the other would be less predictable than covering neither. This test pins
    the contract so nobody reintroduces a half guarantee.
    """
    from lightrag.operate import extract_entities

    async def extract(prompt: str, *args, **kwargs) -> str:
        if _CHUNK_CONTENTS["chunk-alpha"] in prompt:
            return (
                "entity<|#|>ALPHA<|#|>CONCEPT<|#|>Description of ALPHA"
                "\nentity<|#|>BRAVO<|#|>CONCEPT<|#|>Description of BRAVO"
                "\nrelation<|#|>ALPHA<|#|>BRAVO<|#|>related<|#|>ALPHA relates to BRAVO."
                "\n<|COMPLETE|>"
            )
        return _extraction_result("BRAVO")

    def nodes_only(chunk_key, chunk_text, maybe_nodes, maybe_edges):
        # The incomplete rule: entity deleted, its relation left behind.
        maybe_nodes.pop("ALPHA", None)
        return maybe_nodes, maybe_edges

    chunk_results = await extract_entities(
        chunks=_make_chunks(),
        global_config=_make_global_config(
            AsyncMock(side_effect=extract), validator=nodes_only
        ),
    )

    surviving_edges = [key for _, maybe_edges in chunk_results for key in maybe_edges]
    assert ("ALPHA", "BRAVO") in surviving_edges, (
        "core must pass the hook's output through untouched; filtering "
        "endpoints is the rule's job"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_a_name_keyed_rule_filters_nodes_and_endpoints_in_every_chunk():
    """The documented shape, and the reason it is the documented shape.

    ``is_junk`` is a function of the NAME, applied to entities and to relation
    endpoints. Chunk alpha has JUNK as a full entity plus a relation; chunk
    bravo has it ONLY as a relation endpoint — the cross-chunk case core
    cannot see, since JUNK never enters that chunk's ``maybe_nodes``. One rule
    handles both, with no cross-chunk state and no dependence on the order the
    concurrent chunk tasks happen to finish in.
    """
    from lightrag.operate import extract_entities

    async def extract(prompt: str, *args, **kwargs) -> str:
        if _CHUNK_CONTENTS["chunk-alpha"] in prompt:
            return (
                "entity<|#|>ALPHA<|#|>CONCEPT<|#|>Description of ALPHA"
                "\nentity<|#|>JUNK<|#|>CONCEPT<|#|>Description of JUNK"
                "\nrelation<|#|>ALPHA<|#|>JUNK<|#|>related<|#|>ALPHA relates to JUNK."
                "\n<|COMPLETE|>"
            )
        # Relation-only occurrence: no JUNK entity record in this chunk.
        return (
            "entity<|#|>BRAVO<|#|>CONCEPT<|#|>Description of BRAVO"
            "\nrelation<|#|>BRAVO<|#|>JUNK<|#|>related<|#|>BRAVO relates to JUNK."
            "\n<|COMPLETE|>"
        )

    def is_junk(name: str) -> bool:
        return name == "JUNK"

    def rule(chunk_key, chunk_text, maybe_nodes, maybe_edges):
        for name in [n for n in maybe_nodes if is_junk(n)]:
            del maybe_nodes[name]
        for key in [k for k in maybe_edges if is_junk(k[0]) or is_junk(k[1])]:
            del maybe_edges[key]
        return maybe_nodes, maybe_edges

    chunk_results = await extract_entities(
        chunks=_make_chunks(),
        global_config=_make_global_config(
            AsyncMock(side_effect=extract), validator=rule
        ),
    )

    assert "JUNK" not in _collect_names(chunk_results)
    surviving_edges = [key for _, maybe_edges in chunk_results for key in maybe_edges]
    assert all("JUNK" not in key for key in surviving_edges), (
        f"JUNK survived as a relation endpoint: {surviving_edges}"
    )
    # The rule is targeted, not a blanket purge: untouched output survives.
    assert sorted(_collect_names(chunk_results)) == ["ALPHA", "BRAVO"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_validator_receives_the_extraction_visible_text():
    """``chunk_text`` must be what the PROMPT carried, not the stored content.

    The stored chunk keeps parser-internal markup so query-time citations
    still resolve; extraction strips it. Handing the validator the stored
    form breaks grounding in both directions: a hidden id/path/refid would
    ground an entity the model never saw, and stripping a ``<cite>`` wrapper
    joins two words the stored form keeps apart, so a name the model DID see
    would fail the check.
    """
    from lightrag.chunk_schema import (
        strip_internal_multimodal_markup_for_extraction,
    )
    from lightrag.operate import extract_entities

    stored = (
        '<drawing id="im-7f3a9c" path="/data/figs/pump.png" src="pump.png" '
        'caption="Fig 1" />Alpha content about '
        '<cite type="table" refid="tb-91">Machine</cite> Learning.'
    )
    visible = strip_internal_multimodal_markup_for_extraction(stored)

    chunks = _make_chunks()
    chunks["chunk-alpha"]["content"] = stored

    async def extract(prompt: str, *args, **kwargs) -> str:
        if visible in prompt:
            return _extraction_result("ALPHA")
        if _CHUNK_CONTENTS["chunk-bravo"] in prompt:
            return _extraction_result("BRAVO")
        raise AssertionError(f"unexpected prompt: {prompt[:300]!r}")

    seen: dict[str, str] = {}

    def capture(chunk_key, chunk_text, maybe_nodes, maybe_edges):
        seen[chunk_key] = chunk_text
        return maybe_nodes, maybe_edges

    await extract_entities(
        chunks=chunks,
        global_config=_make_global_config(
            AsyncMock(side_effect=extract), validator=capture
        ),
    )

    assert seen["chunk-alpha"] == visible

    # The concrete consequences, spelled out on the text the hook received.
    for hidden in ("im-7f3a9c", "/data/figs/pump.png", "tb-91"):
        assert hidden in stored and hidden not in seen["chunk-alpha"], (
            f"{hidden} is parser-internal: grounding against it would accept "
            "an entity the model never saw"
        )
    assert "Machine Learning" not in stored
    assert "Machine Learning" in seen["chunk-alpha"], (
        "the stored form keeps the words apart, so grounding against it would "
        "reject a name the model plainly did see"
    )


class _FakeKV:
    """Minimal BaseKVStorage stand-in: enough for the cache-key attach path."""

    def __init__(self, rows: dict | None = None, global_config: dict | None = None):
        self.data = dict(rows or {})
        # The cache path reads these gates off the storage itself
        # (utils.py:4161-4164, :5299).
        self.global_config = global_config or {}

    async def get_by_id(self, key):
        return self.data.get(key)

    async def get_by_ids(self, keys):
        return [self.data.get(k) for k in keys]

    async def upsert(self, rows: dict):
        for key, value in rows.items():
            self.data[key] = value

    async def index_done_callback(self):
        return None


@pytest.mark.offline
@pytest.mark.asyncio
async def test_cache_keys_are_attached_before_the_validator_can_fail():
    """A failing validator must not orphan the LLM cache entry.

    ``save_to_cache`` writes each entry durably during extraction, but its key
    lives only in the in-memory collector until ``update_chunk_cache_list``
    attaches it to the chunk. Recovery collects operation-scoped cache ids
    exclusively from a staged chunk's ``llm_cache_list``
    (``_rollback_one_custom_chunk_patch``), so a key never attached is a cache
    row nothing can reach again — orphaned even after ``/documents/scan``
    rolls the operation back and deletes the chunk. The attach therefore has
    to happen before any step that user code can fail.
    """
    from lightrag.operate import extract_entities

    chunks = _make_chunks()
    text_chunks = _FakeKV({key: dict(value) for key, value in chunks.items()})
    llm_cache = _FakeKV(
        global_config={
            "enable_llm_cache": True,
            "enable_llm_cache_for_entity_extract": True,
        }
    )

    def explode(chunk_key, chunk_text, maybe_nodes, maybe_edges):
        raise RuntimeError("validator says no")

    config = _make_global_config(
        AsyncMock(side_effect=_fake_extract), validator=explode
    )
    config["enable_llm_cache_for_entity_extract"] = True

    with pytest.raises(RuntimeError, match="validator says no"):
        await extract_entities(
            chunks=chunks,
            global_config=config,
            llm_response_cache=llm_cache,
            text_chunks_storage=text_chunks,
        )

    attached = [
        key
        for row in text_chunks.data.values()
        for key in (row.get("llm_cache_list") or [])
    ]
    assert attached, (
        "the validator failed before the cache keys were attached; the cache "
        "rows written during extraction are now unreachable by rollback"
    )
    # Every attached key must name a cache row that actually exists.
    assert all(key in llm_cache.data for key in attached), (
        f"attached keys do not resolve: {attached} vs {sorted(llm_cache.data)}"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_validator_text_is_sanitized_like_the_provider_input():
    """``chunk_text`` must match what the provider received, sanitizer included.

    ``use_llm_func_with_cache`` runs the prompt through
    ``sanitize_text_for_encoding`` before calling the provider — control
    characters dropped, HTML entities unescaped, surrogates repaired. Handing
    the hook the unsanitized chunk would leave it comparing against a string
    the model never saw.
    """
    from lightrag.chunk_schema import (
        strip_internal_multimodal_markup_for_extraction,
    )
    from lightrag.operate import extract_entities
    from lightrag.utils import sanitize_text_for_encoding

    # Padded on both sides: the chunk is a FRAGMENT of the prompt, sitting
    # inside the fenced ---Input Text--- section, so its boundary whitespace is
    # interior to the prompt and stays model-visible. Sanitizing it in
    # isolation must therefore not trim it.
    stored = "\n\n  Alpha content about ALP\x01HA and AT&amp;T.  \n\n"
    visible = sanitize_text_for_encoding(
        strip_internal_multimodal_markup_for_extraction(stored), strip=False
    )
    assert visible != stored, "fixture must actually exercise the sanitizer"
    assert visible.startswith("\n\n  ") and visible.endswith("  \n\n"), (
        "boundary whitespace must survive: the model saw it inside the fence"
    )

    chunks = _make_chunks()
    chunks["chunk-alpha"]["content"] = stored

    async def extract(prompt: str, *args, **kwargs) -> str:
        if _CHUNK_CONTENTS["chunk-bravo"] in prompt:
            return _extraction_result("BRAVO")
        return _extraction_result("ALPHA")

    seen: dict[str, str] = {}

    def capture(chunk_key, chunk_text, maybe_nodes, maybe_edges):
        seen[chunk_key] = chunk_text
        return maybe_nodes, maybe_edges

    await extract_entities(
        chunks=chunks,
        global_config=_make_global_config(
            AsyncMock(side_effect=extract), validator=capture
        ),
    )

    assert seen["chunk-alpha"] == visible
    # The concrete difference, spelled out.
    assert "\x01" in stored and "\x01" not in seen["chunk-alpha"]
    assert "&amp;" in stored and "AT&T" in seen["chunk-alpha"]
    # ...and the whitespace the model saw is still there.
    assert seen["chunk-alpha"] != seen["chunk-alpha"].strip()
