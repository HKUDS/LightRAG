"""A spelling the entity API accepted must address that entity.

``POST /graph/entity/create`` stores the extraction-normalized name, and its
response is the only place a client could learn that name. The other manual
entry points therefore resolve the caller's spelling the same way, or a client
re-using the spelling it just sent gets "does not exist". Entity deletion is
the deliberate exception: it is exact-only, so a retry can never fall through
to a different node, and its 404 names the stored spelling instead.

These run the REST routes against a real ``LightRAG`` (NetworkX graph, JSON KV,
nano vector DB), so the resolution is checked through the same stores and the
same admin write gate production uses. The unit-level cases, with fakes, live
in ``test_graph_entity_name_normalization`` and
``test_graph_relation_edit_delete_name_normalization``.
"""

import asyncio
import sys
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, Tokenizer, make_relation_chunk_key

pytestmark = pytest.mark.offline

_API_KEY = "fixture-key"
_HEADERS = {"X-API-Key": _API_KEY}

# Each rewritten by a different normalization rule.
_RAW_TO_STORED = [
    ("Tesla（US）", "Tesla(US)"),  # full-width parentheses
    ("Tesla&nbsp;Inc", "Tesla Inc"),  # HTML entity
    ('"Quoted"', "Quoted"),  # matched quotes
    ("Tesla　Motors", "Tesla Motors"),  # ideographic space
]


class _Characters:
    def encode(self, content):
        return [ord(c) for c in content]

    def decode(self, tokens):
        return "".join(chr(c) for c in tokens)


async def _embed(texts):
    return np.ones((len(texts), 8))


async def _no_llm(*args, **kwargs):
    raise AssertionError("Manual graph edits must not call an LLM")


@pytest.fixture
async def rag(tmp_path):
    instance = LightRAG(
        working_dir=str(tmp_path),
        workspace=f"names_{uuid4().hex}",
        llm_model_func=_no_llm,
        embedding_func=EmbeddingFunc(embedding_dim=8, func=_embed),
        tokenizer=Tokenizer("characters", _Characters()),
    )
    await instance.initialize_storages()
    try:
        yield instance
    finally:
        await instance.finalize_storages()


@pytest.fixture
async def client(rag, monkeypatch):
    # Router imports parse argv; do not hand them pytest's options.
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from lightrag.api.routers.graph_routes import create_graph_routes

    app = FastAPI()
    app.include_router(create_graph_routes(rag, api_key=_API_KEY))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as http:
        yield http


async def _create_entity(client, name):
    response = await client.post(
        "/graph/entity/create",
        json={"entity_name": name, "entity_data": {"description": "d"}},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text


async def _delete(client, resource, payload):
    return await client.request(
        "DELETE", f"/graph/{resource}/delete", json=payload, headers=_HEADERS
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("raw, stored", _RAW_TO_STORED)
async def test_raw_spelling_drives_the_whole_lifecycle(rag, client, raw, stored):
    graph = rag.chunk_entity_relation_graph
    await _create_entity(client, raw)
    await _create_entity(client, "SpaceX")
    assert await graph.has_node(stored)

    exists = await client.get(
        "/graph/entity/exists", params={"name": raw}, headers=_HEADERS
    )
    assert exists.json() == {"exists": True}

    created = await client.post(
        "/graph/relation/create",
        json={
            "source_entity": raw,
            "target_entity": "SpaceX",
            "relation_data": {"description": "rel"},
        },
        headers=_HEADERS,
    )
    assert created.status_code == 200, created.text
    assert await graph.has_edge(stored, "SpaceX")

    # Same edge through the other spelling: refused as a duplicate, not
    # written as a second relation.
    duplicate = await client.post(
        "/graph/relation/create",
        json={
            "source_entity": stored,
            "target_entity": "SpaceX",
            "relation_data": {"description": "rel"},
        },
        headers=_HEADERS,
    )
    assert duplicate.status_code == 400
    assert "already exists" in duplicate.json()["detail"]

    # Caller order reversed on edit: still the one stored edge.
    edited = await client.post(
        "/graph/relation/edit",
        json={
            "source_id": "SpaceX",
            "target_id": raw,
            "updated_data": {"description": "edited"},
        },
        headers=_HEADERS,
    )
    assert edited.status_code == 200, edited.text
    assert (await graph.get_edge(stored, "SpaceX"))["description"] == "edited"

    info = await rag.get_relation_info(raw, "SpaceX")
    assert info["graph_data"]["description"] == "edited"
    assert (await rag.get_entity_info(raw))["entity_name"] == stored

    deleted = await _delete(
        client, "relation", {"source_entity": raw, "target_entity": "SpaceX"}
    )
    assert deleted.status_code == 200, deleted.text
    assert not await graph.has_edge(stored, "SpaceX")
    key = make_relation_chunk_key(stored, "SpaceX")
    assert await rag.relation_chunks.get_by_id(key) is None

    # Entity deletion is exact-only: the raw spelling is refused with the
    # stored one named, and nothing is deleted until the caller uses it.
    refused = await _delete(client, "entity", {"entity_name": raw})
    assert refused.status_code == 404
    assert f"it is stored as '{stored}'" in refused.json()["detail"]
    assert await graph.has_node(stored)

    deleted = await _delete(client, "entity", {"entity_name": stored})
    assert deleted.status_code == 200, deleted.text
    assert not await graph.has_node(stored)


@pytest.mark.asyncio
async def test_exact_legacy_node_wins_over_its_normalized_spelling(rag, client):
    """A node stored under a raw spelling before manual names were normalized
    stays addressable by that spelling, even when the normalized one exists
    as a different node."""
    graph = rag.chunk_entity_relation_graph
    legacy = "Tesla（US）"
    await graph.upsert_node(
        legacy,
        {
            "entity_id": legacy,
            "entity_type": "ORG",
            "description": "legacy",
            "source_id": "",
            "file_path": "",
        },
    )
    await _create_entity(client, "Tesla(US)")
    await _create_entity(client, "SpaceX")

    created = await client.post(
        "/graph/relation/create",
        json={
            "source_entity": legacy,
            "target_entity": "SpaceX",
            "relation_data": {"description": "rel"},
        },
        headers=_HEADERS,
    )
    assert created.status_code == 200, created.text
    assert await graph.has_edge(legacy, "SpaceX")
    assert not await graph.has_edge("Tesla(US)", "SpaceX")

    deleted = await _delete(client, "entity", {"entity_name": legacy})
    assert deleted.status_code == 200, deleted.text
    assert not await graph.has_node(legacy)
    assert await graph.has_node("Tesla(US)")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spellings",
    [("Loop", "Loop"), ('"Loop"', "Loop")],
    ids=["identical", "two-spellings-of-one-node"],
)
async def test_stored_self_loop_is_deleted_without_hanging(rag, client, spellings):
    """Deleting is how a self-loop an older release stored gets removed.

    Identical endpoints used to lock ``sorted([name, name])``; the keyed lock
    then waited on its own mutex, and the request hung inside the admin write
    gate until the hold ceiling failed it.
    """
    graph = rag.chunk_entity_relation_graph
    await _create_entity(client, "Loop")
    await graph.upsert_edge(
        "Loop",
        "Loop",
        {"description": "self", "keywords": "", "source_id": "", "weight": 1.0},
    )
    await graph.index_done_callback()

    response = await asyncio.wait_for(
        _delete(
            client,
            "relation",
            {"source_entity": spellings[0], "target_entity": spellings[1]},
        ),
        timeout=30,
    )

    assert response.status_code == 200, response.text
    assert not await graph.has_edge("Loop", "Loop")


@pytest.mark.asyncio
async def test_two_spellings_of_one_node_are_refused_by_create_and_edit(rag, client):
    await _create_entity(client, "Loop")
    created = await client.post(
        "/graph/relation/create",
        json={
            "source_entity": '"Loop"',
            "target_entity": "Loop",
            "relation_data": {"description": "rel"},
        },
        headers=_HEADERS,
    )
    assert created.status_code == 400
    assert "self-loop" in created.json()["detail"]

    edited = await client.post(
        "/graph/relation/edit",
        json={
            "source_id": '"Loop"',
            "target_id": "Loop",
            "updated_data": {"description": "edited"},
        },
        headers=_HEADERS,
    )
    assert edited.status_code == 400
    assert "self-loop" in edited.json()["detail"]
    assert not await rag.chunk_entity_relation_graph.has_edge("Loop", "Loop")


@pytest.mark.asyncio
async def test_names_that_normalize_to_nothing_are_client_errors(rag, client):
    """A relation deletion refused on its arguments answers 400, not the 500 a
    storage failure gets. Entity deletion does not normalize, so for it the
    same name is simply one no node holds."""
    await _create_entity(client, "SpaceX")

    relation = await _delete(
        client, "relation", {"source_entity": "“”", "target_entity": "SpaceX"}
    )
    assert relation.status_code == 400
    assert "cannot be empty after normalization" in relation.json()["detail"]

    entity = await _delete(client, "entity", {"entity_name": "“”"})
    assert entity.status_code == 404

    assert await rag.chunk_entity_relation_graph.has_node("SpaceX")


@pytest.mark.asyncio
async def test_non_string_names_are_refused_without_raising(rag):
    """The core delete API reports through ``DeletionResult``; a non-string
    endpoint used to escape as ``TypeError`` from the ordering comparison."""
    relation = await rag.adelete_by_relation(None, "SpaceX")
    assert (relation.status, relation.status_code) == ("fail", 400)

    entity = await rag.adelete_by_entity(None)
    assert (entity.status, entity.status_code) == ("fail", 400)


@pytest.mark.asyncio
@pytest.mark.parametrize("first_attempt", ["fails-after-node-removal", "succeeds"])
async def test_repeated_entity_delete_never_reaches_another_entity(
    rag, monkeypatch, first_attempt
):
    """An exact legacy name and its normalized spelling can be two entities.

    A deletion of the legacy one that fails after the node removal committed
    is recovered by retrying the same request; a client that timed out on a
    successful one resends it. Either way the second call must not resolve the
    now-absent legacy name to the other, live entity and delete it too.
    """
    graph = rag.chunk_entity_relation_graph
    legacy = "Tesla（US）"
    await graph.upsert_node(
        legacy,
        {
            "entity_id": legacy,
            "entity_type": "ORG",
            "description": "legacy",
            "source_id": "",
            "file_path": "",
        },
    )
    await rag.entity_chunks.upsert({legacy: {"chunk_ids": ["chunk-1"], "count": 1}})
    await rag.acreate_entity("Tesla(US)", {"description": "a different entity"})

    if first_attempt == "fails-after-node-removal":

        async def _fail_tracking_cleanup(ids):
            raise RuntimeError("injected tracking failure")

        with monkeypatch.context() as patch:
            patch.setattr(rag.entity_chunks, "delete", _fail_tracking_cleanup)
            first = await rag.adelete_by_entity(legacy)
        assert (first.status, first.status_code) == ("fail", 500)
    else:
        first = await rag.adelete_by_entity(legacy)
        assert first.status == "success"
    assert not await graph.has_node(legacy)

    second = await rag.adelete_by_entity(legacy)

    assert (second.status, second.status_code) == ("not_found", 404)
    assert await graph.has_node("Tesla(US)")
    assert await rag.entity_chunks.get_by_id(legacy) is None
