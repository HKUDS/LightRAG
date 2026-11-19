"""Deletion rebuild re-extracts surviving chunks whose initial extraction is
missing, and never reports a failed rebuild as a successful deletion (RFC
#4491, changes 2 and 3).

Every case runs the real pipeline on the file-backed storages with a stub LLM
that answers from markers in the chunk text, so cache keys, ``llm_cache_list``
and the purge journal behave as in production.
"""

from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.kg.shared_storage import get_namespace_data
from lightrag.base import DocStatus
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id

pytestmark = pytest.mark.offline


class _Characters:
    def encode(self, content):
        return [ord(c) for c in content]

    def decode(self, tokens):
        return "".join(chr(c) for c in tokens)


async def _embed(texts):
    return np.ones((len(texts), 8))


# Paragraph marker -> records the stub returns for the initial extraction.
# V2OTHER names an unrelated entity so doc-v2 owns no Foo fact of its own.
INITIAL = {
    "SHARED": [("Foo", "Foo shared fact"), ("Bar", "Bar shared fact")],
    "V1ONLY": [("Foo", "Foo fact from v1")],
    "V2OTHER": [("Baz", "Baz fact from v2")],
    "V3ONLY": [("Foo", "Foo fact from v3"), ("Bar", "Bar fact from v3")],
    # Name Qux only as a relation endpoint, never as an entity record.
    "QUXA": [("Foo", "Foo fact from qa")],
    "QUXB": [("Foo", "Foo fact from qb")],
}
RELATION = "relation<|#|>Foo<|#|>Bar<|#|>linked<|#|>Foo links Bar in {marker}\n"

DOCS = {
    "doc-v1": "SHARED paragraph.\n\nV1ONLY paragraph.",
    "doc-v2": "SHARED paragraph.\n\nV2OTHER paragraph.",
    "doc-v3": "V3ONLY paragraph.",
    "doc-qa": "QUXA paragraph.",
    "doc-qb": "QUXB paragraph.",
}


class _StubLLM:
    """Answers extraction prompts from paragraph markers.

    ``relation_in_gleaning`` moves the Foo~Bar relation of SHARED / V3ONLY
    chunks from the initial answer into the gleaning answer.
    """

    def __init__(self, relation_in_gleaning: bool = False):
        self.relation_in_gleaning = relation_in_gleaning
        self.calls: list[tuple[str, bool]] = []
        self.fail = False
        # Name drift: a new extraction of SHARED calls Foo "Foo Prime".
        self.drift = False

    async def __call__(
        self, prompt, system_prompt=None, history_messages=None, **kwargs
    ):
        gleaning = bool(history_messages)
        source = history_messages[0]["content"] if gleaning else prompt
        for marker, records in INITIAL.items():
            if f"{marker} paragraph" not in source:
                continue
            self.calls.append((marker, gleaning))
            if self.fail:
                raise RuntimeError("LLM unavailable")
            has_relation = marker in ("SHARED", "V3ONLY")
            lines = []
            if not gleaning:
                lines = [
                    f"entity<|#|>{name}<|#|>concept<|#|>{desc}\n"
                    for name, desc in records
                ]
                if has_relation and not self.relation_in_gleaning:
                    lines.append(RELATION.format(marker=marker))
                if marker.startswith("QUX"):
                    lines.append(
                        f"relation<|#|>Foo<|#|>Qux<|#|>meets<|#|>Foo meets Qux in {marker}\n"
                    )
            elif has_relation and self.relation_in_gleaning:
                lines.append(RELATION.format(marker=marker))
            answer = "".join(lines) + "<|COMPLETE|>"
            if self.drift and marker == "SHARED":
                answer = answer.replace("<|#|>Foo<|#|>", "<|#|>Foo Prime<|#|>")
            return answer
        raise AssertionError(f"unexpected LLM prompt: {source[:200]!r}")


async def _make_rag(tmp_path, llm: _StubLLM, **kwargs) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path),
        workspace=f"reextract_{uuid4().hex}",
        llm_model_func=llm,
        embedding_func=EmbeddingFunc(embedding_dim=8, func=_embed),
        tokenizer=Tokenizer("characters", _Characters()),
        max_parallel_insert=1,
        **{"entity_extract_max_gleaning": 0, **kwargs},
    )
    await rag.initialize_storages()
    return rag


async def _insert(rag: LightRAG, *doc_ids: str) -> None:
    for doc_id in doc_ids:
        await rag.ainsert(
            DOCS[doc_id],
            split_by_character="\n\n",
            split_by_character_only=True,
            ids=[doc_id],
            file_paths=[f"{doc_id}.txt"],
        )


async def _chunks(rag: LightRAG, doc_id: str) -> dict[str, dict]:
    status = await rag.doc_status.get_by_id(doc_id)
    ids = list(status["chunks_list"])
    rows = await rag.text_chunks.get_by_ids(ids)
    return dict(zip(ids, rows))


async def _chunk_with(rag: LightRAG, doc_id: str, marker: str) -> tuple[str, dict]:
    for chunk_id, row in (await _chunks(rag, doc_id)).items():
        if marker in row["content"]:
            return chunk_id, row
    raise AssertionError(f"{doc_id} has no {marker} chunk")


def _ids(value: str) -> set[str]:
    return set(value.split(GRAPH_FIELD_SEP))


async def _delete_ok(rag: LightRAG, doc_id: str, **kwargs) -> None:
    result = await rag.adelete_by_doc_id(doc_id, **kwargs)
    assert result.status == "success", result.message
    assert await rag.doc_status.get_by_id(doc_id) is None


@pytest.mark.asyncio
async def test_rfc_reproduction_reextracts_shared_row_removed_by_cache_deletion(
    tmp_path,
):
    """The RFC's three-document reproduction: deleting v1 with
    ``delete_llm_cache=True`` removes the cache row v2's shared chunk relies
    on; the later deletion of v3 must re-extract that chunk rather than leave
    v3's text in place."""
    llm = _StubLLM()
    rag = await _make_rag(tmp_path, llm)
    try:
        await _insert(rag, "doc-v1", "doc-v2", "doc-v3")
        shared_id, shared_row = await _chunk_with(rag, "doc-v2", "SHARED")
        (shared_key,) = shared_row["llm_cache_list"]

        await _delete_ok(rag, "doc-v1", delete_llm_cache=True)
        assert await rag.llm_response_cache.get_by_id(shared_key) is None

        llm.calls.clear()
        await _delete_ok(rag, "doc-v3")
        assert llm.calls == [("SHARED", False)]

        graph = rag.chunk_entity_relation_graph
        foo = await graph.get_node("Foo")
        assert foo["description"] == "Foo shared fact"
        assert _ids(foo["source_id"]) == {shared_id}
        assert foo["file_path"] == "doc-v2.txt"
        edge = await graph.get_edge("Foo", "Bar")
        assert edge["description"] == "Foo links Bar in SHARED"
        assert _ids(edge["source_id"]) == {shared_id}
        relation_vdb = await rag.relationships_vdb.get_by_id(
            compute_mdhash_id("Bar" + "Foo", prefix="rel-")
        )
        assert "v3" not in relation_vdb["content"]

        # The re-extraction is cached again and referenced first, so a later
        # rebuild needs no LLM call.
        row = (await rag.text_chunks.get_by_ids([shared_id]))[0]
        assert row["llm_cache_list"] == [shared_key]
        assert (await rag.llm_response_cache.get_by_id(shared_key))["return"]

        await _delete_ok(rag, "doc-v2")
        assert await graph.get_node("Foo") is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_unparseable_initial_row_is_deleted_and_reextracted(tmp_path):
    llm = _StubLLM()
    rag = await _make_rag(tmp_path, llm)
    try:
        await _insert(rag, "doc-v2", "doc-v3")
        shared_id, shared_row = await _chunk_with(rag, "doc-v2", "SHARED")
        (shared_key,) = shared_row["llm_cache_list"]
        corrupt = await rag.llm_response_cache.get_by_id(shared_key)
        await rag.llm_response_cache.upsert({shared_key: {**corrupt, "return": None}})

        llm.calls.clear()
        await _delete_ok(rag, "doc-v3")
        assert llm.calls == [("SHARED", False)]
        assert (await rag.llm_response_cache.get_by_id(shared_key))[
            "return"
        ].startswith("entity<|#|>Foo")
        foo = await rag.chunk_entity_relation_graph.get_node("Foo")
        assert foo["description"] == "Foo shared fact"
        assert _ids(foo["source_id"]) == {shared_id}
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_lost_gleaning_row_is_accepted_and_its_relation_kept(tmp_path):
    """Only gleaning named Foo~Bar; with the survivor's gleaning row gone the
    initial row is still complete evidence, so nothing is re-extracted. The
    relation no surviving record names is kept with repaired provenance."""
    llm = _StubLLM(relation_in_gleaning=True)
    rag = await _make_rag(tmp_path, llm, entity_extract_max_gleaning=1)
    try:
        await _insert(rag, "doc-v2", "doc-v3")
        shared_id, shared_row = await _chunk_with(rag, "doc-v2", "SHARED")
        initial_key, gleaning_key = shared_row["llm_cache_list"]
        await rag.llm_response_cache.delete([gleaning_key])
        before = await rag.chunk_entity_relation_graph.get_edge("Foo", "Bar")

        llm.calls.clear()
        await _delete_ok(rag, "doc-v3")
        assert llm.calls == []

        edge = await rag.chunk_entity_relation_graph.get_edge("Foo", "Bar")
        assert edge["description"] == before["description"]
        assert _ids(edge["source_id"]) == {shared_id}
        assert edge["file_path"] == "doc-v2.txt"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_gleaning_row_first_in_cache_list_means_initial_is_missing(tmp_path):
    """A truncated initial response is never cached, so the chunk's first
    extract key can be a gleaning row. That is not an initial extraction."""
    llm = _StubLLM(relation_in_gleaning=True)
    rag = await _make_rag(tmp_path, llm, entity_extract_max_gleaning=1)
    try:
        await _insert(rag, "doc-v2", "doc-v3")
        shared_id, shared_row = await _chunk_with(rag, "doc-v2", "SHARED")
        initial_key, gleaning_key = shared_row["llm_cache_list"]
        await rag.text_chunks.upsert(
            {shared_id: {**shared_row, "llm_cache_list": [gleaning_key]}}
        )
        await rag.llm_response_cache.delete([initial_key])

        llm.calls.clear()
        await _delete_ok(rag, "doc-v3")
        # The new initial answer equals the old one, so the gleaning prompt is
        # unchanged and hits the surviving gleaning row instead of the LLM.
        assert llm.calls == [("SHARED", False)]
        row = (await rag.text_chunks.get_by_ids([shared_id]))[0]
        assert row["llm_cache_list"] == [initial_key, gleaning_key]
        foo = await rag.chunk_entity_relation_graph.get_node("Foo")
        assert foo["description"] == "Foo shared fact"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_disabled_extraction_cache_reextracts_on_every_rebuild(tmp_path):
    llm = _StubLLM()
    rag = await _make_rag(tmp_path, llm, enable_llm_cache_for_entity_extract=False)
    try:
        await _insert(rag, "doc-v1", "doc-v2", "doc-v3")
        shared_id, shared_row = await _chunk_with(rag, "doc-v2", "SHARED")
        assert shared_row.get("llm_cache_list", []) == []

        llm.calls.clear()
        await _delete_ok(rag, "doc-v3")
        # Every surviving Foo chunk is re-extracted, uncached.
        assert sorted(llm.calls) == [
            ("SHARED", False),
            ("SHARED", False),
            ("V1ONLY", False),
        ]
        llm.calls.clear()
        await _delete_ok(rag, "doc-v1")
        assert llm.calls == [("SHARED", False)]

        foo = await rag.chunk_entity_relation_graph.get_node("Foo")
        assert foo["description"] == "Foo shared fact"
        assert _ids(foo["source_id"]) == {shared_id}
        row = (await rag.text_chunks.get_by_ids([shared_id]))[0]
        assert row.get("llm_cache_list", []) == []
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
@pytest.mark.parametrize("keeps_source_id", [True, False], ids=["json", "pg-row"])
async def test_custom_kg_chunk_is_never_reextracted(tmp_path, keeps_source_id):
    """User-supplied custom KG text is not sent to the LLM on another
    document's deletion. Its entity is kept with repaired provenance; the
    description keeps the deleted document's text (documented residue)."""
    llm = _StubLLM()
    rag = await _make_rag(tmp_path, llm)
    try:
        await rag.ainsert_custom_kg(
            {
                "chunks": [
                    {
                        "content": "Curated note about Foo.",
                        "source_id": "curated",
                        "file_path": "curated.md",
                    }
                ],
                "entities": [
                    {
                        "entity_name": "Foo",
                        "entity_type": "concept",
                        "description": "Curated Foo",
                        "source_id": "curated",
                    }
                ],
                "relationships": [],
            }
        )
        await _insert(rag, "doc-v3")
        (curated_id,) = (await rag.entity_chunks.get_by_id("Foo"))["chunk_ids"][:1]
        if not keeps_source_id:
            # PGKVStorage stores only the pipeline's chunk columns, so a
            # custom KG chunk comes back without its source_id alias.
            (row,) = await rag.text_chunks.get_by_ids([curated_id])
            row = {k: v for k, v in row.items() if k != "source_id"}
            await rag.text_chunks.upsert({curated_id: row})
            (stored,) = await rag.text_chunks.get_by_ids([curated_id])
            assert "source_id" not in stored

        llm.calls.clear()
        await _delete_ok(rag, "doc-v3")
        assert llm.calls == []
        foo = await rag.chunk_entity_relation_graph.get_node("Foo")
        assert foo["description"].startswith("Curated Foo")
        assert _ids(foo["source_id"]) == {curated_id}
        assert foo["file_path"] == "curated.md"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_failed_reextraction_fails_the_deletion_and_a_retry_completes(
    tmp_path,
):
    llm = _StubLLM()
    rag = await _make_rag(tmp_path, llm)
    try:
        await _insert(rag, "doc-v1", "doc-v2", "doc-v3")
        await _delete_ok(rag, "doc-v1", delete_llm_cache=True)

        llm.fail = True
        result = await rag.adelete_by_doc_id("doc-v3")
        assert result.status == "fail"
        status = await rag.doc_status.get_by_id("doc-v3")
        assert status["status"] == DocStatus.DELETING
        assert (
            "Foo fact from v3"
            in (await rag.chunk_entity_relation_graph.get_node("Foo"))["description"]
        )

        llm.fail = False
        await _delete_ok(rag, "doc-v3")
        foo = await rag.chunk_entity_relation_graph.get_node("Foo")
        assert foo["description"] == "Foo shared fact"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_failed_relation_write_fails_the_deletion_and_a_retry_completes(
    tmp_path,
):
    """Change 3: a per-object rebuild failure under best_effort is raised, the
    document stays DELETING, and no entity is rebuilt over half-rebuilt
    relations. A retry completes and leaves correct graph and vector rows."""
    llm = _StubLLM()
    rag = await _make_rag(tmp_path, llm)
    try:
        await _insert(rag, "doc-v2", "doc-v3")
        shared_id, _ = await _chunk_with(rag, "doc-v2", "SHARED")
        graph = rag.chunk_entity_relation_graph
        original_upsert_edge = graph.upsert_edge
        original_upsert_node = graph.upsert_node
        node_writes = []

        async def failing_upsert_edge(*args, **kwargs):
            raise RuntimeError("edge write boom")

        async def recording_upsert_node(node_id, node_data):
            node_writes.append(node_id)
            return await original_upsert_node(node_id, node_data)

        graph.upsert_edge = failing_upsert_edge
        graph.upsert_node = recording_upsert_node
        result = await rag.adelete_by_doc_id("doc-v3")
        assert result.status == "fail"
        assert "edge write boom" in result.message
        assert node_writes == []
        status = await rag.doc_status.get_by_id("doc-v3")
        assert status["status"] == DocStatus.DELETING

        graph.upsert_edge = original_upsert_edge
        graph.upsert_node = original_upsert_node
        await _delete_ok(rag, "doc-v3")
        edge = await graph.get_edge("Foo", "Bar")
        assert edge["description"] == "Foo links Bar in SHARED"
        assert _ids(edge["source_id"]) == {shared_id}
        relation_vdb = await rag.relationships_vdb.get_by_id(
            compute_mdhash_id("Bar" + "Foo", prefix="rel-")
        )
        assert relation_vdb["content"].endswith("Foo links Bar in SHARED")
        assert _ids(relation_vdb["source_id"]) == {shared_id}
        entity_vdb = await rag.entities_vdb.get_by_id(
            compute_mdhash_id("Foo", prefix="ent-")
        )
        assert entity_vdb["content"] == "Foo\nFoo shared fact"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_name_drift_is_reported_and_recovered_by_reprocessing(tmp_path):
    """The operator recovery procedure for name drift, end to end: the
    re-extraction names Foo differently, Foo is kept with repaired provenance,
    the pipeline history names the surviving document, and deleting that
    document (keeping its cache) and ingesting it again merges the new name."""
    llm = _StubLLM()
    rag = await _make_rag(tmp_path, llm)
    try:
        await _insert(rag, "doc-v1", "doc-v2", "doc-v3")
        await _delete_ok(rag, "doc-v1", delete_llm_cache=True)
        shared_id, _ = await _chunk_with(rag, "doc-v2", "SHARED")

        llm.drift = True
        llm.calls.clear()
        await _delete_ok(rag, "doc-v3")
        assert llm.calls == [("SHARED", False)]
        graph = rag.chunk_entity_relation_graph
        foo = await graph.get_node("Foo")
        assert foo["description"] == "Foo shared fact<SEP>Foo fact from v3"
        assert _ids(foo["source_id"]) == {shared_id}
        assert foo["file_path"] == "doc-v2.txt"
        assert await graph.get_node("Foo Prime") is None

        status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
        history = list(status["history_messages"])
        assert any(
            m.startswith("Re-extracting 1 surviving chunk(s)") and "doc-v2" in m
            for m in history
        ), history
        assert any(
            m.startswith("Kept 1 entity(ies) and 1 relation(s)") for m in history
        )

        # Recovery: delete the surviving document, keeping its cache rows,
        # then ingest the same content again. The re-extracted row answers.
        await _delete_ok(rag, "doc-v2")
        llm.calls.clear()
        await _insert(rag, "doc-v2")
        assert llm.calls == []  # every chunk is answered from the cache
        assert await graph.get_node("Foo") is None
        foo_prime = await graph.get_node("Foo Prime")
        assert foo_prime["description"] == "Foo shared fact"
        assert (await graph.get_edge("Foo Prime", "Bar"))["description"] == (
            "Foo links Bar in SHARED"
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_entity_named_only_as_endpoint_is_resummarized_from_its_edges(
    tmp_path,
):
    """Keeping unattested objects must not stop an endpoint-only entity from
    being rebuilt: its surviving relation records still name it, so it is
    re-summarized from its rebuilt edges and loses the deleted text."""
    llm = _StubLLM()
    rag = await _make_rag(tmp_path, llm)
    try:
        await _insert(rag, "doc-qa", "doc-qb")
        before = await rag.chunk_entity_relation_graph.get_node("Qux")
        assert "QUXA" in before["description"]

        await _delete_ok(rag, "doc-qa")
        qux = await rag.chunk_entity_relation_graph.get_node("Qux")
        assert qux["description"] == "Foo meets Qux in QUXB"
        assert qux["file_path"] == "doc-qb.txt"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_retry_rebuilds_objects_whose_tracking_the_failed_attempt_narrowed(
    tmp_path,
):
    """The failed attempt already persisted the narrowed tracking rows, and a
    FIFO-capped graph ``source_id`` no longer names the deleted chunk, so a
    retry sees no reference to the deleted document. Resuming the purge must
    still rebuild the anchored objects instead of reporting success over the
    deleted document's text."""
    llm = _StubLLM()
    rag = await _make_rag(
        tmp_path,
        llm,
        source_ids_limit_method="FIFO",
        max_source_ids_per_entity=1,
        max_source_ids_per_relation=1,
    )
    try:
        await _insert(rag, "doc-v3", "doc-v2")
        shared_id, shared_row = await _chunk_with(rag, "doc-v2", "SHARED")
        (shared_key,) = shared_row["llm_cache_list"]
        graph = rag.chunk_entity_relation_graph
        # FIFO keeps the newest chunk only: the graph no longer names v3's.
        assert _ids((await graph.get_node("Foo"))["source_id"]) == {shared_id}
        await rag.llm_response_cache.delete([shared_key])

        llm.fail = True
        result = await rag.adelete_by_doc_id("doc-v3")
        assert result.status == "fail"
        assert "Foo fact from v3" in (await graph.get_node("Foo"))["description"]

        llm.fail = False
        llm.calls.clear()
        await _delete_ok(rag, "doc-v3")
        assert llm.calls == [("SHARED", False)]
        foo = await graph.get_node("Foo")
        assert foo["description"] == "Foo shared fact"
        edge = await graph.get_edge("Foo", "Bar")
        assert edge["description"] == "Foo links Bar in SHARED"
    finally:
        await rag.finalize_storages()


class _TaggingValidator:
    """Tags every entity description, so a rebuilt description shows whether
    the hook ran on the records it was rebuilt from. ``raise_next`` makes the
    next call fail once."""

    def __init__(self):
        self.calls: list[str] = []
        self.raise_next = False

    def __call__(self, chunk_key, chunk_text, maybe_nodes, maybe_edges):
        self.calls.append(chunk_text)
        if self.raise_next:
            self.raise_next = False
            raise RuntimeError("validator rejected the chunk")
        for records in maybe_nodes.values():
            for record in records:
                record["description"] = f"{record['description']} [checked]"
        return maybe_nodes, maybe_edges


@pytest.mark.asyncio
async def test_cached_records_go_through_the_extraction_validator(tmp_path):
    """A cache row holds the raw LLM output from before the hook ran, so the
    rebuild applies the hook to it, as ingestion did."""
    llm = _StubLLM()
    validator = _TaggingValidator()
    rag = await _make_rag(tmp_path, llm, kg_extraction_validator=validator)
    try:
        await _insert(rag, "doc-v2", "doc-v3")
        validator.calls.clear()
        llm.calls.clear()
        await _delete_ok(rag, "doc-v3")
        assert llm.calls == []
        assert any("SHARED paragraph" in text for text in validator.calls)
        foo = await rag.chunk_entity_relation_graph.get_node("Foo")
        assert foo["description"] == "Foo shared fact [checked]"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_validator_failure_during_reextraction_is_revalidated_on_retry(
    tmp_path,
):
    """The re-extraction caches the raw answer before the hook runs; when the
    hook then raises, the retry finds that row and must still apply the hook
    instead of rebuilding from the rejected records."""
    llm = _StubLLM()
    validator = _TaggingValidator()
    rag = await _make_rag(tmp_path, llm, kg_extraction_validator=validator)
    try:
        await _insert(rag, "doc-v2", "doc-v3")
        shared_id, shared_row = await _chunk_with(rag, "doc-v2", "SHARED")
        (shared_key,) = shared_row["llm_cache_list"]
        await rag.llm_response_cache.delete([shared_key])

        validator.raise_next = True
        result = await rag.adelete_by_doc_id("doc-v3")
        assert result.status == "fail"
        assert "validator rejected the chunk" in result.message
        # The raw re-extraction is cached despite the failed hook.
        assert await rag.llm_response_cache.get_by_id(shared_key) is not None

        validator.calls.clear()
        llm.calls.clear()
        await _delete_ok(rag, "doc-v3")
        assert llm.calls == []
        assert any("SHARED paragraph" in text for text in validator.calls)
        foo = await rag.chunk_entity_relation_graph.get_node("Foo")
        assert foo["description"] == "Foo shared fact [checked]"
    finally:
        await rag.finalize_storages()
