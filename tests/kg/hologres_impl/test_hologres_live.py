import asyncio
from datetime import datetime, timedelta, timezone
import math
import uuid

import pytest

from lightrag.base import (
    CURSOR_END,
    CURSOR_START,
    CursorAfter,
    DocStatus,
    SourceAbsent,
    SourceConflict,
    SourceUnique,
)
from lightrag.kg.hologres.capabilities import (
    ProbeKind,
    ProbeStatus,
    probe_production_capabilities,
    run_initial_isolated_probes,
)
from lightrag.kg.hologres.client import quote_qualified_identifier
from lightrag.kg.hologres.doc_status import HologresDocStatusStorage
from lightrag.kg.hologres.kv import HologresKVStorage
from lightrag.kg.hologres.schema import (
    LEDGER_TABLE_NAME,
    HologresSchemaManager,
    SchemaDescriptor,
    SchemaState,
    doc_status_schema_descriptors,
    kv_schema_descriptors,
    vector_schema_descriptors,
)
from lightrag.kg.hologres.vector import (
    HologresVectorError,
    HologresVectorStorage,
)
from lightrag.namespace import NameSpace
from lightrag.utils import compute_mdhash_id


pytestmark = [pytest.mark.integration, pytest.mark.hologres_live]


async def test_initial_hologres_capabilities(hologres_live_client):
    client, schema = hologres_live_client

    production_report = await probe_production_capabilities(client)
    isolated_report = await run_initial_isolated_probes(client, schema)

    assert production_report.version.major >= 5
    assert isolated_report.supports(ProbeKind.SINGLE_AUTOCOMMIT_DDL)
    assert isolated_report.supports(ProbeKind.ASYNCPG_SETUP_RESET_BINDINGS)
    assert isolated_report.supports(ProbeKind.JSONB_ON_CONFLICT_ARRAYS_RECONNECT)
    hgraph_result = next(
        result
        for result in isolated_report.results
        if result.kind is ProbeKind.HGRAPH
    )
    assert hgraph_result.status is ProbeStatus.PASSED
    assert hgraph_result.detail_code == "hgraph_semantics_frozen"
    assert hgraph_result.evidence is not None
    assert [
        identifier
        for identifier, _raw_score in hgraph_result.evidence.ordered_raw_scores
    ] == [1, 4, 2, 3]
    observed_scores = dict(hgraph_result.evidence.ordered_raw_scores)
    for identifier, expected in {1: 1.0, 4: 1.0, 2: 0.0, 3: -1.0}.items():
        assert math.isfinite(observed_scores[identifier])
        assert abs(observed_scores[identifier] - expected) <= 1e-3
    assert hgraph_result.evidence.vector_filter_used is False


async def test_resumable_schema_management(hologres_live_client):
    client, schema = hologres_live_client
    ledger = quote_qualified_identifier(schema, LEDGER_TABLE_NAME)
    stale_table = "schema_stale_probe"
    concurrent_table = "schema_concurrent_probe"

    def descriptor(name, table):
        qualified = quote_qualified_identifier(schema, table)
        return SchemaDescriptor(
            name=name,
            component="live_schema",
            version=1,
            step=1 if table == stale_table else 2,
            sql=f"CREATE TABLE IF NOT EXISTS {qualified} (id text PRIMARY KEY)",
            postcondition_sql=(
                "SELECT EXISTS ("
                "SELECT 1 FROM pg_catalog.pg_class c "
                "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = $1 AND c.relname = $2 AND c.relkind = 'r')"
            ),
            postcondition_args=(schema, table),
            replay_safe=False,
        )

    class CrashAfterDdlClient:
        def __init__(self, delegate):
            self.delegate = delegate

        async def execute_one(self, sql, *values, **kwargs):
            result = await self.delegate.execute_one(sql, *values, **kwargs)
            if kwargs.get("descriptor") == "schema.descriptor.apply":
                raise asyncio.CancelledError("simulated crash after DDL")
            return result

        async def fetch_one(self, sql, *values, **kwargs):
            return await self.delegate.fetch_one(sql, *values, **kwargs)

        async def fetch_all(self, sql, *values, **kwargs):
            return await self.delegate.fetch_all(sql, *values, **kwargs)

        async def fetch_value(self, sql, *values, **kwargs):
            return await self.delegate.fetch_value(sql, *values, **kwargs)

    stale = descriptor("stale_probe", stale_table)
    concurrent = descriptor("concurrent_probe", concurrent_table)

    expired_now = datetime.now(timezone.utc) - timedelta(minutes=5)
    interrupted = HologresSchemaManager(
        CrashAfterDdlClient(client),
        schema=schema,
        owner_token="live-interrupted-owner",
        lease_seconds=1,
        now_provider=lambda: expired_now,
    )
    with pytest.raises(asyncio.CancelledError, match="simulated crash"):
        await interrupted.initialize([stale])

    recovered = HologresSchemaManager(
        client,
        schema=schema,
        owner_token="live-recovery-owner",
    )
    recovered_result = await recovered.initialize([stale])
    assert recovered_result[0].state is SchemaState.APPLIED
    assert recovered_result[0].resumed is True
    assert recovered_result[0].executed_ddl is False

    first = HologresSchemaManager(
        client, schema=schema, owner_token="live-concurrent-a"
    )
    second = HologresSchemaManager(
        client, schema=schema, owner_token="live-concurrent-b"
    )
    concurrent_results = await asyncio.gather(
        first.initialize([stale, concurrent]),
        second.initialize([stale, concurrent]),
    )
    assert all(
        result[-1].state is SchemaState.APPLIED for result in concurrent_results
    )
    assert sum(result[-1].executed_ddl for result in concurrent_results) == 1

    rows = await client.fetch_all(
        f"SELECT descriptor_name, state FROM {ledger} "
        "WHERE component = $1 ORDER BY step",
        "live_schema",
        descriptor="live.schema.ledger.read",
    )
    assert [(row["descriptor_name"], row["state"]) for row in rows] == [
        ("stale_probe", SchemaState.APPLIED.value),
        ("concurrent_probe", SchemaState.APPLIED.value),
    ]


async def test_hologres_kv_logical_partitions_and_crud(hologres_live_client):
    client, schema = hologres_live_client
    suffix = uuid.uuid4().hex
    workspace_a = f"lightrag_test_kv_a_{suffix}"
    workspace_b = f"lightrag_test_kv_b_{suffix}"

    def storage(namespace, workspace):
        return HologresKVStorage(
            namespace=namespace,
            workspace=workspace,
            global_config={},
            embedding_func=None,
            config=client.config,
            client=client,
        )

    primary = storage(NameSpace.KV_STORE_TEXT_CHUNKS, workspace_a)
    isolated = storage(NameSpace.KV_STORE_TEXT_CHUNKS, workspace_b)
    documents = storage(NameSpace.KV_STORE_FULL_DOCS, workspace_a)
    tracking = storage(NameSpace.KV_STORE_ENTITY_CHUNKS, workspace_a)
    entity_anchors = storage(NameSpace.KV_STORE_FULL_ENTITIES, workspace_a)
    relation_anchors = storage(NameSpace.KV_STORE_FULL_RELATIONS, workspace_a)
    storages = (
        primary,
        isolated,
        documents,
        tracking,
        entity_anchors,
        relation_anchors,
    )
    initialized = []

    try:
        for item in storages:
            await item.initialize()
            initialized.append(item)

        (descriptor,) = kv_schema_descriptors(schema)
        assert (
            await client.fetch_value(
                descriptor.postcondition_sql,
                *descriptor.postcondition_args,
                descriptor="live.kv.catalog",
            )
            is True
        )

        await primary.upsert({"shared": {"value": "a", "old": True}})
        await isolated.upsert({"shared": {"value": "b"}})
        assert await primary.get_by_id_strict("shared") == {
            "value": "a",
            "old": True,
        }
        assert await isolated.get_by_id_strict("shared") == {"value": "b"}

        await primary.upsert({"shared": {"value": "replaced"}, "empty": {}})
        assert await primary.get_by_id_strict("shared") == {"value": "replaced"}
        assert await primary.get_by_ids(["shared", "missing", "shared"]) == [
            {"value": "replaced"},
            None,
            {"value": "replaced"},
        ]
        assert await primary.filter_keys({"shared", "missing"}) == {"missing"}

        protected_a = {
            "sidecar_location": "sidecar-a",
            "parse_format": "markdown",
            "content_hash": "hash-a",
            "process_options": {"mode": "a"},
            "parse_engine": "native",
            "chunk_options": {"size": 100},
        }
        await documents.upsert(
            {
                "doc": {
                    "content": "old",
                    **protected_a,
                    "ordinary": "old",
                }
            }
        )
        await documents.upsert(
            {
                "doc": {
                    "content": "",
                    "sidecar_location": None,
                    "parse_format": "",
                    "content_hash": "",
                    "process_options": None,
                    "parse_engine": "",
                    "chunk_options": {},
                    "ordinary": "new",
                }
            }
        )
        assert await documents.get_by_id_strict("doc") == {
            "content": "",
            **protected_a,
            "ordinary": "new",
        }

        await documents.upsert({"doc": {"ordinary": "newer"}})
        assert await documents.get_by_id_strict("doc") == {
            "content": "",
            **protected_a,
            "ordinary": "newer",
        }

        protected_b = {
            "sidecar_location": "sidecar-b",
            "parse_format": "html",
            "content_hash": "hash-b",
            "process_options": {"mode": "b"},
            "parse_engine": "docling",
            "chunk_options": {"size": 200},
        }
        await documents.upsert({"doc": protected_b})
        assert await documents.get_by_id_strict("doc") == {
            "content": "",
            **protected_b,
            "ordinary": "newer",
        }

        await documents.upsert(
            {
                "new-doc": {
                    "content": "new",
                    "sidecar_location": None,
                    "parse_format": "",
                    "content_hash": "",
                    "process_options": None,
                    "parse_engine": "",
                    "chunk_options": {},
                    "ordinary": "inserted",
                }
            }
        )
        assert await documents.get_by_id_strict("new-doc") == {
            "content": "new",
            "ordinary": "inserted",
        }

        await tracking.upsert({"tracking": {"chunk_ids": []}})
        await entity_anchors.upsert(
            {"entity-anchor": {"entity_names": [], "metadata": {}}}
        )
        await relation_anchors.upsert(
            {"relation-anchor": {"relation_pairs": [], "metadata": {}}}
        )
        assert await tracking.get_by_id_strict("tracking") == {"chunk_ids": []}
        assert await entity_anchors.get_by_id_strict("entity-anchor") == {
            "entity_names": [],
            "metadata": {},
        }
        assert await relation_anchors.get_by_id_strict("relation-anchor") == {
            "relation_pairs": [],
            "metadata": {},
        }

        await primary.delete(["empty"])
        assert await primary.get_by_id_strict("empty") is None
        assert await primary.is_empty() is False
        await primary.drop()
        assert await primary.is_empty() is True
        assert await isolated.get_by_id_strict("shared") == {"value": "b"}
        assert await documents.get_by_id_strict("doc") is not None
    finally:
        for item in initialized:
            try:
                await item.drop()
            finally:
                await item.finalize()


async def test_hologres_doc_status_contract(hologres_live_client):
    client, schema = hologres_live_client
    suffix = uuid.uuid4().hex
    workspace_a = f"lightrag_test_doc_status_a_{suffix}"
    workspace_b = f"lightrag_test_doc_status_b_{suffix}"

    def storage(workspace):
        return HologresDocStatusStorage(
            namespace=NameSpace.DOC_STATUS,
            workspace=workspace,
            global_config={},
            embedding_func=None,
            config=client.config,
            client=client,
        )

    def document(
        *,
        status,
        created_at,
        file_path,
        content_hash=None,
        metadata=None,
        chunks_list=None,
    ):
        return {
            "content_summary": f"summary:{file_path}",
            "content_length": len(file_path),
            "file_path": file_path,
            "status": status,
            "created_at": created_at.isoformat(),
            "updated_at": created_at.isoformat(),
            "track_id": f"track:{file_path}",
            "chunks_count": 1,
            "chunks_list": chunks_list or [f"chunk:{file_path}"],
            "error_msg": None,
            "metadata": metadata or {},
            "multimodal_processed": True,
            "content_hash": content_hash,
            "producer_extension": {"source": "live-test"},
        }

    primary = storage(workspace_a)
    isolated = storage(workspace_b)
    initialized = []
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tied = start + timedelta(minutes=1)

    try:
        for item in (primary, isolated):
            await item.initialize()
            initialized.append(item)

        (descriptor,) = doc_status_schema_descriptors(schema)
        assert (
            await client.fetch_value(
                descriptor.postcondition_sql,
                *descriptor.postcondition_args,
                descriptor="live.doc_status.catalog",
            )
            is True
        )

        await primary.upsert(
            {
                "unique": document(
                    status=DocStatus.PENDING,
                    created_at=start,
                    file_path="unique.md",
                ),
                "tie-a": document(
                    status=DocStatus.ANALYZING,
                    created_at=tied,
                    file_path="tie-a.md",
                    chunks_list=["tie-a-1", "tie-a-2"],
                ),
                "tie-b": document(
                    status=DocStatus.ANALYZING,
                    created_at=tied,
                    file_path="tie-b.md",
                ),
                "conflict-a": document(
                    status=DocStatus.PARSING,
                    created_at=start + timedelta(minutes=2),
                    file_path="conflict.md",
                ),
                "conflict-b": document(
                    status=DocStatus.PARSING,
                    created_at=start + timedelta(minutes=3),
                    file_path="conflict.md",
                ),
                "hash-primary": document(
                    status=DocStatus.PROCESSED,
                    created_at=start + timedelta(minutes=4),
                    file_path="hash-primary.md",
                    content_hash="same-hash",
                ),
                "hash-pointer": document(
                    status=DocStatus.PROCESSED,
                    created_at=start + timedelta(minutes=5),
                    file_path="hash-pointer.md",
                    content_hash="same-hash",
                    metadata={
                        "is_duplicate": True,
                        "original_doc_id": "hash-primary",
                    },
                ),
                "hash-third": document(
                    status=DocStatus.PROCESSED,
                    created_at=start + timedelta(minutes=6),
                    file_path="hash-third.md",
                    content_hash="same-hash",
                ),
                "delete-me": document(
                    status=DocStatus.FAILED,
                    created_at=start + timedelta(minutes=7),
                    file_path="delete.md",
                ),
            }
        )
        await isolated.upsert(
            {
                "unique": document(
                    status=DocStatus.FAILED,
                    created_at=start,
                    file_path="isolated.md",
                )
            }
        )

        counts = await primary.get_status_counts()
        assert counts[DocStatus.PENDING.value] == 1
        assert counts[DocStatus.ANALYZING.value] == 2
        assert counts[DocStatus.PARSING.value] == 2
        assert counts[DocStatus.PROCESSED.value] == 3
        assert counts[DocStatus.FAILED.value] == 1
        assert (await primary.get_all_status_counts())["all"] == 9
        assert (await isolated.get_status_counts())[DocStatus.FAILED.value] == 1

        first = await primary.get_docs_by_statuses_page(
            [DocStatus.ANALYZING], limit=1, position=CURSOR_START, strict=True
        )
        assert list(first.docs) == ["tie-a"]
        assert isinstance(first.next_position, CursorAfter)
        second = await primary.get_docs_by_statuses_page(
            [DocStatus.ANALYZING],
            limit=1,
            position=first.next_position,
            strict=True,
        )
        assert list(second.docs) == ["tie-b"]
        assert isinstance(second.next_position, CursorAfter)
        exhausted = await primary.get_docs_by_statuses_page(
            [DocStatus.ANALYZING],
            limit=1,
            position=second.next_position,
            strict=True,
        )
        assert exhausted.docs == {}
        assert exhausted.next_position is CURSOR_END

        scheduling = await primary.get_docs_by_ids(
            ["tie-b", "missing", "tie-a", "tie-b"], strict=True
        )
        assert list(scheduling) == ["tie-b", "tie-a"]
        assert scheduling["tie-a"].status is DocStatus.ANALYZING
        hydrated = await primary.get_full_docs_by_ids(
            ["tie-a", "missing"], strict=True
        )
        assert hydrated["tie-a"].chunks_list == ["tie-a-1", "tie-a-2"]
        assert "missing" not in hydrated

        holder = await primary.get_doc_by_content_hash("same-hash")
        assert holder is not None and holder[0] == "hash-primary"
        excluded = await primary.get_doc_by_content_hash(
            "same-hash", exclude_doc_id="hash-primary"
        )
        assert excluded is not None and excluded[0] == "hash-third"

        assert isinstance(
            await primary.resolve_doc_source_strict("absent.md"), SourceAbsent
        )
        unique = await primary.resolve_doc_source_strict("unique.md")
        assert isinstance(unique, SourceUnique)
        assert unique.doc_id == "unique"
        conflict = await primary.resolve_doc_source_strict("conflict.md")
        assert isinstance(conflict, SourceConflict)
        assert conflict.candidate_count == 2
        assert conflict.sample_doc_ids == ("conflict-a", "conflict-b")

        conflicts = await primary.list_source_conflicts_page(
            limit=1, position=CURSOR_START
        )
        assert [entry.canonical_source_key for entry in conflicts.conflicts] == [
            "conflict.md"
        ]
        assert isinstance(conflicts.next_position, CursorAfter)
        no_more_conflicts = await primary.list_source_conflicts_page(
            limit=1, position=conflicts.next_position
        )
        assert no_more_conflicts.conflicts == ()
        assert no_more_conflicts.next_position is CURSOR_END

        dry_run = await primary.repair_source_conflict(
            "conflict.md",
            primary_doc_id="conflict-a",
            expected_candidate_count=0,
            expected_candidate_fingerprint="",
            dry_run=True,
        )
        assert dry_run.candidate_count == 2
        assert dry_run.demoted_sample_doc_ids == ("conflict-b",)
        assert dry_run.committed is False
        repaired = await primary.repair_source_conflict(
            "conflict.md",
            primary_doc_id="conflict-a",
            expected_candidate_count=dry_run.candidate_count,
            expected_candidate_fingerprint=dry_run.fingerprint,
            dry_run=False,
        )
        assert repaired.committed is True
        resolved = await primary.resolve_doc_source_strict("conflict.md")
        assert isinstance(resolved, SourceUnique)
        assert resolved.doc_id == "conflict-a"
        demoted = await primary.get_by_id_strict("conflict-b")
        assert demoted is not None
        assert demoted["metadata"] == {
            "is_duplicate": True,
            "original_doc_id": "conflict-a",
        }
        assert demoted["chunks_list"] == ["chunk:conflict.md"]

        before_update = await primary.get_by_id_strict("unique")
        assert before_update is not None
        await primary.update_doc_status_fields(
            "unique",
            {
                "status": DocStatus.PROCESSING,
                "updated_at": start + timedelta(days=1),
                "metadata": {"targeted": True},
            },
        )
        after_update = await primary.get_by_id_strict("unique")
        assert after_update is not None
        assert after_update["status"] is DocStatus.PROCESSING
        assert after_update["created_at"] == before_update["created_at"]
        assert after_update["chunks_list"] == before_update["chunks_list"]
        assert after_update["metadata"] == {"targeted": True}

        await primary.delete(["delete-me"])
        assert await primary.get_by_id_strict("delete-me") is None
        await primary.drop()
        assert await primary.is_empty() is True
        isolated_row = await isolated.get_by_id_strict("unique")
        assert isolated_row is not None
        assert isolated_row["file_path"] == "isolated.md"
    finally:
        for item in initialized:
            try:
                await item.drop()
            finally:
                await item.finalize()


class _LiveVectorEmbedding:
    def __init__(self, dimension=3):
        self.embedding_dim = dimension

    async def __call__(self, _texts, **_kwargs):
        raise AssertionError("Live vector tests supply normalized embeddings")


def _live_vector_storage(client, *, workspace, namespace, dimension=3):
    return HologresVectorStorage(
        namespace=namespace,
        workspace=workspace,
        global_config={
            "embedding_batch_num": 2,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.2
            },
        },
        embedding_func=_LiveVectorEmbedding(dimension),
        meta_fields={"content", "source", "src_id", "tgt_id"},
        config=client.config,
        client=client,
    )


async def test_hologres_vector_catalog_crud_and_live_similarity_query(
    hologres_live_client,
):
    client, schema = hologres_live_client
    suffix = uuid.uuid4().hex
    workspace_a = f"lightrag_test_vector_a_{suffix}"
    workspace_b = f"lightrag_test_vector_b_{suffix}"
    primary = _live_vector_storage(
        client,
        workspace=workspace_a,
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
    )
    isolated = _live_vector_storage(
        client,
        workspace=workspace_b,
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
    )
    relations = _live_vector_storage(
        client,
        workspace=workspace_a,
        namespace=NameSpace.VECTOR_STORE_RELATIONSHIPS,
    )
    entities = _live_vector_storage(
        client,
        workspace=workspace_a,
        namespace=NameSpace.VECTOR_STORE_ENTITIES,
    )
    initialized = []

    try:
        for storage in (primary, isolated, relations, entities):
            await storage.initialize()
            initialized.append(storage)

        (descriptor,) = vector_schema_descriptors(schema, 3)
        assert (
            await client.fetch_value(
                descriptor.postcondition_sql,
                *descriptor.postcondition_args,
                descriptor="live.vector.catalog",
            )
            is True
        )

        wrong_dimension = _live_vector_storage(
            client,
            workspace=workspace_a,
            namespace=NameSpace.VECTOR_STORE_CHUNKS,
            dimension=2,
        )
        with pytest.raises(HologresVectorError, match="initialization failed"):
            await wrong_dimension.initialize()

        known_vectors = {
            "identical": {
                "content": "identical",
                "embedding": [1.0, 0.0, 0.0],
                "source": {"kind": "known", "rank": 1},
            },
            "orthogonal": {
                "content": "orthogonal",
                "embedding": [0.0, 1.0, 0.0],
                "source": {"kind": "known", "rank": 2},
            },
            "opposite": {
                "content": "opposite",
                "embedding": [-1.0, 0.0, 0.0],
                "source": {"kind": "known", "rank": 3},
            },
            "tie": {
                "content": "tie",
                "embedding": [1.0, 0.0, 0.0],
                "source": {"kind": "known", "rank": 4},
            },
        }
        await primary.upsert(known_vectors)
        await isolated.upsert(
            {
                "identical": {
                    "content": "isolated",
                    "embedding": [0.0, 0.0, 1.0],
                    "source": {"workspace": "other"},
                }
            }
        )

        assert await primary.get_by_id("identical") == {
            "id": "identical",
            "content": "identical",
            "source": {"kind": "known", "rank": 1},
        }
        assert await primary.get_by_ids(
            ["tie", "missing", "identical", "tie"]
        ) == [
            {
                "id": "tie",
                "content": "tie",
                "source": {"kind": "known", "rank": 4},
            },
            None,
            {
                "id": "identical",
                "content": "identical",
                "source": {"kind": "known", "rank": 1},
            },
            {
                "id": "tie",
                "content": "tie",
                "source": {"kind": "known", "rank": 4},
            },
        ]
        assert await primary.get_vectors_by_ids(
            ["opposite", "missing", "orthogonal", "identical"]
        ) == {
            "identical": [1.0, 0.0, 0.0],
            "opposite": [-1.0, 0.0, 0.0],
            "orthogonal": [0.0, 1.0, 0.0],
        }
        assert (await isolated.get_by_id("identical"))["content"] == "isolated"

        await primary.upsert(
            {
                "identical": {
                    "content": "updated",
                    "embedding": [0.0, 0.0, 1.0],
                    "source": {"replacement": True},
                }
            }
        )
        assert await primary.get_by_id("identical") == {
            "id": "identical",
            "content": "updated",
            "source": {"replacement": True},
        }
        assert await primary.get_vectors_by_ids(["identical"]) == {
            "identical": [0.0, 0.0, 1.0]
        }

        await relations.upsert(
            {
                "relation-a": {
                    "content": "relation-a",
                    "embedding": [1.0, 0.0, 0.0],
                    "src_id": "Alice",
                    "tgt_id": "Bob",
                },
                "relation-b": {
                    "content": "relation-b",
                    "embedding": [0.0, 1.0, 0.0],
                    "src_id": "Carol",
                    "tgt_id": "Alice",
                },
            }
        )
        await relations.delete_entity_relation("Alice")
        assert await relations.get_by_ids(["relation-a", "relation-b"]) == [
            None,
            None,
        ]

        entity_id = compute_mdhash_id("Alice", prefix="ent-")
        await entities.upsert(
            {
                entity_id: {
                    "content": "Alice",
                    "embedding": [1.0, 0.0, 0.0],
                }
            }
        )
        await entities.delete_entity("Alice")
        assert await entities.get_by_id(entity_id) is None

        await primary.upsert(
            {
                "near": {
                    "content": "near",
                    "embedding": [0.8, 0.6, 0.0],
                    "source": {"kind": "known", "rank": 5},
                }
            }
        )
        # Frozen contract: approx_cosine_distance returns cosine similarity,
        # results ordered nearest-first, threshold (0.2) filters the rest.
        results = await primary.query(
            "known query", top_k=4, query_embedding=[1.0, 0.0, 0.0]
        )
        assert [row["id"] for row in results] == ["tie", "near"]
        assert results[0]["content"] == "tie"
        assert results[0]["source"] == {"kind": "known", "rank": 4}
        assert abs(results[0]["distance"] - 1.0) <= 1e-3
        assert results[1]["content"] == "near"
        assert abs(results[1]["distance"] - 0.8) <= 1e-3
        assert all(isinstance(row["created_at"], int) for row in results)
        top_one = await primary.query(
            "known query", top_k=1, query_embedding=[1.0, 0.0, 0.0]
        )
        assert [row["id"] for row in top_one] == ["tie"]
        assert (
            await isolated.query(
                "known query", top_k=4, query_embedding=[1.0, 0.0, 0.0]
            )
            == []
        )
        await primary.delete(["near"])

        await primary.delete(["opposite"])
        assert await primary.get_by_id("opposite") is None
        await primary.drop()
        assert await primary.get_by_ids(["identical", "orthogonal", "tie"]) == [
            None,
            None,
            None,
        ]
        assert (await isolated.get_by_id("identical"))["content"] == "isolated"
    finally:
        for storage in initialized:
            try:
                await storage.drop()
            finally:
                await storage.finalize()
