"""PGGraphStorage initializes and stores data for every valid workspace name.

Runs against a real PostgreSQL with Apache AGE (POSTGRES_* in the environment).
Before the fix, a workspace longer than 41 characters or starting with a digit
failed in initialize() with "graph name is invalid".
"""

import os

import pytest

from lightrag.kg.shared_storage import initialize_share_data

asyncpg = pytest.importorskip("asyncpg")


async def _noop_embedding(texts):
    return None


async def _age_is_available() -> bool:
    conn = await asyncpg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        database=os.getenv("POSTGRES_DATABASE", "postgres"),
    )
    try:
        return bool(
            await conn.fetchval(
                "SELECT 1 FROM pg_available_extensions WHERE name = 'age'"
            )
        )
    finally:
        await conn.close()


async def _drop_graph(storage):
    async def _operation(connection):
        await connection.execute(
            "SELECT ag_catalog.drop_graph($1::name, true) "
            "WHERE EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = $1::name)",
            storage.graph_name,
        )

    await storage.db._run_with_retry(_operation)


@pytest.mark.integration
@pytest.mark.requires_db
@pytest.mark.parametrize(
    "workspace",
    [
        "tenant_a",
        "w" * 41,
        "w" * 42,
        "acme_corp_customer_support_knowledge_base_production_eu",
        "2026kb",
    ],
)
async def test_initialize_and_round_trip_for_workspace(workspace):
    if not os.getenv("POSTGRES_HOST"):
        pytest.skip("PostgreSQL not configured (POSTGRES_HOST not set)")
    if os.getenv("POSTGRES_WORKSPACE"):
        pytest.skip("POSTGRES_WORKSPACE overrides the workspace under test")
    if not await _age_is_available():
        pytest.skip("Apache AGE extension is not available")

    from lightrag.kg.postgres_impl import PGGraphStorage

    initialize_share_data()
    storage = PGGraphStorage(
        namespace="chunk_entity_relation",
        workspace=workspace,
        global_config={"max_graph_nodes": 1000},
        embedding_func=_noop_embedding,
    )
    try:
        await storage.initialize()
    except RuntimeError as e:
        if "AGE" in str(e):
            pytest.skip(f"Apache AGE is not usable here: {e}")
        raise

    try:
        await storage.upsert_node("A", {"entity_id": "A", "entity_type": "t"})
        await storage.upsert_node("B", {"entity_id": "B", "entity_type": "t"})
        await storage.upsert_edge("A", "B", {"weight": 1.0})

        assert sorted(await storage.get_all_labels()) == ["A", "B"]
        assert await storage.has_edge("A", "B")
    finally:
        await _drop_graph(storage)
        await storage.finalize()
