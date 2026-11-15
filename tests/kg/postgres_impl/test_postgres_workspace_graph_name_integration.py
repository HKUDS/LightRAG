"""PGGraphStorage initializes and stores data for every valid workspace name.

Runs against a real PostgreSQL with Apache AGE (POSTGRES_* in the environment).
On AGE 1.6+, a workspace longer than 41 characters, or one starting with a
digit, used to fail in initialize() with "graph name is invalid". Up to AGE 1.5
a long workspace worked under its name clipped to 63 bytes, and that existing
graph must still be the one the storage opens.
"""

import os

import pytest

from lightrag.kg.shared_storage import initialize_share_data

asyncpg = pytest.importorskip("asyncpg")


async def _noop_embedding(texts):
    return None


async def _connect():
    return await asyncpg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        database=os.getenv("POSTGRES_DATABASE", "postgres"),
    )


async def _skip_unless_age_is_usable():
    if not os.getenv("POSTGRES_HOST"):
        pytest.skip("PostgreSQL not configured (POSTGRES_HOST not set)")
    if os.getenv("POSTGRES_WORKSPACE"):
        pytest.skip("POSTGRES_WORKSPACE overrides the workspace under test")
    conn = await _connect()
    try:
        available = await conn.fetchval(
            "SELECT 1 FROM pg_available_extensions WHERE name = 'age'"
        )
    finally:
        await conn.close()
    if not available:
        pytest.skip("Apache AGE extension is not available")


def _make_storage(workspace):
    from lightrag.kg.postgres_impl import PGGraphStorage

    return PGGraphStorage(
        namespace="chunk_entity_relation",
        workspace=workspace,
        global_config={"max_graph_nodes": 1000},
        embedding_func=_noop_embedding,
    )


async def _initialize(storage):
    try:
        await storage.initialize()
    except RuntimeError as e:
        if "AGE" in str(e):
            pytest.skip(f"Apache AGE is not usable here: {e}")
        raise


async def _write_a_to_b(storage):
    await storage.upsert_node("A", {"entity_id": "A", "entity_type": "t"})
    await storage.upsert_node("B", {"entity_id": "B", "entity_type": "t"})
    await storage.upsert_edge("A", "B", {"weight": 1.0})


async def _drop_graph(graph_name):
    conn = await _connect()
    try:
        await conn.execute(
            "SELECT ag_catalog.drop_graph($1::name, true) "
            "WHERE EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = $1::name)",
            graph_name,
        )
    finally:
        await conn.close()


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
    await _skip_unless_age_is_usable()
    initialize_share_data()
    storage = _make_storage(workspace)
    await _initialize(storage)

    try:
        await _write_a_to_b(storage)

        assert sorted(await storage.get_all_labels()) == ["A", "B"]
        assert await storage.has_edge("A", "B")
    finally:
        await _drop_graph(storage.graph_name)
        await storage.finalize()


@pytest.mark.integration
@pytest.mark.requires_db
async def test_existing_clipped_graph_is_still_used(monkeypatch):
    """Data written under the clipped name (AGE 1.5 or earlier) stays visible."""
    from lightrag.kg.postgres_impl import PGGraphStorage

    await _skip_unless_age_is_usable()
    initialize_share_data()
    workspace = "legacy_" + "x" * 50
    old = _make_storage(workspace)
    clipped = old._get_legacy_clipped_graph_name()
    assert clipped is not None

    # Write the way AGE 1.5 and earlier did: every statement used the clipped name.
    monkeypatch.setattr(
        PGGraphStorage, "_get_workspace_graph_name", lambda self: clipped
    )
    await _initialize(old)
    await _write_a_to_b(old)
    await old.finalize()
    monkeypatch.undo()

    storage = _make_storage(workspace)
    await _initialize(storage)
    try:
        assert storage.graph_name == clipped
        assert sorted(await storage.get_all_labels()) == ["A", "B"]
        assert await storage.has_edge("A", "B")
    finally:
        await _drop_graph(clipped)
        await storage.finalize()


@pytest.mark.integration
@pytest.mark.requires_db
async def test_clipped_graph_left_by_a_failed_startup_is_not_reused():
    """On AGE 1.6+ a failed startup leaves an empty clipped graph without labels."""
    await _skip_unless_age_is_usable()
    initialize_share_data()
    workspace = "leftover_" + "x" * 50
    storage = _make_storage(workspace)
    clipped = storage._get_legacy_clipped_graph_name()
    assert clipped is not None

    # What create_graph() leaves when the create_vlabel() after it fails.
    conn = await _connect()
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS age")
        # AGE 1.7's create_graph() needs ag_catalog on the search_path.
        await conn.execute('SET search_path = ag_catalog, "$user", public')
        await conn.execute("SELECT ag_catalog.create_graph($1::name)", clipped)
    finally:
        await conn.close()

    try:
        await _initialize(storage)
        assert storage.graph_name == storage._get_workspace_graph_name()
        assert storage.graph_name.startswith("ws_")
        await _write_a_to_b(storage)
        assert sorted(await storage.get_all_labels()) == ["A", "B"]
    finally:
        await _drop_graph(storage.graph_name)
        await _drop_graph(clipped)
        await storage.finalize()
