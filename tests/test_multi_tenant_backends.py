from lightrag.kg.postgres_tenant_support import (
    TenantSQLBuilder,
    get_composite_key,
    ensure_tenant_context,
)
from lightrag.kg.mongo_tenant_support import MongoTenantHelper
from lightrag.kg.redis_tenant_support import RedisTenantHelper
from lightrag.kg.vector_tenant_support import (
    QdrantTenantHelper,
    MilvusTenantHelper,
    VectorTenantHelper,
)
from lightrag.kg.graph_tenant_support import (
    Neo4jTenantHelper,
    NetworkXTenantHelper,
    GraphTenantHelper,
)


# ============================================================================
# POSTGRES MULTI-TENANT TESTS
# ============================================================================


class TestPostgresMultiTenant:
    """Test PostgreSQL multi-tenant isolation"""

    def test_tenant_sql_builder_add_filter(self):
        sql = "SELECT * FROM users"
        filtered_sql, next_param = TenantSQLBuilder.add_tenant_filter(sql)
        assert "tenant_id=$1" in filtered_sql
        assert "kb_id=$2" in filtered_sql
        assert next_param == 3

    def test_tenant_sql_builder_add_filter_with_where(self):
        sql = "SELECT * FROM users WHERE age > 18"
        filtered_sql, next_param = TenantSQLBuilder.add_tenant_filter(sql)
        assert "tenant_id=$1" in filtered_sql
        assert "AND age > 18" in filtered_sql
        assert next_param == 3

    def test_composite_key_generation(self):
        key = get_composite_key("tenant1", "kb1", "user123")
        assert key == "tenant1:kb1:user123"

    def test_composite_key_with_special_chars(self):
        key = get_composite_key("tenant-1", "kb-2", "doc_id")
        assert key == "tenant-1:kb-2:doc_id"

    def test_ensure_tenant_context_default(self):
        tenant_id, kb_id = ensure_tenant_context()
        assert tenant_id == "default"
        assert kb_id == "default"

    def test_ensure_tenant_context_provided(self):
        tenant_id, kb_id = ensure_tenant_context("acme", "kb-prod")
        assert tenant_id == "acme"
        assert kb_id == "kb-prod"

    def test_ensure_tenant_context_partial(self):
        tenant_id, kb_id = ensure_tenant_context(tenant_id="acme")
        assert tenant_id == "acme"
        assert kb_id == "default"


# ============================================================================
# MONGODB MULTI-TENANT TESTS
# ============================================================================


class TestMongoMultiTenant:
    """Test MongoDB multi-tenant isolation"""

    def test_add_tenant_fields(self):
        doc = {"name": "test", "value": 123}
        updated = MongoTenantHelper.add_tenant_fields(doc, "tenant1", "kb1")
        assert updated["tenant_id"] == "tenant1"
        assert updated["kb_id"] == "kb1"
        assert updated["name"] == "test"

    def test_get_tenant_filter_basic(self):
        filter_dict = MongoTenantHelper.get_tenant_filter("tenant1", "kb1")
        assert filter_dict["tenant_id"] == "tenant1"
        assert filter_dict["kb_id"] == "kb1"

    def test_get_tenant_filter_with_additional(self):
        additional = {"status": "active", "age": {"$gt": 18}}
        filter_dict = MongoTenantHelper.get_tenant_filter("tenant1", "kb1", additional)
        assert filter_dict["tenant_id"] == "tenant1"
        assert filter_dict["kb_id"] == "kb1"
        assert filter_dict["status"] == "active"
        assert filter_dict["age"] == {"$gt": 18}

    def test_create_tenant_indexes(self):
        indexes = MongoTenantHelper.create_tenant_indexes("users")
        assert len(indexes) == 2
        assert any(idx["name"] == "idx_users_tenant_kb" for idx in indexes)
        assert any(idx["name"] == "idx_users_tenant_kb_id" for idx in indexes)

    def test_build_upsert_with_tenant(self):
        filter_dict = {"_id": "user123"}
        update_dict = {"$set": {"name": "John", "email": "john@example.com"}}

        result_filter, result_update = MongoTenantHelper.build_upsert_with_tenant(
            filter_dict, update_dict, "tenant1", "kb1"
        )

        assert result_filter["tenant_id"] == "tenant1"
        assert result_filter["kb_id"] == "kb1"
        assert result_update["$set"]["tenant_id"] == "tenant1"
        assert result_update["$set"]["kb_id"] == "kb1"


# ============================================================================
# REDIS MULTI-TENANT TESTS
# ============================================================================


class TestRedisMultiTenant:
    """Test Redis multi-tenant isolation"""

    def test_make_tenant_key(self):
        key = RedisTenantHelper.make_tenant_key("tenant1", "kb1", "user123")
        assert key == "tenant1:kb1:user123"

    def test_parse_tenant_key(self):
        key = "tenant1:kb1:user123"
        parsed = RedisTenantHelper.parse_tenant_key(key)
        assert parsed["tenant_id"] == "tenant1"
        assert parsed["kb_id"] == "kb1"
        assert parsed["original_key"] == "user123"

    def test_parse_non_tenant_key(self):
        key = "regular_key"
        parsed = RedisTenantHelper.parse_tenant_key(key)
        assert parsed["original_key"] == "regular_key"
        assert "tenant_id" not in parsed

    def test_get_tenant_key_pattern(self):
        pattern = RedisTenantHelper.get_tenant_key_pattern("tenant1", "kb1")
        assert pattern == "tenant1:kb1:*"

    def test_get_tenant_key_pattern_custom(self):
        pattern = RedisTenantHelper.get_tenant_key_pattern("tenant1", "kb1", "user*")
        assert pattern == "tenant1:kb1:user*"

    def test_extract_original_key(self):
        tenant_key = "tenant1:kb1:document_123"
        original = RedisTenantHelper.extract_original_key(tenant_key)
        assert original == "document_123"

    def test_batch_make_tenant_keys(self):
        keys = ["key1", "key2", "key3"]
        tenant_keys = RedisTenantHelper.batch_make_tenant_keys("tenant1", "kb1", keys)
        assert len(tenant_keys) == 3
        assert tenant_keys[0] == "tenant1:kb1:key1"
        assert tenant_keys[2] == "tenant1:kb1:key3"


# ============================================================================
# VECTOR DB MULTI-TENANT TESTS
# ============================================================================


class TestVectorMultiTenant:
    """Test vector DB multi-tenant isolation"""

    def test_add_tenant_metadata(self):
        payload = {"content": "test vector", "id": "vec1"}
        updated = VectorTenantHelper.add_tenant_metadata(payload, "tenant1", "kb1")
        assert updated["tenant_id"] == "tenant1"
        assert updated["kb_id"] == "kb1"
        assert updated["content"] == "test vector"

    def test_make_tenant_vector_id(self):
        vector_id = VectorTenantHelper.make_tenant_id("tenant1", "kb1", "vec123")
        assert vector_id == "tenant1:kb1:vec123"

    def test_parse_tenant_vector_id(self):
        vector_id = "tenant1:kb1:vec123"
        parsed = VectorTenantHelper.parse_tenant_id(vector_id)
        assert parsed["tenant_id"] == "tenant1"
        assert parsed["kb_id"] == "kb1"
        assert parsed["original_id"] == "vec123"

    def test_qdrant_build_filter(self):
        filter_dict = QdrantTenantHelper.build_qdrant_filter("tenant1", "kb1")
        assert "must" in filter_dict
        assert len(filter_dict["must"]) == 2
        assert any(c.get("key") == "tenant_id" for c in filter_dict["must"])
        assert any(c.get("key") == "kb_id" for c in filter_dict["must"])

    def test_milvus_build_expr(self):
        expr = MilvusTenantHelper.build_milvus_expr("tenant1", "kb1")
        assert 'tenant_id == "tenant1"' in expr
        assert 'kb_id == "kb1"' in expr

    def test_milvus_build_expr_with_additional(self):
        expr = MilvusTenantHelper.build_milvus_expr("tenant1", "kb1", "score > 0.8")
        assert 'tenant_id == "tenant1"' in expr
        assert 'kb_id == "kb1"' in expr
        assert "score > 0.8" in expr


# ============================================================================
# GRAPH DB MULTI-TENANT TESTS
# ============================================================================


class TestGraphMultiTenant:
    """Test graph DB multi-tenant isolation"""

    def test_create_tenant_node_id(self):
        node_id = GraphTenantHelper.create_tenant_node_id("tenant1")
        assert node_id == "tenant_tenant1"

    def test_create_kb_node_id(self):
        node_id = GraphTenantHelper.create_kb_node_id("tenant1", "kb1")
        assert node_id == "kb_tenant1_kb1"

    def test_add_tenant_properties(self):
        node_data = {"label": "Entity", "name": "John"}
        updated = GraphTenantHelper.add_tenant_properties(node_data, "tenant1", "kb1")
        assert updated["tenant_id"] == "tenant1"
        assert updated["kb_id"] == "kb1"
        assert updated["name"] == "John"

    def test_neo4j_build_tenant_constraint_cypher(self):
        cypher = Neo4jTenantHelper.build_tenant_constraint_cypher("tenant1")
        assert "CREATE" in cypher
        assert "Tenant" in cypher
        assert "tenant1" in cypher

    def test_networkx_add_tenant_node(self):
        import networkx as nx

        G = nx.DiGraph()

        NetworkXTenantHelper.add_tenant_node(
            G, "node1", "tenant1", "kb1", label="Entity"
        )

        assert "node1" in G.nodes
        assert G.nodes["node1"]["tenant_id"] == "tenant1"
        assert G.nodes["node1"]["kb_id"] == "kb1"
        assert G.nodes["node1"]["label"] == "Entity"

    def test_networkx_filter_edges_by_tenant(self):
        import networkx as nx

        G = nx.DiGraph()

        G.add_node("n1", tenant_id="tenant1", kb_id="kb1")
        G.add_node("n2", tenant_id="tenant1", kb_id="kb1")
        G.add_node("n3", tenant_id="tenant2", kb_id="kb2")

        G.add_edge("n1", "n2")
        G.add_edge("n2", "n3")
        G.add_edge("n3", "n1")

        edges = [("n1", "n2"), ("n2", "n3"), ("n3", "n1")]
        filtered = NetworkXTenantHelper.filter_edges_by_tenant(
            edges, G, "tenant1", "kb1"
        )

        assert len(filtered) == 1
        assert ("n1", "n2") in filtered


# ============================================================================
# ISOLATION SECURITY TESTS
# ============================================================================


class TestTenantIsolationSecurity:
    """Test that tenant isolation is secure"""

    def test_no_cross_tenant_filter_bypass(self):
        # Ensure tenant filter cannot be bypassed
        filter1 = MongoTenantHelper.get_tenant_filter("tenant1", "kb1", {})
        filter2 = MongoTenantHelper.get_tenant_filter("tenant2", "kb2", {})

        assert filter1["tenant_id"] != filter2["tenant_id"]
        assert filter1["kb_id"] != filter2["kb_id"]

    def test_redis_keys_do_not_collide(self):
        key1 = RedisTenantHelper.make_tenant_key("t1", "kb1", "doc1")
        key2 = RedisTenantHelper.make_tenant_key("t2", "kb2", "doc1")
        key3 = RedisTenantHelper.make_tenant_key("t1", "kb2", "doc1")

        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_vector_ids_are_unique(self):
        id1 = VectorTenantHelper.make_tenant_id("t1", "kb1", "v1")
        id2 = VectorTenantHelper.make_tenant_id("t1", "kb1", "v2")
        id3 = VectorTenantHelper.make_tenant_id("t1", "kb2", "v1")

        assert id1 != id2
        assert id1 != id3


# ============================================================================
# BACKWARD COMPATIBILITY TESTS
# ============================================================================


class TestBackwardCompatibility:
    """Test backward compatibility with non-tenant operations"""

    def test_default_tenant_context(self):
        tenant_id, kb_id = ensure_tenant_context(None, None)
        assert tenant_id == "default"
        assert kb_id == "default"

    def test_redis_legacy_key_parsing(self):
        legacy_key = "users:123"
        parsed = RedisTenantHelper.parse_tenant_key(legacy_key)
        assert parsed["original_key"] == legacy_key

    def test_composite_key_single_level(self):
        key = get_composite_key("single", "default")
        assert key == "single:default"


if __name__ == "__main__":
    pass
