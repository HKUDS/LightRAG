# Graph Database Multi-Tenant Support Module
# Supports: Neo4j, Memgraph, NetworkX

from typing import Any, Dict, List, Tuple


class GraphTenantHelper:
    """Helper class for graph DB multi-tenant operations"""

    # Node labels and properties for tenant isolation
    TENANT_NODE_LABEL = "Tenant"
    KB_NODE_LABEL = "KnowledgeBase"
    TENANT_PROPERTY = "tenant_id"
    KB_PROPERTY = "kb_id"

    @staticmethod
    def create_tenant_node_id(tenant_id: str) -> str:
        """Create a node ID for tenant root node"""
        return f"tenant_{tenant_id}"

    @staticmethod
    def create_kb_node_id(tenant_id: str, kb_id: str) -> str:
        """Create a node ID for knowledge base node"""
        return f"kb_{tenant_id}_{kb_id}"

    @staticmethod
    def build_tenant_filter_cypher(
        tenant_id: str, kb_id: str, node_var: str = "n"
    ) -> str:
        """Build a Cypher WHERE clause for tenant isolation"""
        return f"({node_var}:{GraphTenantHelper.TENANT_NODE_LABEL} {{tenant_id: '{tenant_id}'}}) OR EXISTS(({node_var})<-[:IN_KB]-(:KnowledgeBase {{tenant_id: '{tenant_id}', kb_id: '{kb_id}'}}))"

    @staticmethod
    def add_tenant_properties(
        node_data: Dict[str, Any], tenant_id: str, kb_id: str
    ) -> Dict[str, Any]:
        """Add tenant properties to a node"""
        node_data[GraphTenantHelper.TENANT_PROPERTY] = tenant_id
        node_data[GraphTenantHelper.KB_PROPERTY] = kb_id
        return node_data


class Neo4jTenantHelper(GraphTenantHelper):
    """Neo4j-specific tenant helper"""

    @staticmethod
    def build_tenant_constraint_cypher(tenant_id: str) -> str:
        """Build Cypher to create tenant node with constraints"""
        return f"""
        CREATE (t:{GraphTenantHelper.TENANT_NODE_LABEL} {{
            id: '{GraphTenantHelper.create_tenant_node_id(tenant_id)}',
            tenant_id: '{tenant_id}',
            created_at: datetime()
        }})
        """

    @staticmethod
    def build_kb_node_cypher(tenant_id: str, kb_id: str) -> str:
        cypher_query = f"MATCH (t:Tenant {{tenant_id: '{tenant_id}'}}) CREATE (kb:KnowledgeBase {{id: 'kb_{tenant_id}_{kb_id}', tenant_id: '{tenant_id}', kb_id: '{kb_id}', created_at: datetime()}}) CREATE (kb)-[:BELONGS_TO]->(t)"
        return cypher_query

    @staticmethod
    def build_tenant_aware_query(
        base_query: str, tenant_id: str, kb_id: str, node_var: str = "n"
    ) -> Tuple[str, Dict[str, Any]]:
        """Add tenant filtering to a Cypher query"""
        params = {"tenant_id": tenant_id, "kb_id": kb_id}

        # Inject tenant filter into WHERE clause
        where_clause = f"""
        WHERE EXISTS((({node_var})-[:HAS_ENTITY]->(:Entity))-[:BELONGS_TO]->(:KnowledgeBase {{tenant_id: $tenant_id, kb_id: $kb_id}}))
        OR ({node_var}.tenant_id = $tenant_id AND {node_var}.kb_id = $kb_id)
        """

        if "WHERE" in base_query:
            modified_query = base_query.replace("WHERE", "WHERE", 1)
            modified_query = modified_query.replace("WHERE", where_clause, 1)
        else:
            modified_query = base_query + "\n" + where_clause

        return modified_query, params

    @staticmethod
    def delete_tenant_graph(tenant_id: str, kb_id: str) -> str:
        """Create Cypher to delete all data for a tenant/KB"""
        return f"""
        MATCH (kb:KnowledgeBase {{tenant_id: '{tenant_id}', kb_id: '{kb_id}'}})
        MATCH (kb)<-[r1]-(n)
        DETACH DELETE kb, n, r1
        WITH * MATCH (n) WHERE n.tenant_id = '{tenant_id}' AND n.kb_id = '{kb_id}'
        DETACH DELETE n
        """


class MemgraphTenantHelper(GraphTenantHelper):
    """Memgraph-specific tenant helper"""

    @staticmethod
    def build_tenant_openCypher(tenant_id: str) -> str:
        """Build openCypher to create tenant node in Memgraph"""
        return f"""
        CREATE (t:{GraphTenantHelper.TENANT_NODE_LABEL} {{
            id: '{GraphTenantHelper.create_tenant_node_id(tenant_id)}',
            tenant_id: '{tenant_id}'
        }})
        """

    @staticmethod
    def build_tenant_index_cypher(property_name: str) -> str:
        """Build openCypher to create index for tenant filtering"""
        return (
            f"CREATE INDEX ON :{GraphTenantHelper.TENANT_NODE_LABEL}({property_name})"
        )

    @staticmethod
    def build_tenant_aware_query(
        base_query: str, tenant_id: str, kb_id: str, node_var: str = "n"
    ) -> Tuple[str, Dict[str, Any]]:
        """Add tenant filtering to an openCypher query"""
        params = {"tenant_id": tenant_id, "kb_id": kb_id}

        # For Memgraph, similar to Neo4j but using openCypher syntax
        where_clause = (
            f"WHERE {node_var}.tenant_id = $tenant_id AND {node_var}.kb_id = $kb_id"
        )

        if "WHERE" in base_query:
            parts = base_query.split("WHERE", 1)
            modified_query = (
                parts[0] + "WHERE " + where_clause + " AND (" + parts[1] + ")"
            )
        else:
            modified_query = base_query + " " + where_clause

        return modified_query, params


class NetworkXTenantHelper(GraphTenantHelper):
    """NetworkX-specific tenant helper"""

    @staticmethod
    def create_tenant_subgraph(G, tenant_id: str, kb_id: str):
        """Extract a subgraph for a specific tenant/KB from NetworkX graph"""
        tenant_nodes = [
            node
            for node, attr in G.nodes(data=True)
            if attr.get(GraphTenantHelper.TENANT_PROPERTY) == tenant_id
            and attr.get(GraphTenantHelper.KB_PROPERTY) == kb_id
        ]

        return G.subgraph(tenant_nodes).copy()

    @staticmethod
    def filter_edges_by_tenant(
        edges: List[Tuple], G, tenant_id: str, kb_id: str
    ) -> List[Tuple]:
        """Filter edges to include only those in tenant's KB"""
        filtered = []
        for src, tgt in edges:
            src_attrs = G.nodes[src]
            tgt_attrs = G.nodes[tgt]

            if (
                src_attrs.get(GraphTenantHelper.TENANT_PROPERTY) == tenant_id
                and src_attrs.get(GraphTenantHelper.KB_PROPERTY) == kb_id
                and tgt_attrs.get(GraphTenantHelper.TENANT_PROPERTY) == tenant_id
                and tgt_attrs.get(GraphTenantHelper.KB_PROPERTY) == kb_id
            ):
                filtered.append((src, tgt))

        return filtered

    @staticmethod
    def add_tenant_node(G, node_id: str, tenant_id: str, kb_id: str, **attrs):
        """Add a node with tenant properties to NetworkX graph"""
        attrs[GraphTenantHelper.TENANT_PROPERTY] = tenant_id
        attrs[GraphTenantHelper.KB_PROPERTY] = kb_id
        G.add_node(node_id, **attrs)

    @staticmethod
    def delete_tenant_subgraph(G, tenant_id: str, kb_id: str):
        """Delete all nodes/edges for a tenant/KB"""
        nodes_to_delete = [
            node
            for node, attr in G.nodes(data=True)
            if attr.get(GraphTenantHelper.TENANT_PROPERTY) == tenant_id
            and attr.get(GraphTenantHelper.KB_PROPERTY) == kb_id
        ]

        G.remove_nodes_from(nodes_to_delete)
        return len(nodes_to_delete)


# ============================================================================
# TRANSACTION HELPER FOR MULTI-TENANT OPERATIONS
# ============================================================================


class GraphTenantTransaction:
    """Helper for managing tenant-aware graph transactions"""

    def __init__(self, driver, tenant_id: str, kb_id: str):
        self.driver = driver
        self.tenant_id = tenant_id
        self.kb_id = kb_id

    async def create_tenant_structure(self):
        """Create tenant and KB nodes in graph"""
        async with self.driver.session() as session:
            # Create tenant node
            await session.run(
                Neo4jTenantHelper.build_tenant_constraint_cypher(self.tenant_id)
            )

            # Create KB node linked to tenant
            await session.run(
                Neo4jTenantHelper.build_kb_node_cypher(self.tenant_id, self.kb_id)
            )

    async def delete_tenant_data(self):
        """Delete all data for this tenant/KB"""
        async with self.driver.session() as session:
            await session.run(
                Neo4jTenantHelper.delete_tenant_graph(self.tenant_id, self.kb_id)
            )
