"""
Pytest configuration and fixtures for multi-tenant testing.

Provides:
- Database fixtures for different testing modes
- Tenant and KB context fixtures
- Mock LLM and embedding services
- Multi-tenant test utilities
"""

import os
import pytest
import asyncio
import psycopg2
import json
from typing import Dict, List, Optional, Generator
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch
import uuid

# ============================================================================
# Environment and Mode Detection
# ============================================================================

MULTITENANT_MODE = os.getenv("MULTITENANT_MODE", "demo")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "lightrag")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "lightrag_secure_password")
POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE", "lightrag_multitenant")


# ============================================================================
# Database Connection Management
# ============================================================================

@pytest.fixture(scope="session")
def db_connection_string():
    """Generate PostgreSQL connection string."""
    return f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DATABASE}"


@pytest.fixture(scope="session")
def postgres_connection():
    """Create persistent PostgreSQL connection for session."""
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database=POSTGRES_DATABASE
        )
        conn.autocommit = False
        yield conn
        conn.close()
    except psycopg2.Error as e:
        pytest.skip(f"PostgreSQL not available: {e}")


@contextmanager
def database_transaction(postgres_connection):
    """Context manager for database transactions with rollback."""
    cursor = postgres_connection.cursor()
    try:
        yield cursor
        postgres_connection.commit()
    except Exception as e:
        postgres_connection.rollback()
        raise e
    finally:
        cursor.close()


# ============================================================================
# Mode-Specific Fixtures
# ============================================================================

@pytest.fixture
def testing_mode():
    """Return current testing mode."""
    return MULTITENANT_MODE


@pytest.fixture
def is_compatibility_mode():
    """Check if running in compatibility mode (MULTITENANT_MODE=off)."""
    return MULTITENANT_MODE == "off"


@pytest.fixture
def is_single_tenant_mode():
    """Check if running in single-tenant mode (MULTITENANT_MODE=on)."""
    return MULTITENANT_MODE == "on"


@pytest.fixture
def is_demo_mode():
    """Check if running in demo mode (MULTITENANT_MODE=demo)."""
    return MULTITENANT_MODE == "demo"


# ============================================================================
# Tenant and KB Fixtures
# ============================================================================

@pytest.fixture
def demo_tenant_acme():
    """Acme Corp tenant for demo mode."""
    return {
        "tenant_id": "acme-corp",
        "name": "Acme Corporation",
        "kbs": ["kb-prod", "kb-dev"]
    }


@pytest.fixture
def demo_tenant_techstart():
    """TechStart tenant for demo mode."""
    return {
        "tenant_id": "techstart",
        "name": "TechStart Inc",
        "kbs": ["kb-main", "kb-backup"]
    }


@pytest.fixture
def default_tenant():
    """Default tenant for compatibility and on modes."""
    return {
        "tenant_id": "default",
        "name": "Default Tenant",
        "kbs": ["default"]
    }


@pytest.fixture
def test_tenant_1():
    """Test tenant 1 for single-tenant mode."""
    return {
        "tenant_id": "tenant-1",
        "name": "Test Tenant 1",
        "kbs": ["kb-default", "kb-secondary", "kb-experimental"]
    }


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_document():
    """Sample document for testing."""
    return {
        "title": "Test Document",
        "content": "This is a test document for LightRAG multi-tenant testing.",
        "file_type": "text",
        "metadata": {
            "source": "test",
            "version": "1.0"
        }
    }


@pytest.fixture
def sample_entity():
    """Sample entity for testing."""
    return {
        "name": "TestEntity",
        "type": "Person",
        "description": "A test entity for multi-tenant isolation testing",
        "metadata": {
            "test": True,
            "created_by": "pytest"
        }
    }


@pytest.fixture
def sample_relation():
    """Sample relation for testing."""
    return {
        "source_entity": "Entity1",
        "target_entity": "Entity2",
        "relation_type": "knows",
        "description": "Test relationship between entities",
        "weight": 0.8
    }


# ============================================================================
# Database Query Helpers
# ============================================================================

class DatabaseHelper:
    """Helper class for database operations in tests."""

    def __init__(self, connection):
        self.connection = connection

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute a SELECT query and return results."""
        with database_transaction(self.connection) as cursor:
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def execute_insert(self, table: str, data: Dict) -> None:
        """Insert a row into a table."""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["%s"] * len(data))
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        with database_transaction(self.connection) as cursor:
            cursor.execute(query, tuple(data.values()))

    def execute_delete(self, table: str, where: Dict) -> int:
        """Delete rows from a table."""
        where_clause = " AND ".join([f"{k} = %s" for k in where.keys()])
        query = f"DELETE FROM {table} WHERE {where_clause}"
        with database_transaction(self.connection) as cursor:
            cursor.execute(query, tuple(where.values()))
            return cursor.rowcount

    def count_documents(self, tenant_id: str, kb_id: str) -> int:
        """Count documents for a tenant/KB."""
        query = "SELECT COUNT(*) as count FROM documents WHERE tenant_id = %s AND kb_id = %s"
        result = self.execute_query(query, (tenant_id, kb_id))
        return result[0]["count"] if result else 0

    def count_entities(self, tenant_id: str, kb_id: str) -> int:
        """Count entities for a tenant/KB."""
        query = "SELECT COUNT(*) as count FROM entities WHERE tenant_id = %s AND kb_id = %s"
        result = self.execute_query(query, (tenant_id, kb_id))
        return result[0]["count"] if result else 0

    def get_all_documents(self, tenant_id: str, kb_id: str) -> List[Dict]:
        """Get all documents for a tenant/KB."""
        query = "SELECT * FROM documents WHERE tenant_id = %s AND kb_id = %s ORDER BY created_at DESC"
        return self.execute_query(query, (tenant_id, kb_id))

    def get_all_entities(self, tenant_id: str, kb_id: str) -> List[Dict]:
        """Get all entities for a tenant/KB."""
        query = "SELECT * FROM entities WHERE tenant_id = %s AND kb_id = %s ORDER BY created_at DESC"
        return self.execute_query(query, (tenant_id, kb_id))

    def verify_tenant_isolation(self, tenant_id: str) -> bool:
        """Verify that no cross-tenant data exists when querying this tenant."""
        # Check that all documents belong to this tenant
        query = """
            SELECT COUNT(*) as count FROM documents 
            WHERE tenant_id != %s AND EXISTS (
                SELECT 1 FROM documents d2 
                WHERE d2.tenant_id = %s AND d2.id = documents.id
            )
        """
        result = self.execute_query(query, (tenant_id, tenant_id))
        return result[0]["count"] == 0 if result else True

    def clear_tenant_data(self, tenant_id: str, kb_id: Optional[str] = None) -> None:
        """Clear all data for a tenant/KB."""
        tables = ["document_status", "embeddings", "documents", "entities", "relations"]
        
        for table in tables:
            if kb_id:
                where = {"tenant_id": tenant_id, "kb_id": kb_id}
            else:
                where = {"tenant_id": tenant_id}
            self.execute_delete(table, where)


@pytest.fixture
def db_helper(postgres_connection):
    """Provide database helper for tests."""
    return DatabaseHelper(postgres_connection)


# ============================================================================
# Mock Services
# ============================================================================

@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing."""
    mock = MagicMock()
    mock.generate = MagicMock(return_value="Mock LLM response")
    mock.extract_entities = MagicMock(return_value=["Entity1", "Entity2"])
    return mock


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for testing."""
    mock = MagicMock()
    mock.embed_text = MagicMock(return_value=[0.1] * 1024)  # 1024-dim vector
    mock.embed_batch = MagicMock(return_value=[[0.1] * 1024 for _ in range(10)])
    return mock


# ============================================================================
# Async Event Loop
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Markers and Parametrization
# ============================================================================

def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "compatibility: mark test to run only in compatibility mode"
    )
    config.addinivalue_line(
        "markers", "single_tenant: mark test to run only in single-tenant mode"
    )
    config.addinivalue_line(
        "markers", "multi_tenant: mark test to run only in demo/multi-tenant mode"
    )
    config.addinivalue_line(
        "markers", "database: mark test that requires database connection"
    )
    config.addinivalue_line(
        "markers", "isolation: mark test that verifies data isolation"
    )


# ============================================================================
# Test Collection Hooks
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """Skip tests based on testing mode."""
    skip_compatibility = pytest.mark.skip(reason="Not in compatibility mode")
    skip_single_tenant = pytest.mark.skip(reason="Not in single-tenant mode")
    skip_multi_tenant = pytest.mark.skip(reason="Not in multi-tenant mode")

    for item in items:
        if "compatibility" in item.keywords and MULTITENANT_MODE != "off":
            item.add_marker(skip_compatibility)
        if "single_tenant" in item.keywords and MULTITENANT_MODE != "on":
            item.add_marker(skip_single_tenant)
        if "multi_tenant" in item.keywords and MULTITENANT_MODE != "demo":
            item.add_marker(skip_multi_tenant)
