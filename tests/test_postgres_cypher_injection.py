"""
Unit tests for Cypher injection prevention in PGGraphStorage write paths.

Verifies that upsert_node and upsert_edge use parameterized Cypher queries
(via AGE's $1::agtype mechanism) instead of string interpolation, preventing
injection through crafted entity names or property values.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from lightrag.kg.postgres_impl import PGGraphStorage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_graph_storage() -> PGGraphStorage:
    """Construct a PGGraphStorage instance with a mocked _query method."""
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = "test_ws"
    storage.namespace = "test_graph"
    storage.graph_name = "test_graph"
    storage.db = MagicMock()
    return storage


# ---------------------------------------------------------------------------
# upsert_node — parameterized Cypher
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_node_uses_parameterized_cypher():
    """upsert_node must pass entity_id as a Cypher parameter, not interpolate it."""
    storage = make_graph_storage()
    captured_calls: list[dict] = []

    async def fake_query(sql, **kwargs):
        captured_calls.append({"sql": sql, **kwargs})
        return []

    with patch.object(storage, "_query", side_effect=fake_query):
        await storage.upsert_node(
            "Alice", {"entity_id": "Alice", "description": "A person"}
        )

    assert len(captured_calls) == 1
    call = captured_calls[0]
    # SQL must use $1::agtype parameter binding
    assert "$1::agtype" in call["sql"]
    # Entity name must NOT appear literally in the SQL
    assert '"Alice"' not in call["sql"].replace("$1::agtype", "")
    # Params must be passed
    assert "params" in call
    params = json.loads(call["params"]["params"])
    assert params["entity_id"] == "Alice"
    assert params["props"]["description"] == "A person"


@pytest.mark.asyncio
async def test_upsert_node_injection_payload_in_entity_id():
    """A Cypher injection payload in entity_id must be treated as data, not code."""
    storage = make_graph_storage()
    injection = 'test"}) RETURN n; MATCH (m) DETACH DELETE m; //'
    captured_calls: list[dict] = []

    async def fake_query(sql, **kwargs):
        captured_calls.append({"sql": sql, **kwargs})
        return []

    with patch.object(storage, "_query", side_effect=fake_query):
        await storage.upsert_node(
            injection, {"entity_id": injection, "description": "malicious"}
        )

    call = captured_calls[0]
    # The injection payload must NOT appear in the SQL string
    assert "DETACH DELETE" not in call["sql"]
    assert injection not in call["sql"]
    # It must be safely contained in the JSON parameter
    params = json.loads(call["params"]["params"])
    assert params["entity_id"] == injection


@pytest.mark.asyncio
async def test_upsert_node_special_chars_in_properties():
    """Property values with special characters are safely parameterized."""
    storage = make_graph_storage()
    captured_calls: list[dict] = []

    async def fake_query(sql, **kwargs):
        captured_calls.append({"sql": sql, **kwargs})
        return []

    node_data = {
        "entity_id": "test_node",
        "description": 'He said "hello" and used a backslash \\',
        "notes": "Line1\nLine2\tTabbed",
        "formula": "x < 5 && y > 3",
    }

    with patch.object(storage, "_query", side_effect=fake_query):
        await storage.upsert_node("test_node", node_data)

    call = captured_calls[0]
    params = json.loads(call["params"]["params"])
    assert params["props"]["description"] == node_data["description"]
    assert params["props"]["notes"] == node_data["notes"]
    assert params["props"]["formula"] == node_data["formula"]


@pytest.mark.asyncio
async def test_upsert_node_unicode_entity_id():
    """Unicode entity names are safely parameterized."""
    storage = make_graph_storage()
    captured_calls: list[dict] = []

    async def fake_query(sql, **kwargs):
        captured_calls.append({"sql": sql, **kwargs})
        return []

    unicode_id = "\u4e2d\u6587\u5b9e\u4f53"  # Chinese characters
    with patch.object(storage, "_query", side_effect=fake_query):
        await storage.upsert_node(
            unicode_id, {"entity_id": unicode_id, "description": "\u63cf\u8ff0"}
        )

    call = captured_calls[0]
    params = json.loads(call["params"]["params"])
    assert params["entity_id"] == unicode_id
    assert params["props"]["description"] == "\u63cf\u8ff0"


@pytest.mark.asyncio
async def test_upsert_node_dollar_signs_in_entity_id():
    """Dollar signs in entity_id don't break dollar-quoting of the Cypher template."""
    storage = make_graph_storage()
    captured_calls: list[dict] = []

    async def fake_query(sql, **kwargs):
        captured_calls.append({"sql": sql, **kwargs})
        return []

    dollar_id = "price is $100 or $$200$$"
    with patch.object(storage, "_query", side_effect=fake_query):
        await storage.upsert_node(
            dollar_id, {"entity_id": dollar_id, "description": "has dollars"}
        )

    call = captured_calls[0]
    # The dollar signs are in the params, not the SQL template
    params = json.loads(call["params"]["params"])
    assert params["entity_id"] == dollar_id


@pytest.mark.asyncio
async def test_upsert_node_requires_entity_id():
    """upsert_node still raises ValueError when entity_id is missing."""
    storage = make_graph_storage()
    with pytest.raises(ValueError, match="entity_id"):
        await storage.upsert_node("test", {"description": "no entity_id"})


# ---------------------------------------------------------------------------
# upsert_edge — parameterized Cypher
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_edge_uses_parameterized_cypher():
    """upsert_edge must pass entity IDs as Cypher parameters."""
    storage = make_graph_storage()
    captured_calls: list[dict] = []

    async def fake_query(sql, **kwargs):
        captured_calls.append({"sql": sql, **kwargs})
        return []

    with patch.object(storage, "_query", side_effect=fake_query):
        await storage.upsert_edge("Alice", "Bob", {"weight": "1.0", "description": "knows"})

    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert "$1::agtype" in call["sql"]
    # Entity names must not appear in SQL
    assert '"Alice"' not in call["sql"].replace("$1::agtype", "")
    assert '"Bob"' not in call["sql"].replace("$1::agtype", "")
    params = json.loads(call["params"]["params"])
    assert params["src_id"] == "Alice"
    assert params["tgt_id"] == "Bob"
    assert params["props"]["weight"] == "1.0"


@pytest.mark.asyncio
async def test_upsert_edge_injection_payload():
    """Injection payloads in edge entity IDs are safely parameterized."""
    storage = make_graph_storage()
    injection_src = 'src"}) MATCH (x) DETACH DELETE x; //'
    injection_tgt = 'tgt"})-[r]-() DELETE r; //'
    captured_calls: list[dict] = []

    async def fake_query(sql, **kwargs):
        captured_calls.append({"sql": sql, **kwargs})
        return []

    with patch.object(storage, "_query", side_effect=fake_query):
        await storage.upsert_edge(
            injection_src, injection_tgt, {"description": "edge"}
        )

    call = captured_calls[0]
    assert "DETACH DELETE" not in call["sql"]
    assert "DELETE r" not in call["sql"]
    params = json.loads(call["params"]["params"])
    assert params["src_id"] == injection_src
    assert params["tgt_id"] == injection_tgt


@pytest.mark.asyncio
async def test_upsert_edge_unicode_entity_ids():
    """Unicode entity IDs in edges are safely parameterized."""
    storage = make_graph_storage()
    captured_calls: list[dict] = []

    async def fake_query(sql, **kwargs):
        captured_calls.append({"sql": sql, **kwargs})
        return []

    with patch.object(storage, "_query", side_effect=fake_query):
        await storage.upsert_edge(
            "\u5317\u4eac", "\u4e0a\u6d77", {"description": "\u8def\u7ebf"}
        )

    params = json.loads(captured_calls[0]["params"]["params"])
    assert params["src_id"] == "\u5317\u4eac"
    assert params["tgt_id"] == "\u4e0a\u6d77"


# ---------------------------------------------------------------------------
# _normalize_node_id — defence-in-depth for remaining interpolation paths
# ---------------------------------------------------------------------------


def test_normalize_node_id_strips_null_bytes():
    """Null bytes are stripped to prevent string truncation."""
    assert PGGraphStorage._normalize_node_id("before\x00after") == "beforeafter"


def test_normalize_node_id_escapes_backslash_and_quote():
    """Backslashes and double quotes are escaped."""
    assert PGGraphStorage._normalize_node_id('a\\"b') == 'a\\\\\\"b'


def test_normalize_node_id_injection_payload():
    """Injection payload is escaped so it cannot break out of Cypher string."""
    payload = 'test"}) RETURN n; MATCH (m) DETACH DELETE m; //'
    normalized = PGGraphStorage._normalize_node_id(payload)
    # The double quote must be escaped
    assert '\\"' in normalized
    # The escaped string must not contain an unescaped double quote
    # (remove all escaped quotes and check no raw ones remain)
    unescaped = normalized.replace('\\"', "")
    assert '"' not in unescaped


# ---------------------------------------------------------------------------
# _query write path passes params to db.execute
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_write_path_passes_params():
    """When readonly=False, _query must forward params to db.execute."""
    storage = make_graph_storage()
    captured_execute_kwargs: list[dict] = []

    async def fake_execute(sql, **kwargs):
        captured_execute_kwargs.append(kwargs)
        return None

    storage.db.execute = fake_execute

    test_params = {"params": json.dumps({"entity_id": "test"})}
    await storage._query(
        "SELECT 1",
        readonly=False,
        upsert=True,
        params=test_params,
    )

    assert len(captured_execute_kwargs) == 1
    assert captured_execute_kwargs[0]["data"] == test_params
