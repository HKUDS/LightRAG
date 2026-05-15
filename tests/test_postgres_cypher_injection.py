"""
Unit tests for Cypher injection prevention in PGGraphStorage write paths.

Verifies that upsert_node and upsert_edge keep entity IDs parameterized while
rendering property maps as safely escaped Cypher literals, which is required by
Apache AGE because ``SET ... += $props`` is not supported.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from lightrag.kg.postgres_impl import PGGraphStorage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_graph_storage() -> PGGraphStorage:
    """Construct a PGGraphStorage instance with a mocked db."""
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = "test_ws"
    storage.namespace = "test_graph"
    storage.graph_name = "test_graph"
    storage.db = MagicMock()
    return storage


class _FakeConnection:
    """Captures statements + args passed to a fake asyncpg connection."""

    def __init__(self):
        self.calls: list[dict] = []

    def transaction(self):
        return _FakeTransaction()

    async def execute(self, sql, *args):
        self.calls.append({"sql": sql, "args": args})
        return ""


class _FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


async def _capture_upsert_edge(storage: PGGraphStorage, src: str, tgt: str, edge_data):
    """Invoke upsert_edge against a fake connection and return the captured calls."""
    conn = _FakeConnection()

    async def fake_run_with_retry(operation, **_kwargs):
        return await operation(conn)

    storage.db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)
    await storage.upsert_edge(src, tgt, edge_data)
    return conn.calls


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
    assert "$1::agtype" in call["sql"]
    assert '"Alice"' not in call["sql"].replace("$1::agtype", "")
    assert "params" in call
    params = json.loads(call["params"]["params"])
    assert params["entity_id"] == "Alice"
    assert "props" not in params
    assert '`description`: "A person"' in call["sql"]


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
    """Property values with special characters are safely escaped in Cypher."""
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
    assert (
        '`description`: "He said \\"hello\\" and used a backslash \\\\"' in call["sql"]
    )
    assert '`notes`: "Line1\\nLine2\\tTabbed"' in call["sql"]
    assert '`formula`: "x < 5 && y > 3"' in call["sql"]


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
    assert '`description`: "描述"' in call["sql"]


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
async def test_upsert_node_escapes_backticks_in_property_keys():
    """Backticks in property keys must be escaped before inlining the map."""
    storage = make_graph_storage()
    captured_calls: list[dict] = []

    async def fake_query(sql, **kwargs):
        captured_calls.append({"sql": sql, **kwargs})
        return []

    with patch.object(storage, "_query", side_effect=fake_query):
        await storage.upsert_node(
            "node",
            {"entity_id": "node", "danger`key": 'value "quoted"'},
        )

    assert '`danger``key`: "value \\"quoted\\""' in captured_calls[0]["sql"]


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
    calls = await _capture_upsert_edge(
        storage, "Alice", "Bob", {"weight": "1.0", "description": "knows"}
    )

    # Two statements run on the connection: advisory lock first, then cypher.
    assert len(calls) == 2

    lock_sql = calls[0]["sql"]
    # Raw node IDs are positional params on the lock, never interpolated.
    assert "Alice" not in lock_sql
    assert "Bob" not in lock_sql
    assert calls[0]["args"] == ("Alice", "Bob")

    cypher_call = calls[1]
    cypher_sql = cypher_call["sql"]
    assert "$1::agtype" in cypher_sql
    assert '"Alice"' not in cypher_sql.replace("$1::agtype", "")
    assert '"Bob"' not in cypher_sql.replace("$1::agtype", "")
    # Cypher params arrive as a single positional agtype JSON arg.
    params = json.loads(cypher_call["args"][0])
    assert params["src_id"] == "Alice"
    assert params["tgt_id"] == "Bob"
    assert "props" not in params
    assert '`weight`: "1.0"' in cypher_sql
    assert '`description`: "knows"' in cypher_sql


@pytest.mark.asyncio
async def test_upsert_edge_injection_payload():
    """Injection payloads in edge entity IDs are safely parameterized."""
    storage = make_graph_storage()
    injection_src = 'src"}) MATCH (x) DETACH DELETE x; //'
    injection_tgt = 'tgt"})-[r]-() DELETE r; //'
    calls = await _capture_upsert_edge(
        storage, injection_src, injection_tgt, {"description": "edge"}
    )

    # Injection payloads must never appear in either SQL template — they only
    # flow through positional params.
    for call in calls:
        assert "DETACH DELETE" not in call["sql"]
        assert "DELETE r" not in call["sql"]
        assert injection_src not in call["sql"]
        assert injection_tgt not in call["sql"]

    # Lock statement passes raw IDs as positional params.
    assert calls[0]["args"] == (injection_src, injection_tgt)

    # Cypher params arrive as a single positional agtype JSON arg.
    params = json.loads(calls[1]["args"][0])
    assert params["src_id"] == injection_src
    assert params["tgt_id"] == injection_tgt


@pytest.mark.asyncio
async def test_upsert_edge_unicode_entity_ids():
    """Unicode entity IDs in edges are safely parameterized."""
    storage = make_graph_storage()
    src = "\u5317\u4eac"
    tgt = "\u4e0a\u6d77"
    calls = await _capture_upsert_edge(
        storage, src, tgt, {"description": "\u8def\u7ebf"}
    )

    # Lock statement carries the raw IDs as positional params, not interpolated.
    assert calls[0]["args"] == (src, tgt)
    assert src not in calls[0]["sql"]
    assert tgt not in calls[0]["sql"]

    # Cypher params parsed from the positional agtype JSON arg.
    cypher_sql = calls[1]["sql"]
    params = json.loads(calls[1]["args"][0])
    assert params["src_id"] == src
    assert params["tgt_id"] == tgt
    assert '`description`: "路线"' in cypher_sql


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
