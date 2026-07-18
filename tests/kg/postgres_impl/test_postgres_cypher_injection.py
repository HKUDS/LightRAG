"""
Unit tests for Cypher injection prevention in PGGraphStorage write paths.

Verifies that upsert_node and upsert_edge keep entity IDs parameterized while
rendering property maps as safely escaped Cypher literals, which is required by
Apache AGE because ``SET ... += $props`` is not supported.
"""

import json
import re
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from lightrag.kg.postgres_impl import PGGraphStorage, _dollar_quote


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


def _parse_dollar_quoted(wrapped: str) -> tuple[str, str]:
    """Decode a single dollar-quoted literal the way PostgreSQL's scanner does.

    Reads the opening ``$tag$`` delimiter, then treats everything up to the
    *next* occurrence of that exact delimiter as the literal body. Returns
    ``(content, trailing)`` where ``trailing`` is whatever follows the closing
    delimiter. A correctly quoted literal round-trips to ``(original, "")``;
    a broken one (premature close from a seam/interior collision) leaks the
    remainder into ``trailing``.
    """
    assert wrapped.startswith("$"), wrapped
    tag_close = wrapped.index("$", 1)
    delim = wrapped[: tag_close + 1]  # e.g. "$AGE1$"
    body = wrapped[len(delim) :]
    idx = body.find(delim)
    assert idx != -1, f"no closing delimiter {delim!r} in {wrapped!r}"
    return body[:idx], body[idx + len(delim) :]


def _strip_dollar_literals(sql: str) -> str:
    """Remove every dollar-quoted literal, leaving only the SQL skeleton.

    Mirrors PostgreSQL tokenizing: an opening ``$tag$`` (empty or identifier
    tag) consumes everything through its matching close. Whatever remains is
    code that the server would actually execute — injection payloads that are
    correctly contained inside a literal must not appear here.
    """
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        if sql[i] == "$":
            j = sql.find("$", i + 1)
            if j != -1:
                delim = sql[i : j + 1]
                tag = delim[1:-1]
                if tag == "" or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tag):
                    end = sql.find(delim, j + 1)
                    if end != -1:
                        i = end + len(delim)
                        continue
        out.append(sql[i])
        i += 1
    return "".join(out)


async def _capture_bfs_subgraph_query(storage: PGGraphStorage, node_label: str) -> str:
    """Run _bfs_subgraph with a stubbed _query and return the built SQL string."""
    captured: list[str] = []

    async def fake_query(sql, **kwargs):
        captured.append(sql)
        return []  # empty result → _bfs_subgraph returns after the first query

    with patch.object(storage, "_query", side_effect=fake_query):
        await storage._bfs_subgraph(node_label, max_depth=1, max_nodes=10)

    assert captured, "expected _bfs_subgraph to issue at least one query"
    return captured[0]


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

    # Three statements: per-edge lock, graph-wide shared lock, then cypher.
    assert len(calls) == 3

    lock_sql = calls[0]["sql"]
    # Raw node IDs are positional params on the lock, never interpolated.
    assert "Alice" not in lock_sql
    assert "Bob" not in lock_sql
    # graph_name flows as $1, the endpoint pair as $2/$3.
    assert calls[0]["args"] == ("test_graph", "Alice", "Bob")

    cypher_call = calls[2]
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

    # Lock statement passes graph_name + raw IDs as positional params.
    assert calls[0]["args"] == ("test_graph", injection_src, injection_tgt)

    # Cypher params arrive as a single positional agtype JSON arg (3rd statement,
    # after the per-edge and graph-wide-shared locks).
    params = json.loads(calls[2]["args"][0])
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

    # Lock statement carries graph_name + raw IDs as positional params, not
    # interpolated.
    assert calls[0]["args"] == ("test_graph", src, tgt)
    assert src not in calls[0]["sql"]
    assert tgt not in calls[0]["sql"]

    # Cypher params parsed from the positional agtype JSON arg (3rd statement).
    cypher_sql = calls[2]["sql"]
    params = json.loads(calls[2]["args"][0])
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


# ---------------------------------------------------------------------------
# _dollar_quote — round-trip integrity (GHSA-25qj-68xc-22r7 hardening)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "hello",
        "$",
        "$$",
        "$$$",
        "$AGE1$",
        "$AGE1$ test",
        "price is $100 or $$200$$",
        'MATCH (n:base {entity_id: "x"}) RETURN n',
        # The exact GHSA-25qj-68xc-22r7 PoC payload embedded in a label.
        "test$$) AS (a agtype); SELECT version(); --",
        # Seam-collision regression: content ending in "$" + the candidate tag
        # used to complete a premature closing delimiter and truncate the body.
        "ends with $AGE1",
        "trailing tag $AGE2",
        "$AGE1",
    ],
)
def test_dollar_quote_round_trips_exactly(payload):
    """_dollar_quote output must decode back to the original content, nothing more.

    Non-empty trailing text would mean the literal closed early and the rest of
    the payload leaked into executable SQL — the core of the injection.
    """
    content, trailing = _parse_dollar_quoted(_dollar_quote(payload))
    assert content == payload
    assert trailing == ""


def test_dollar_quote_seam_collision_is_rejected():
    """A tag whose delimiter would form across the content/closing seam is skipped."""
    # "ends with $AGE1" ends with "$AGE1" == "$AGE1$"[:-1]; tag AGE1 must be
    # rejected in favour of a non-colliding tag (AGE2).
    quoted = _dollar_quote("ends with $AGE1")
    assert quoted == "$AGE2$ends with $AGE1$AGE2$"


def test_dollar_quote_delimiter_never_appears_inside_body():
    """The chosen delimiter must not occur anywhere inside the quoted content."""
    payload = "x$AGE1$AGE2$AGE3$y"  # forces several tag bumps
    quoted = _dollar_quote(payload)
    close_tag = quoted[: quoted.index("$", 1) + 1]
    body = quoted[len(close_tag) : -len(close_tag)]
    assert body == payload
    assert close_tag not in body


# ---------------------------------------------------------------------------
# get_knowledge_graph / _bfs_subgraph — the GHSA-25qj-68xc-22r7 sink
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bfs_subgraph_label_injection_is_contained():
    """The `label` read path must carry injection payloads as data, not SQL.

    This is the exact sink reported in GHSA-25qj-68xc-22r7: `/graphs?label=`
    → get_knowledge_graph → _bfs_subgraph → cypher(...). A `$$`/statement
    payload must stay inside the dollar-quoted literal.
    """
    storage = make_graph_storage()
    payload = "test$$) AS (a agtype); SELECT version(); --"

    sql = await _capture_bfs_subgraph_query(storage, payload)

    # The dangerous tokens must not survive into the executable SQL skeleton
    # once the dollar-quoted literals are removed.
    skeleton = _strip_dollar_literals(sql)
    assert "SELECT version()" not in skeleton
    assert "agtype)" in skeleton  # the legitimate AS (...) clause remains
    assert "version" not in skeleton
    # Sanity: the payload is present in the full query (carried as data).
    assert "version()" in sql


@pytest.mark.asyncio
async def test_bfs_subgraph_dollar_tag_collision_label_is_contained():
    """A label crafted to collide with the AGE tag scheme is still contained."""
    storage = make_graph_storage()
    # Attempt to pre-place the delimiter the sink would choose.
    payload = "$AGE1$ RETURN 1; DROP TABLE users; -- "

    sql = await _capture_bfs_subgraph_query(storage, payload)

    skeleton = _strip_dollar_literals(sql)
    assert "DROP TABLE" not in skeleton
    assert "RETURN 1" not in skeleton


@pytest.mark.asyncio
async def test_get_knowledge_graph_routes_label_through_dollar_quote():
    """Non-wildcard get_knowledge_graph must build its query via dollar-quoting."""
    storage = make_graph_storage()
    storage.global_config = {"max_graph_nodes": 1000}
    captured: list[str] = []

    async def fake_query(sql, **kwargs):
        captured.append(sql)
        return []

    with patch.object(storage, "_query", side_effect=fake_query):
        await storage.get_knowledge_graph(node_label="Alice$$; SELECT 1; --")

    assert captured
    # The starting-node lookup must never use the old `cypher('%s', $$ ... $$)`
    # static template with the label interpolated.
    assert "; SELECT 1" not in _strip_dollar_literals(captured[0])
