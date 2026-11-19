"""Offline hybrid-retrieval semantics: RRF fusion and ``hybrid_query``.

``rrf_fuse`` is pinned with hand-computed ranks: every expected order below
is derived from the RRF formula ``sum 1/(60+rank)`` and the documented
tie-break, never from the code under test. ``hybrid_query`` runs on a
``PGVectorStorage`` built via ``__new__`` with a fake db that dispatches by
SQL shape (the ``test_postgres_iterative_scan.py`` pattern), pinning the
distance sentinel, the empty-tokenization skip, fail-open degrade and its
once-per-failure-streak warning. The real SQL, index lifecycle and fusion
against live PostgreSQL are pinned in ``test_postgres_hybrid_query.py``.
"""

from __future__ import annotations

import copy

import pytest

from lightrag.kg.lexical_tokenizer import tokenize_for_query
from lightrag.kg.postgres_impl import PGVectorStorage, rrf_fuse

pytestmark = pytest.mark.offline


def _rows(key: str, *names: str) -> list[dict]:
    return [{key: name, "content": f"content-{name}"} for name in names]


# ---------------------------------------------------------------------------
# rrf_fuse: hand-computed ranks
# ---------------------------------------------------------------------------


def test_rrf_disjoint_legs_hand_computed_order():
    # vector: v1=1/61, v2=1/62. lexical: l1=1/61, l2=1/62.
    # v1/l1 tie -> vector rank present wins; v2/l2 tie -> same rule.
    vector = _rows("id", "v1", "v2")
    lexical = _rows("id", "l1", "l2")

    fused = rrf_fuse(vector, lexical, "id", top_k=4)

    assert [r["id"] for r in fused] == ["v1", "l1", "v2", "l2"]


def test_rrf_overlap_sums_both_ranks_and_keeps_vector_record():
    # a=1/61, b=1/62+1/61, c=1/62 -> b first. On overlap the VECTOR leg's
    # record object must win (it carries the original vector fields).
    vector = _rows("id", "a", "b")
    lexical = _rows("id", "b", "c")

    fused = rrf_fuse(vector, lexical, "id", top_k=3)

    assert [r["id"] for r in fused] == ["b", "a", "c"]
    assert fused[0] is vector[1]
    assert fused[0]["content"] == "content-b"


def test_rrf_swapped_overlap_ranks_break_tie_by_vector_rank():
    # a=1/61+1/62, b=1/62+1/61: equal scores, vector rank 1 < 2 -> a first.
    vector = _rows("id", "a", "b")
    lexical = _rows("id", "b", "a")

    fused = rrf_fuse(vector, lexical, "id", top_k=2)

    assert [r["id"] for r in fused] == ["a", "b"]


@pytest.mark.parametrize(
    ("top_k", "expected"),
    [(1, ["v1"]), (2, ["v1", "l1"]), (3, ["v1", "l1", "v2"]), (9, ["v1", "l1", "v2", "l2"])],
)
def test_rrf_returns_at_most_top_k_in_fusion_order(top_k, expected):
    fused = rrf_fuse(_rows("id", "v1", "v2"), _rows("id", "l1", "l2"), "id", top_k)
    assert [r["id"] for r in fused] == expected


def test_rrf_empty_lexical_leg_preserves_vector_order():
    vector = _rows("id", "a", "b", "c")
    assert rrf_fuse(vector, [], "id", top_k=3) == vector


def test_rrf_empty_vector_leg_preserves_lexical_order():
    lexical = _rows("id", "x", "y")
    assert rrf_fuse([], lexical, "id", top_k=2) == lexical


def test_rrf_lexical_only_ranks_order_by_score():
    # x=1/61 > y=1/62 > z=1/63, no vector leg to prefer.
    fused = rrf_fuse([], _rows("id", "x", "y", "z"), "id", top_k=3)
    assert [r["id"] for r in fused] == ["x", "y", "z"]


def test_rrf_never_mutates_inputs():
    vector = _rows("id", "a", "b")
    lexical = _rows("id", "b", "c")
    vector_before = copy.deepcopy(vector)
    lexical_before = copy.deepcopy(lexical)

    rrf_fuse(vector, lexical, "id", top_k=3)

    assert vector == vector_before
    assert lexical == lexical_before


def test_rrf_entity_name_key_field():
    vector = _rows("entity_name", "E1")
    lexical = _rows("entity_name", "E2", "E1")
    # E1=1/61+1/62 > E2=1/61.
    fused = rrf_fuse(vector, lexical, "entity_name", top_k=2)
    assert [r["entity_name"] for r in fused] == ["E1", "E2"]


# ---------------------------------------------------------------------------
# hybrid_query orchestration on a fake db
# ---------------------------------------------------------------------------


class _FakeDB:
    """Answers vector / index-validity / lexical SQL by shape, recording calls.

    An unexpected SQL shape raises AND is recorded, so a broad except in the
    code under test cannot hide a wrong statement.
    """

    def __init__(self, vector_rows, lexical_rows, validity):
        self.vector_rows = vector_rows
        self.lexical_rows = lexical_rows
        self.validity = validity
        self.calls: list[str] = []
        self.lexical_params: list | None = None
        self.lexical_sql: str | None = None
        self.vector_error: Exception | None = None
        self.lexical_error: Exception | None = None

    async def query(self, sql, params=None, multirows=False, **kwargs):
        if "content_vector <=>" in sql:
            self.calls.append("vector")
            if self.vector_error is not None:
                raise self.vector_error
            return self.vector_rows
        if "indisvalid" in sql:
            self.calls.append("validity")
            return self.validity
        if "ts_rank_cd" in sql:
            self.calls.append("lexical")
            self.lexical_params = list(params) if params else None
            self.lexical_sql = sql
            if self.lexical_error is not None:
                raise self.lexical_error
            return self.lexical_rows
        self.calls.append(f"unexpected:{sql}")
        raise AssertionError(f"unexpected SQL shape: {sql}")


def _storage(db: _FakeDB, namespace: str = "chunks") -> PGVectorStorage:
    storage = PGVectorStorage.__new__(PGVectorStorage)
    storage.namespace = namespace
    storage.workspace = "gw"
    storage.table_name = "t"
    storage.cosine_better_than_threshold = 0.2
    storage._lexical_degrade_warned = False
    storage.db = db
    return storage


_QUERY = "乐高机器人"
_EMBEDDING = [0.1] * 8


async def test_hybrid_fuses_legs_and_sentinels_lexical_only_records():
    vector = [{"id": "v1", "content": "V1"}, {"id": "v2", "content": "V2"}]
    lexical = [{"id": "l1", "content": "L1"}, {"id": "v1", "content": "L1-dup"}]
    db = _FakeDB(vector, lexical, {"indisvalid": True})
    storage = _storage(db)

    fused = await storage.hybrid_query(_QUERY, top_k=4, query_embedding=_EMBEDDING)

    # v1=1/61+1/62, l1=1/61, v2=1/62.
    assert [r["id"] for r in fused] == ["v1", "l1", "v2"]
    assert fused[0] is vector[0]
    assert "distance" not in fused[0]
    assert "distance" not in fused[2]
    assert fused[1]["distance"] == 0.2
    assert db.calls == ["vector", "validity", "lexical"]
    # The tsquery text is a BIND parameter, never interpolated into the SQL.
    assert db.lexical_params == ["gw", tokenize_for_query(_QUERY), 4]
    assert "$2" in db.lexical_sql
    assert tokenize_for_query(_QUERY) not in db.lexical_sql


async def test_hybrid_entities_namespace_uses_entity_name_key():
    vector = [{"entity_name": "E1", "created_at": 1}]
    lexical = [{"entity_name": "E2", "created_at": 2}]
    db = _FakeDB(vector, lexical, {"indisvalid": True})
    storage = _storage(db, namespace="entities")

    fused = await storage.hybrid_query(_QUERY, top_k=2, query_embedding=_EMBEDDING)

    assert [r["entity_name"] for r in fused] == ["E1", "E2"]
    assert fused[1]["distance"] == 0.2
    assert "distance" not in fused[0]
    assert "SELECT entity_name" in db.lexical_sql


@pytest.mark.parametrize("query", ["", "   ", "!&|", "---"], ids=["empty", "spaces", "operators", "punct"])
async def test_hybrid_empty_tokenization_skips_lexical_leg(query, lightrag_log_records):
    vector = [{"id": "v1", "content": "V1"}]
    db = _FakeDB(vector, [], {"indisvalid": True})
    storage = _storage(db)

    result = await storage.hybrid_query(query, top_k=3, query_embedding=_EMBEDDING)

    assert result is vector
    assert db.calls == ["vector"]
    assert not any("degraded" in r.getMessage() for r in lightrag_log_records)


@pytest.mark.parametrize(
    "validity",
    [None, {"indisvalid": False}],
    ids=["missing", "invalid"],
)
async def test_hybrid_degrades_to_vector_only_on_unusable_index(
    validity, lightrag_log_records
):
    vector = [{"id": "v1", "content": "V1"}]
    db = _FakeDB(vector, [{"id": "l1", "content": "L1"}], validity)
    storage = _storage(db)

    result = await storage.hybrid_query(_QUERY, top_k=3, query_embedding=_EMBEDDING)

    assert result is vector
    assert db.calls == ["vector", "validity"]
    warnings = [r for r in lightrag_log_records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "missing or invalid" in warnings[0].getMessage()


async def test_hybrid_degrades_on_lexical_query_error(lightrag_log_records):
    vector = [{"id": "v1", "content": "V1"}]
    db = _FakeDB(vector, [], {"indisvalid": True})
    db.lexical_error = RuntimeError("injected lexical failure")
    storage = _storage(db)

    result = await storage.hybrid_query(_QUERY, top_k=3, query_embedding=_EMBEDDING)

    assert result is vector
    # The injected fault actually fired: the lexical statement was reached.
    assert db.calls == ["vector", "validity", "lexical"]
    warnings = [r for r in lightrag_log_records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "injected lexical failure" in warnings[0].getMessage()


async def test_hybrid_degrade_warning_dedup_and_rearm(lightrag_log_records):
    vector = [{"id": "v1", "content": "V1"}]
    lexical = [{"id": "l1", "content": "L1"}]
    db = _FakeDB(vector, lexical, None)
    storage = _storage(db)

    for _ in range(3):
        assert await storage.hybrid_query(
            _QUERY, top_k=3, query_embedding=_EMBEDDING
        ) is vector
    degraded_warnings = [
        r for r in lightrag_log_records if "degraded to vector-only" in r.getMessage()
    ]
    assert len(degraded_warnings) == 1

    # A successful lexical leg re-arms the warning without logging.
    db.validity = {"indisvalid": True}
    healed = await storage.hybrid_query(_QUERY, top_k=3, query_embedding=_EMBEDDING)
    assert [r["id"] for r in healed] == ["v1", "l1"]
    assert len(degraded_warnings) == 1

    # The next failure streak warns again.
    db.validity = None
    await storage.hybrid_query(_QUERY, top_k=3, query_embedding=_EMBEDDING)
    degraded_warnings = [
        r for r in lightrag_log_records if "degraded to vector-only" in r.getMessage()
    ]
    assert len(degraded_warnings) == 2


async def test_hybrid_vector_leg_error_propagates(lightrag_log_records):
    db = _FakeDB([], [], {"indisvalid": True})
    db.vector_error = RuntimeError("vector backend unavailable")
    storage = _storage(db)

    with pytest.raises(RuntimeError, match="vector backend unavailable"):
        await storage.hybrid_query(_QUERY, top_k=3, query_embedding=_EMBEDDING)

    assert db.calls == ["vector"]
    assert not any("degraded" in r.getMessage() for r in lightrag_log_records)


async def test_hybrid_relationships_namespace_is_out_of_scope():
    db = _FakeDB([], [], {"indisvalid": True})
    storage = _storage(db, namespace="relationships")

    with pytest.raises(ValueError, match="covers entities and chunks"):
        await storage.hybrid_query(_QUERY, top_k=3, query_embedding=_EMBEDDING)

    assert db.calls == []


async def test_hybrid_returns_top_k_of_the_fused_union():
    vector = _rows("id", "v1", "v2", "v3")
    lexical = _rows("id", "l1", "l2", "l3")
    db = _FakeDB(vector, lexical, {"indisvalid": True})
    storage = _storage(db)

    fused = await storage.hybrid_query(_QUERY, top_k=2, query_embedding=_EMBEDDING)

    assert [r["id"] for r in fused] == ["v1", "l1"]
    assert db.lexical_params[2] == 2
