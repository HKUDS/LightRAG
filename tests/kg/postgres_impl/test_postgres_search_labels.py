"""PGGraphStorage.search_labels must treat the query as literal text.

The query was spliced into LIKE patterns unescaped, so ``%`` and ``_`` in it
acted as wildcards: ``a_b`` matched ``axb`` and ``50%`` matched every label
starting with ``50``. The word-boundary probe ``%_query%`` had the same flaw,
so its ``_`` matched any character and almost every contains match got the
+50 bonus. The bonus was also added on top of exact and prefix matches,
unlike NetworkXStorage and PGTableGraphStorage.
"""

import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from lightrag.kg.postgres_impl import PGGraphStorage

pytestmark = pytest.mark.offline


def make_graph_storage() -> PGGraphStorage:
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = "test_ws"
    storage.namespace = "test_graph"
    storage.graph_name = "test_graph"
    storage.__post_init__()
    storage.db = MagicMock()
    return storage


def like_matches(pattern: str, text: str) -> bool:
    """PostgreSQL LIKE with backslash as the escape character."""
    regex = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\" and i + 1 < len(pattern):
            regex.append(re.escape(pattern[i + 1]))
            i += 2
            continue
        if ch == "%":
            regex.append(".*")
        elif ch == "_":
            regex.append(".")
        else:
            regex.append(re.escape(ch))
        i += 1
    return re.fullmatch("".join(regex), text, flags=re.DOTALL) is not None


async def captured_search(query: str) -> tuple[str, list]:
    storage = make_graph_storage()
    storage._query = AsyncMock(return_value=[])
    await storage.search_labels(query)
    call = storage._query.await_args
    sql = call.args[0]
    params = list(call.kwargs["params"].values())
    return sql, params


@pytest.mark.asyncio
async def test_underscore_in_query_is_not_a_wildcard():
    sql, params = await captured_search("a_b")
    assert "ESCAPE" in sql

    contains = params[0]
    assert like_matches(contains, "a_b")
    assert like_matches(contains, "x a_b y")
    assert not like_matches(contains, "axb")


@pytest.mark.asyncio
async def test_percent_in_query_is_not_a_wildcard():
    _, params = await captured_search("50%")

    contains, prefix = params[0], params[2]
    assert like_matches(contains, "rate 50% off")
    assert not like_matches(contains, "500 units")
    assert like_matches(prefix, "50% off")
    assert not like_matches(prefix, "500 units")


@pytest.mark.asyncio
async def test_exact_match_parameter_is_the_raw_query():
    _, params = await captured_search("A_B%")
    assert params[1] == "a_b%"


@pytest.mark.asyncio
async def test_underscore_boundary_bonus_needs_a_literal_underscore():
    _, params = await captured_search("foo")

    space_boundary, underscore_boundary = params[3], params[4]
    assert like_matches(underscore_boundary, "xx_foo")
    assert not like_matches(underscore_boundary, "barfoo")
    assert like_matches(space_boundary, "bar foo")
    assert not like_matches(space_boundary, "barfoo")


@pytest.mark.asyncio
async def test_boundary_bonus_only_applies_to_contains_matches():
    """Exact and prefix matches keep their fixed 1000 / 500 scores."""
    sql, _ = await captured_search("foo")
    normalized = re.sub(r"\s+", " ", sql)
    assert re.search(
        r"ELSE \(100 - LENGTH\(label\)\) \+ CASE WHEN label_lower LIKE \$4",
        normalized,
    )
    assert not re.search(r"END \+ CASE", normalized)
