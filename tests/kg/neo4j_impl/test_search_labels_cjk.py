"""search_labels must not append a wildcard to Chinese, Japanese or Korean queries.

The full-text index uses Lucene's ``cjk`` analyzer, which indexes Han, kana and
Hangul as overlapping bigrams. A wildcard term bypasses the analyzer, so
``삼성전자*`` is compared against 2-character terms such as ``삼성`` / ``전자``
and matches nothing — silently, so the CONTAINS fallback never runs either.
Only Han was routed to the no-wildcard path, so Korean and kana-only label
searches of three or more characters returned ``[]`` against a real Neo4j 5.
"""

import pytest

from lightrag.kg.neo4j_impl import Neo4JStorage


pytestmark = pytest.mark.offline


class _FakeResult:
    def __init__(self, records):
        self._records = list(records)

    def __aiter__(self):
        self._iter = iter(self._records)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration

    async def consume(self):
        return None


class _FakeSession:
    def __init__(self, calls):
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run(self, query, **params):
        self._calls.append((query, params))
        return _FakeResult([])


class _FakeDriver:
    def __init__(self, calls):
        self._calls = calls

    def session(self, **kwargs):
        return _FakeSession(self._calls)


def _make_storage():
    calls = []
    storage = Neo4JStorage(
        namespace="chunk_entity_relation",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        workspace="test",
    )
    storage._driver = _FakeDriver(calls)
    storage._DATABASE = "neo4j"
    return storage, calls


@pytest.mark.parametrize(
    "text, expected",
    [
        ("北京大学", True),
        ("삼성전자", True),  # Hangul syllables
        ("LG전자", True),  # mixed Latin + Hangul
        ("ㅅㅅ", True),  # Hangul compatibility jamo
        ("タワー", True),  # Katakana only
        ("ひらがな", True),  # Hiragana only
        ("ﾀﾜｰ", True),  # Halfwidth Katakana
        ("𠀀", True),  # CJK Extension B (supplementary plane)
        ("Samsung", False),
        ("machine learning", False),
        ("ＡＢＣ", False),  # Fullwidth Latin is not bigrammed by the analyzer
        ("", False),
    ],
)
def test_is_cjk_text(text, expected):
    storage, _ = _make_storage()
    assert storage._is_cjk_text(text) is expected


@pytest.mark.parametrize("query", ["삼성전자", "현대자동차", "タワー", "北京大学"])
async def test_search_labels_sends_cjk_query_without_wildcard(query):
    storage, calls = _make_storage()

    await storage.search_labels(query)

    assert len(calls) == 1
    assert calls[0][1]["search_query"] == query


async def test_search_labels_keeps_prefix_wildcard_for_latin_query():
    storage, calls = _make_storage()

    await storage.search_labels("Samsung")

    assert len(calls) == 1
    assert calls[0][1]["search_query"] == "Samsung*"
