"""search_labels must send Chinese, Japanese and Korean queries as analyzed terms.

The full-text index uses Lucene's ``cjk`` analyzer, which indexes Han, kana and
Hangul as overlapping bigrams. A wildcard term is not split into bigrams, so a
bare ``삼성전자*`` is compared against 2-character terms such as ``삼성`` /
``전자`` and matches nothing — silently, so the CONTAINS fallback never runs
either. Only Han was routed to the CJK path, so Korean and kana-only label
searches of three or more characters returned ``[]`` against a real Neo4j 5.

The CJK path sends ``<query> <query>*``: the analyzed part matches queries of
two or more characters, and the prefix part keeps a single character working
(``서*`` matches the bigram ``서울``; the analyzed ``서`` matches nothing).
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
    def __init__(self, calls, fail_fulltext=False):
        self._calls = calls
        self._fail_fulltext = fail_fulltext

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run(self, query, **params):
        self._calls.append((query, params))
        if self._fail_fulltext and "db.index.fulltext.queryNodes" in query:
            raise RuntimeError("full-text index unavailable")
        return _FakeResult([])


class _FakeDriver:
    def __init__(self, calls, fail_fulltext=False):
        self._calls = calls
        self._fail_fulltext = fail_fulltext

    def session(self, **kwargs):
        return _FakeSession(self._calls, self._fail_fulltext)


def _make_storage(fail_fulltext=False):
    calls = []
    storage = Neo4JStorage(
        namespace="chunk_entity_relation",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        workspace="test",
    )
    storage._driver = _FakeDriver(calls, fail_fulltext)
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
        ("ＡＢＣ", False),  # Fullwidth Latin is not bigrammed by the analyzer
        ("", False),
    ],
)
def test_is_cjk_text(text, expected):
    storage, _ = _make_storage()
    assert storage._is_cjk_text(text) is expected


@pytest.mark.parametrize(
    "query",
    [
        "삼성전자",
        "서",
    ],
)
async def test_search_labels_sends_cjk_query_as_terms_plus_prefix(query):
    storage, calls = _make_storage()

    await storage.search_labels(query)

    assert len(calls) == 1
    assert calls[0][1]["search_query"] == f"{query} {query}*"


async def test_search_labels_keeps_prefix_wildcard_for_latin_query():
    storage, calls = _make_storage()

    await storage.search_labels("Samsung")

    assert len(calls) == 1
    assert calls[0][1]["search_query"] == "Samsung*"


async def test_fallback_matches_case_insensitively_for_every_script():
    """A mixed query routed to the CJK path keeps the lowercased CONTAINS.

    Before Hangul and kana were routed to the CJK path, `lg전자` reached the
    Latin fallback and matched a stored `LG전자` through toLower; the CJK
    fallback compared the raw query and returned nothing.
    """
    query = "lg전자"
    storage, calls = _make_storage(fail_fulltext=True)

    await storage.search_labels(query)

    assert len(calls) == 2  # full-text attempt, then the CONTAINS fallback
    fallback_query, fallback_params = calls[1]
    assert "toLower(n.entity_id) AS label_lower" in fallback_query
    assert "label_lower CONTAINS $query_lower" in fallback_query
    assert fallback_params == {"query_lower": query.lower(), "limit": 50}
