import pytest

from lightrag.base import QueryContextResult, QueryParam
from lightrag.operate import kg_query
from lightrag.utils import Tokenizer


class _Tokenizer:
    def encode(self, content):
        return [ord(char) for char in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


class _Cache:
    def __init__(self):
        self.global_config = {"enable_llm_cache": True}
        self.store = {}

    async def get_by_id(self, key):
        return self.store.get(key)

    async def upsert(self, entries):
        self.store.update(entries)


class _TextChunks:
    def __init__(self, global_config):
        self.global_config = global_config


class _Model:
    def __init__(self):
        self.calls = 0

    async def __call__(self, *_args, **_kwargs):
        self.calls += 1
        return f"answer-{self.calls}"


def _config(model):
    return {
        "tokenizer": Tokenizer("fake", _Tokenizer()),
        "role_llm_funcs": {"query": model},
        "addon_params": {"language": "en"},
        "related_chunk_number": 1,
        "kg_chunk_pick_method": "WEIGHT",
    }


async def _run(config, cache):
    return await kg_query(
        "query",
        None,
        None,
        None,
        _TextChunks(config),
        QueryParam(mode="local", enable_rerank=False, ll_keywords=["topic"]),
        config,
        hashing_kv=cache,
    )


@pytest.fixture
def stub_query_context(monkeypatch):
    async def fake_keywords(*_args, **_kwargs):
        return "", "topic"

    async def fake_context(*args, **_kwargs):
        config = args[6].global_config
        context = (
            f"chunks={config['related_chunk_number']};"
            f"method={config['kg_chunk_pick_method']}"
        )
        return QueryContextResult(context=context, raw_data={})

    monkeypatch.setattr("lightrag.operate.get_keywords_from_query", fake_keywords)
    monkeypatch.setattr("lightrag.operate._build_query_context", fake_context)


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "setting,first_value,second_value",
    [
        ("related_chunk_number", 1, 5),
        ("kg_chunk_pick_method", "WEIGHT", "VECTOR"),
    ],
)
async def test_kg_chunk_selection_partitions_answer_cache(
    setting, first_value, second_value, stub_query_context
):
    model = _Model()
    config = _config(model)
    cache = _Cache()

    config[setting] = first_value
    first = await _run(config, cache)
    config[setting] = second_value
    second = await _run(config, cache)

    assert first.content == "answer-1"
    assert second.content == "answer-2"
    assert model.calls == 2
    assert len([key for key in cache.store if ":query:" in key]) == 2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_same_kg_chunk_selection_still_hits_answer_cache(stub_query_context):
    model = _Model()
    config = _config(model)
    cache = _Cache()

    first = await _run(config, cache)
    second = await _run(config, cache)

    assert first.content == second.content == "answer-1"
    assert model.calls == 1
    assert len([key for key in cache.store if ":query:" in key]) == 1
