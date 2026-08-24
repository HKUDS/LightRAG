"""Offline tests for LoLLMs' generic output-token alias."""

import pytest

from lightrag.llm.lollms import lollms_model_if_cache

pytestmark = pytest.mark.offline

sent_requests = []


class FakeResponse:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False

    async def text(self):
        return "ok"


class FakeSession:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False

    def post(self, url, json):
        sent_requests.append(json)
        return FakeResponse()


async def _sent_request(monkeypatch, **generation_kwargs):
    sent_requests.clear()
    monkeypatch.setattr("lightrag.llm.lollms.aiohttp.ClientSession", FakeSession)

    result = await lollms_model_if_cache(
        model="test-model",
        prompt="hello",
        **generation_kwargs,
    )

    assert result == "ok"
    assert len(sent_requests) == 1
    return sent_requests[0]


@pytest.mark.parametrize(
    ("generation_kwargs", "expected_n_predict"),
    [
        pytest.param({}, None, id="provider-default-is-preserved"),
        pytest.param(
            {"max_tokens": 37},
            37,
            id="generic-limit-maps-to-native-field",
        ),
        pytest.param(
            {"n_predict": 41},
            41,
            id="native-limit-is-preserved",
        ),
        pytest.param(
            {"max_tokens": 37, "n_predict": 41},
            41,
            id="native-limit-takes-precedence",
        ),
        pytest.param(
            {"max_tokens": 37, "n_predict": 0},
            0,
            id="explicit-native-zero-takes-precedence",
        ),
        pytest.param(
            {"max_tokens": 37, "n_predict": None},
            37,
            id="native-none-falls-back-to-generic-limit",
        ),
    ],
)
async def test_max_tokens_alias_precedence(
    monkeypatch, generation_kwargs, expected_n_predict
):
    request = await _sent_request(monkeypatch, **generation_kwargs)

    assert request["n_predict"] == expected_n_predict
    assert "max_tokens" not in request
