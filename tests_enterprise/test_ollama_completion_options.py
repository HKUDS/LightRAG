import pytest

from lightrag.llm import ollama as ollama_module


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("options", "expected_num_predict"),
    [
        (None, 120),
        ({"temperature": 0.2}, 120),
        ({"num_predict": 200}, 120),
        ({"num_predict": 80}, 80),
    ],
)
async def test_ollama_completion_maps_max_tokens_to_num_predict(
    monkeypatch,
    options,
    expected_num_predict,
):
    recorded: dict = {}

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            self._client = self

        async def chat(self, **kwargs):
            recorded.update(kwargs)
            return {"message": {"content": "ok"}}

        async def aclose(self):
            return None

    monkeypatch.setattr(ollama_module.ollama, "AsyncClient", FakeAsyncClient)

    response = await ollama_module._ollama_model_if_cache(
        "qwen-local",
        "Explique com fontes.",
        max_tokens=120,
        options=options,
    )

    assert response == "ok"
    assert recorded["options"]["num_predict"] == expected_num_predict
    assert "max_tokens" not in recorded
