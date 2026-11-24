import json
import runpy
import sys
from pathlib import Path

import httpx
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "generate_query.py"


@pytest.mark.offline
@pytest.mark.parametrize(
    ("args", "model", "reasoning_effort"),
    [
        ([], "gpt-6-luna", "none"),
        (["--model", "gpt-4o-mini"], "gpt-4o-mini", None),
    ],
)
def test_generate_query_uses_selected_model(
    monkeypatch, tmp_path, args, model, reasoning_effort
):
    openai = pytest.importorskip("openai")
    requests = []

    def respond(request):
        assert request.url.path == "/v1/chat/completions"
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-generated-queries",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Sample question"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    with openai.OpenAI(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        monkeypatch.setattr(openai, "OpenAI", lambda: client)
        monkeypatch.setattr(sys, "argv", [str(SCRIPT), *args])
        monkeypatch.chdir(tmp_path)
        runpy.run_path(str(SCRIPT), run_name="__main__")

    assert len(requests) == 1
    assert requests[0]["model"] == model
    assert requests[0].get("reasoning_effort") == reasoning_effort
    assert "generate 5 questions" in requests[0]["messages"][0]["content"]
    assert (tmp_path / "queries.txt").read_text() == "Sample question"
