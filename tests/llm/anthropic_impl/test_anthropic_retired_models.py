"""The claude_3_* helpers name models Anthropic has retired.

Requests to a retired model fail with 404 not_found_error, so each helper
warns, says which model and when it was retired, and points to
anthropic_complete. The call itself is unchanged.
"""

import pytest

import lightrag.llm.anthropic as anthropic_mod

pytestmark = pytest.mark.offline

_RETIRED = [
    ("claude_3_opus_complete", "claude-3-opus-20240229", "2026-01-05"),
    ("claude_3_sonnet_complete", "claude-3-sonnet-20240229", "2025-07-21"),
    ("claude_3_haiku_complete", "claude-3-haiku-20240307", "2026-04-20"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("helper,model,retired", _RETIRED, ids=[r[0] for r in _RETIRED])
async def test_retired_claude_3_helper_warns_and_still_forwards(
    monkeypatch, helper, model, retired
):
    calls = []

    async def _record(model_name, prompt, **kwargs):
        calls.append(model_name)
        return "ok"

    monkeypatch.setattr(anthropic_mod, "anthropic_complete_if_cache", _record)

    with pytest.warns(DeprecationWarning) as record:
        assert await getattr(anthropic_mod, helper)("prompt") == "ok"

    message = str(record[0].message)
    assert model in message
    assert retired in message
    assert "anthropic_complete" in message
    assert record[0].filename == __file__
    assert calls == [model]
