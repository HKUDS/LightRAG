"""hf_model_if_cache()'s ``@retry`` decorator did not set ``reraise=True``.

Once ``stop_after_attempt(3)`` was exhausted on a matching exception
(``RateLimitError`` / ``APIConnectionError`` / ``APITimeoutError`` from
``lightrag.exceptions`` -- the types ``retry_if_exception_type`` names),
tenacity's default behaviour wraps the last attempt's exception in an opaque
``tenacity.RetryError`` instead of re-raising it. A caller that wants to
catch the real error (to branch on it, log its message, or match on type)
could not, and the actual failure's message -- e.g. "Request timed out." --
was swallowed behind a generic "RetryError[...]" repr.

This binding runs inference entirely locally (no HTTP client), so the
exception path exercised here is not a real API wrapper raising these
types organically -- it is whatever this retry predicate is actually
declared to catch, which is the contract under test. The fix is adding
``reraise=True`` to the decorator, mirroring the same fix already applied
to ``lightrag/llm/nvidia_openai.py`` and ``lightrag/llm/voyageai.py``.

lightrag/llm/hf.py imports transformers and torch at module level. Neither
is a project dependency, so both are stubbed here, same approach as the
sibling hf tests in this directory.
"""

from __future__ import annotations

import sys
import types
import importlib
from unittest.mock import AsyncMock

import pytest

pytestmark = pytest.mark.offline


def install_fake_transformers_and_torch(monkeypatch):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = object
    fake_transformers.AutoModelForCausalLM = object
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    fake_torch.device = lambda name: name
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    import pipmaster as pm

    monkeypatch.setattr(pm, "is_installed", lambda name: True)


@pytest.fixture
def hf_module(monkeypatch):
    # Teardown belongs to this directory's autouse conftest fixture.
    install_fake_transformers_and_torch(monkeypatch)
    sys.modules.pop("lightrag.llm.hf", None)
    return importlib.import_module("lightrag.llm.hf")


@pytest.mark.asyncio
async def test_hf_model_if_cache_reraises_original_error_after_retries(
    hf_module, monkeypatch
):
    """Exhausted retries must surface the real exception, not RetryError.

    ``_run_hf_inference`` is what ``hf_model_if_cache`` awaits to run
    generation off the event loop thread, so raising from it exercises the
    same retry boundary a real timeout/connection failure would hit.
    """
    from lightrag.exceptions import APIConnectionError

    error = APIConnectionError(message="connection reset", request=None)
    fake_run_hf_inference = AsyncMock(side_effect=error)
    monkeypatch.setattr(hf_module, "_run_hf_inference", fake_run_hf_inference)
    monkeypatch.setattr(hf_module.hf_model_if_cache.retry, "sleep", AsyncMock())

    with pytest.raises(APIConnectionError) as exc_info:
        await hf_module.hf_model_if_cache("fake-model", "hello")

    assert exc_info.value is error
    assert fake_run_hf_inference.await_count == 3
