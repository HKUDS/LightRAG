"""Config wiring for the openai_responses binding.

``create_llm_model_func()`` ends in a bare ``else:  # openai and compatible``,
so a binding missing from any one of the exact in-list checks does not raise —
it silently routes to ``/v1/chat/completions``. These tests pin the wiring that
prevents that, and pin the namespace separation that keeps Chat Completions
options from leaking into the Responses driver.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from lightrag.api.config import parse_args
from lightrag.llm.binding_options import OpenAIResponsesLLMOptions

pytestmark = pytest.mark.offline

# Env that a developer-local .env could otherwise use to decide the outcome.
_AMBIENT_ENV = (
    "LLM_BINDING",
    "VLM_PROCESS_ENABLE",
    "OPENAI_LLM_MAX_COMPLETION_TOKENS",
    "OPENAI_LLM_REASONING_EFFORT",
    "OPENAI_RESPONSES_LLM_MAX_OUTPUT_TOKENS",
    "OPENAI_RESPONSES_LLM_REASONING_EFFORT",
)


@pytest.fixture
def clean_env(monkeypatch):
    """Pin argv and clear ambient binding env.

    parse_args() reads sys.argv, and importing the server module runs a
    module-level parse_args(), so argv must be pinned before either happens.
    """
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    for name in _AMBIENT_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_binding_is_an_accepted_llm_binding_choice(clean_env):
    clean_env.setenv("LLM_BINDING", "openai_responses")

    # An unlisted value would make argparse exit with SystemExit(2).
    assert parse_args().llm_binding == "openai_responses"


def test_options_use_their_own_env_namespace(clean_env):
    clean_env.setenv("LLM_BINDING", "openai_responses")
    clean_env.setenv("OPENAI_RESPONSES_LLM_MAX_OUTPUT_TOKENS", "4096")

    options = OpenAIResponsesLLMOptions.options_dict(parse_args())

    assert options["max_output_tokens"] == 4096


def test_chat_completions_options_do_not_leak_into_responses(clean_env):
    """OPENAI_LLM_* must not reach the Responses driver.

    ``max_completion_tokens`` does not exist on the Responses API, and a flat
    ``reasoning_effort`` is rejected where a nested ``reasoning`` object is
    expected — so a leak is a request-level failure, not a cosmetic one.
    """
    clean_env.setenv("LLM_BINDING", "openai_responses")
    clean_env.setenv("OPENAI_LLM_MAX_COMPLETION_TOKENS", "999")

    options = OpenAIResponsesLLMOptions.options_dict(parse_args())

    assert "max_completion_tokens" not in options
    assert options.get("max_output_tokens") != 999


def test_startup_routes_the_binding_to_the_responses_driver(clean_env, prepare_startup):
    """Startup accepts the binding and its LLM function calls the Responses API.

    Adding the binding to the argparse choices but not to the startup
    allow-list refuses to start, and missing it from the dispatch chain sends
    the traffic to /v1/chat/completions with no error at all.
    """
    import lightrag.llm.openai as chat_completions
    import lightrag.llm.openai_responses as responses

    calls: list[str] = []

    async def fake_responses(model, prompt, **kwargs):
        calls.append("responses")
        return "ok"

    async def fake_chat_completions(model, prompt, **kwargs):
        calls.append("chat_completions")
        return "ok"

    clean_env.setattr(responses, "openai_responses_complete_if_cache", fake_responses)
    clean_env.setattr(
        chat_completions, "openai_complete_if_cache", fake_chat_completions
    )
    startup = prepare_startup(
        {"LLM_BINDING": "openai_responses", "LLM_BINDING_API_KEY": "a-key"}
    )

    llm_model_func = startup.constructor_kwargs()["llm_model_func"]
    asyncio.run(llm_model_func("hello"))

    assert calls == ["responses"]
