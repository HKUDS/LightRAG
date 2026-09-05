"""Config wiring for the openai_responses binding.

``create_llm_model_func()`` ends in a bare ``else:  # openai and compatible``,
so a binding missing from any one of the exact in-list checks does not raise —
it silently routes to ``/v1/chat/completions``. These tests pin the wiring that
prevents that, and pin the namespace separation that keeps Chat Completions
options from leaking into the Responses driver.
"""

from __future__ import annotations

import sys

import pytest

from lightrag.api.config import get_default_host, parse_args
from lightrag.llm.binding_options import OpenAIResponsesLLMOptions

pytestmark = pytest.mark.offline

# Env that a developer-local .env could otherwise use to decide the outcome.
_AMBIENT_ENV = (
    "LLM_BINDING",
    "LLM_BINDING_HOST",
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


def test_default_host_is_the_openai_endpoint(clean_env):
    # Without this mapping the binding falls back to the catch-all Ollama
    # default (http://localhost:11434) and every request fails at connect time.
    assert get_default_host("openai_responses") == "https://api.openai.com/v1"


def test_binding_host_override_is_honoured(clean_env):
    clean_env.setenv("LLM_BINDING_HOST", "https://gateway.example/v1")

    assert get_default_host("openai_responses") == "https://gateway.example/v1"


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


def test_the_responses_binding_does_not_reach_the_chat_completions_fallthrough(
    clean_env,
):
    """The dispatch branch must precede the bare Chat Completions ``else``.

    Adding the binding to the argparse choices but not to the dispatch chain
    produces no error at all — requests just go to the wrong endpoint. This is
    the guard the issue explicitly asked for.
    """
    import inspect

    from lightrag.api import lightrag_server

    source = inspect.getsource(lightrag_server)

    # There are two `else:  # openai and compatible` fallthroughs: the earlier
    # one is the *embedding* dispatch, which this binding never reaches because
    # openai_responses is not an embedding choice (the Responses API does not
    # replace /v1/embeddings). The LLM dispatch is the later one.
    dispatch_index = source.index('elif binding == "openai_responses":')
    llm_fallthrough_index = source.rindex("else:  # openai and compatible")

    assert dispatch_index < llm_fallthrough_index, (
        "the openai_responses branch must precede the Chat Completions "
        "fallthrough, or Responses traffic is silently sent to "
        "/v1/chat/completions"
    )
    # Pin the assumption above: if a third fallthrough is ever added, this
    # test's rindex could start pointing at the wrong one.
    assert source.count("else:  # openai and compatible") == 2


def test_binding_is_in_the_server_startup_allow_list(clean_env):
    """Startup rejects any binding absent from its allow-list."""
    import inspect

    from lightrag.api import lightrag_server

    source = inspect.getsource(lightrag_server)
    allow_list_start = source.index("if args.llm_binding not in [")
    allow_list_end = source.index('raise Exception("llm binding not supported")')
    allow_list = source[allow_list_start:allow_list_end]

    assert '"openai_responses"' in allow_list
