"""What lmdeploy_model_if_cache hands to lmdeploy and what it returns.

``do_sample`` defaults to False (greedy decoding) as the docstring documents;
it used to be overwritten to True on lmdeploy >= 0.6.0.
"""

import warnings
from types import SimpleNamespace

import pytest

from lightrag.llm.lmdeploy import lmdeploy_model_if_cache
from lightrag.utils import TruncatedResponse

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]

_DEFAULT_GEN_CONFIG = {"skip_special_tokens": True, "max_new_tokens": 512}


@pytest.mark.parametrize(
    ("call_kwargs", "expected_gen_config", "expected_warnings"),
    [
        pytest.param({}, {"do_sample": False}, [], id="defaults"),
        pytest.param(
            {"do_sample": True}, {"do_sample": True}, [], id="do-sample-forwarded"
        ),
        pytest.param(
            {"max_tokens": 37},
            {"do_sample": False, "max_new_tokens": 37},
            [],
            id="generic-max-tokens",
        ),
        pytest.param(
            {"max_tokens": 37, "max_new_tokens": 41},
            {"do_sample": False, "max_new_tokens": 41},
            [],
            id="native-max-new-tokens-wins",
        ),
        pytest.param(
            {
                "hashing_kv": object(),
                "response_format": {"type": "json_object"},
                "keyword_extraction": True,
                "entity_extraction": True,
            },
            {"do_sample": False},
            ["entity_extraction=True", "keyword_extraction=True"],
            id="lightrag-only-kwargs-dropped",
        ),
    ],
)
async def test_generation_config_kwargs(
    fake_lmdeploy, call_kwargs, expected_gen_config, expected_warnings
):
    """GenerationConfig receives exactly the generation settings; LightRAG's
    own kwargs never reach it, and the deprecated extraction flags warn."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await lmdeploy_model_if_cache(
            model="lmdeploy-model", prompt="hello", **call_kwargs
        )

    assert result == "generated text"
    assert fake_lmdeploy.gen_config == _DEFAULT_GEN_CONFIG | expected_gen_config
    messages = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(messages) == len(expected_warnings), messages
    for fragment in expected_warnings:
        assert any(fragment in message for message in messages), messages


async def test_pre_0_6_0_version_guard_raises(fake_lmdeploy):
    """lmdeploy < 0.6.0 cannot take do_sample, which is always set."""
    fake_lmdeploy.module.version_info = (0, 5, 3)

    with pytest.raises(RuntimeError, match="do_sample.*not supported"):
        await lmdeploy_model_if_cache(model="lmdeploy-model", prompt="hello")


@pytest.mark.parametrize(
    ("output", "expected_type", "expected_text"),
    [
        pytest.param(
            {"text": "natural stop", "finish_reason": "stop"},
            str,
            "natural stop",
            id="natural-stop",
        ),
        pytest.param(
            {"text": "cut off", "finish_reason": "length"},
            TruncatedResponse,
            "cut off",
            id="token-limit-truncated",
        ),
        pytest.param(
            {"response": "legacy text", "finish_reason": "stop"},
            str,
            "legacy text",
            id="legacy-response-attribute",
        ),
    ],
)
async def test_generated_output(fake_lmdeploy, output, expected_type, expected_text):
    async def fake_generate(*_args, **_kwargs):
        yield SimpleNamespace(**output)

    fake_lmdeploy.generate = fake_generate

    result = await lmdeploy_model_if_cache(model="lmdeploy-model", prompt="hello")

    assert type(result) is expected_type
    assert result == expected_text
