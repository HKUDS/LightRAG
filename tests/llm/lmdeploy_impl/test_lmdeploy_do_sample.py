"""Regression coverage for lmdeploy_model_if_cache's do_sample handling.

The function's own docstring documents ``do_sample`` defaulting to False for
greedy decoding, but the generation branch for lmdeploy >= 0.6.0 used to
unconditionally overwrite it to True regardless of what was passed in (or
its documented default) -- see lightrag/llm/lmdeploy.py.

Imports only from lightrag.llm.lmdeploy directly (not via
tests/extraction/test_keyword_extraction_drivers.py, which also imports
lightrag.llm.ollama at module level -- that import fails in environments
without the real `ollama` package installed, which would make these tests
uncollectable for a reason unrelated to what they're testing).
"""

import sys
from types import SimpleNamespace

import pytest

from lightrag.llm.lmdeploy import lmdeploy_model_if_cache

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


def _install_fake_lmdeploy(monkeypatch, *, less_than_0_6_0: bool) -> dict:
    """Install a fake `lmdeploy` module and pipeline, mirroring the mock
    setup already proven in test_keyword_extraction_drivers.py::
    test_lmdeploy_strips_response_format_before_generation_config.

    version_info is a plain tuple, matching lmdeploy's real attribute --
    `version < (0, 6, 0)` works natively without a fake comparable object.

    Returns the dict FakeGenerationConfig populates with its kwargs.
    """
    captured_gen_config_kwargs: dict = {}

    class FakeGenerationConfig:
        def __init__(self, **kwargs):
            captured_gen_config_kwargs.update(kwargs)

    async def fake_generate(*_args, **_kwargs):
        yield SimpleNamespace(response="{}")

    monkeypatch.setattr(
        "lightrag.llm.lmdeploy.initialize_lmdeploy_pipeline",
        lambda **_kwargs: SimpleNamespace(generate=fake_generate),
    )
    version_info = (0, 5, 3) if less_than_0_6_0 else (0, 6, 1)
    monkeypatch.setitem(
        sys.modules,
        "lmdeploy",
        SimpleNamespace(
            __version__=".".join(map(str, version_info)),
            version_info=version_info,
            GenerationConfig=FakeGenerationConfig,
        ),
    )
    return captured_gen_config_kwargs


@pytest.mark.parametrize(
    "call_kwargs,expected_do_sample",
    [
        pytest.param({}, False, id="omitted-defaults-to-false"),
        pytest.param({"do_sample": False}, False, id="explicit-false-preserved"),
        pytest.param({"do_sample": True}, True, id="explicit-true-preserved"),
    ],
)
async def test_do_sample_forwarded_unchanged_on_modern_lmdeploy(
    monkeypatch, call_kwargs, expected_do_sample
):
    """On lmdeploy >= 0.6.0, do_sample must reach GenerationConfig exactly
    as the caller passed it (or its documented False default), not be
    silently overwritten to True."""
    captured = _install_fake_lmdeploy(monkeypatch, less_than_0_6_0=False)

    result = await lmdeploy_model_if_cache(
        model="lmdeploy-model",
        prompt="hello",
        **call_kwargs,
    )

    assert result == "{}"
    assert captured["do_sample"] is expected_do_sample


@pytest.mark.parametrize(
    ("generation_kwargs", "expected_max_new_tokens"),
    [
        pytest.param({}, 512, id="omitted-defaults-to-512"),
        pytest.param({"max_tokens": 37}, 37, id="generic-max-tokens"),
        pytest.param({"max_new_tokens": 41}, 41, id="native-max-new-tokens"),
        pytest.param(
            {"max_tokens": 37, "max_new_tokens": 41},
            41,
            id="native-value-takes-precedence",
        ),
    ],
)
async def test_generation_token_limit_precedence(
    monkeypatch, generation_kwargs, expected_max_new_tokens
):
    """The native setting wins over LightRAG's generic alias."""
    captured = _install_fake_lmdeploy(monkeypatch, less_than_0_6_0=False)

    result = await lmdeploy_model_if_cache(
        model="lmdeploy-model",
        prompt="hello",
        **generation_kwargs,
    )

    assert result == "{}"
    assert captured["max_new_tokens"] == expected_max_new_tokens
    assert "max_tokens" not in captured


@pytest.mark.parametrize(
    "do_sample_kwarg",
    [
        pytest.param({}, id="omitted"),
        pytest.param({"do_sample": False}, id="explicit-false"),
        pytest.param({"do_sample": True}, id="explicit-true"),
    ],
)
async def test_pre_0_6_0_version_guard_still_raises(monkeypatch, do_sample_kwarg):
    """Control: the pre-existing version guard for lmdeploy < 0.6.0 must be
    unaffected by the do_sample fix -- it should still refuse to proceed,
    exactly as before, regardless of what (if anything) do_sample was set
    to."""
    _install_fake_lmdeploy(monkeypatch, less_than_0_6_0=True)

    with pytest.raises(RuntimeError, match="do_sample.*not supported"):
        await lmdeploy_model_if_cache(
            model="lmdeploy-model",
            prompt="hello",
            **do_sample_kwarg,
        )
