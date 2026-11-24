"""How the API server resolves each role's LLM timeout from the environment.

``{ROLE}_LLM_TIMEOUT`` overrides ``LLM_TIMEOUT``; an explicit 0 means "no
timeout" and must not fall through to the base value.
"""

import pytest

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    ("query_timeout_env", "expected"),
    [
        pytest.param(None, 240, id="unset-inherits-base"),
        pytest.param("120", 120, id="explicit-override"),
        pytest.param("0", 0, id="zero-means-no-timeout"),
    ],
)
def test_role_timeout_resolution(prepare_startup, query_timeout_env, expected):
    startup = prepare_startup(
        {
            "LLM_TIMEOUT": "240",
            "QUERY_LLM_TIMEOUT": query_timeout_env,
            "EXTRACT_LLM_TIMEOUT": None,
        }
    )

    role_configs = startup.constructor_kwargs()["role_llm_configs"]

    assert role_configs["query"].timeout == expected
    # Roles without an override keep the base value.
    assert role_configs["extract"].timeout == 240
