"""Tests for ``lightrag.api.asgi_helpers``."""

from __future__ import annotations

from fastapi.security.utils import get_authorization_scheme_param
import pytest

from lightrag.api.asgi_helpers import bearer_token

pytestmark = pytest.mark.offline


def _scope(authorization: str | None) -> dict:
    if authorization is None:
        return {"headers": []}
    return {"headers": [(b"authorization", authorization.encode("latin-1"))]}


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Bearer abc", "abc"),
        ("bearer abc", "abc"),
        ("Bearer  abc", "abc"),
        ("Bearer   abc   ", "abc"),
        ("Bearer\tabc", None),
        ("Bearer   ", None),
        ("Basic abc", None),
        ("", None),
        (None, None),
    ],
)
def test_bearer_token_matches_fastapi_scheme_extraction(
    header: str | None, expected: str | None
) -> None:
    assert bearer_token(_scope(header)) == expected
    if header is not None:
        scheme, param = get_authorization_scheme_param(header)
        fastapi_token = param if (scheme.lower() == "bearer" and param) else None
        assert bearer_token(_scope(header)) == fastapi_token
