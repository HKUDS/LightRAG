"""Parity test: every QueryParam field must be represented in QueryRequest.

When adding a new parameter to QueryParam, this test fails until you also
add the corresponding field to QueryRequest. That is the intended behaviour
— it enforces the single-edit-point discipline at CI time.

QueryRequest is parsed via ast rather than imported to avoid triggering
the API auth chain (which calls parse_args() and fails outside a server).

Intentional exclusions (must be documented here when added):
  model_func      — callable; cannot be serialised as an HTTP field.
  history_turns   — deprecated; intentionally not exposed in the API.
"""

import ast
import dataclasses
import pathlib
import pytest

from lightrag.query_config import QueryParam

# Fields on QueryParam intentionally absent from QueryRequest.
_EXCLUDED_FROM_REQUEST = {
    "model_func",     # callable, not serialisable as an HTTP field
    "history_turns",  # deprecated, not exposed via the API
}

_QUERY_ROUTES = (
    pathlib.Path(__file__).parent.parent
    / "lightrag" / "api" / "routers" / "query_routes.py"
)


def _query_request_fields() -> set[str]:
    """Parse QueryRequest field names from source without importing the module."""
    tree = ast.parse(_QUERY_ROUTES.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "QueryRequest":
            return {
                target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                for target in [stmt.target]
            }
    raise RuntimeError("QueryRequest class not found in query_routes.py")


@pytest.mark.offline
def test_query_request_covers_all_query_param_fields():
    """Every non-excluded QueryParam field must appear in QueryRequest."""
    param_fields = {f.name for f in dataclasses.fields(QueryParam)}
    request_fields = _query_request_fields()

    expected = param_fields - _EXCLUDED_FROM_REQUEST
    missing = expected - request_fields

    assert not missing, (
        f"QueryParam fields missing from QueryRequest: {sorted(missing)}\n"
        "Add them as Optional[...] = Field(default=None, ...) in QueryRequest, "
        "or add to _EXCLUDED_FROM_REQUEST with a documented reason."
    )


@pytest.mark.offline
def test_excluded_fields_are_still_in_query_param():
    """Exclusion list must not silently become stale."""
    param_fields = {f.name for f in dataclasses.fields(QueryParam)}
    stale = _EXCLUDED_FROM_REQUEST - param_fields

    assert not stale, (
        f"Fields in _EXCLUDED_FROM_REQUEST no longer exist in QueryParam: {sorted(stale)}\n"
        "Remove them from _EXCLUDED_FROM_REQUEST."
    )
