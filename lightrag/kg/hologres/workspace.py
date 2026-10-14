"""Workspace selection shared by all Hologres storage roles."""

from __future__ import annotations

import os

from ...utils import logger, validate_workspace


WORKSPACE_ENV_VAR = "HOLOGRES_WORKSPACE"


def resolve_workspace(
    workspace: object, *, role: str, environment=None
) -> str:
    """Apply the database-backend workspace priority chain.

    ``HOLOGRES_WORKSPACE`` overrides the instance value so an operator can
    move a deployment to existing data without changing application code.
    An empty instance value falls back to ``default`` for compatibility with
    the other database backends. Invalid values fail before construction
    completes.
    """

    source = os.environ if environment is None else environment
    override = source.get(WORKSPACE_ENV_VAR)
    selected = (
        override.strip()
        if isinstance(override, str) and override.strip()
        else workspace
    )
    if not isinstance(selected, str) or not selected.strip():
        selected = "default"
    else:
        selected = selected.strip()
    try:
        selected = validate_workspace(selected)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid Hologres {role} workspace") from None
    if isinstance(override, str) and override.strip():
        logger.warning(
            "Using %s: '%s' (overriding instance workspace for Hologres %s)",
            WORKSPACE_ENV_VAR,
            selected,
            role,
        )
    return selected
