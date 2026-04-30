"""Utility functions for the LightRAG API."""

from __future__ import annotations

import re

from fastapi import HTTPException, Request

__all__ = ["sanitize_workspace_name", "WorkspaceNameError", "extract_workspace_from_header"]


class WorkspaceNameError(ValueError):
    """Custom exception for workspace name validation errors.

    Attributes:
        detail: The validation error message describing why the workspace name is invalid.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def sanitize_workspace_name(name: str | None) -> str:
    """Sanitize a workspace name extracted from the LIGHTRAG-WORKSPACE HTTP header.

    Applies the following sanitization rules:
    1. Returns empty string for None or empty input (default workspace).
    2. Strips leading/trailing whitespace.
    3. Converts to lowercase.
    4. Rejects path traversal attempts containing '..', '/', or '\\'.
    5. Rejects names exceeding 64 characters.
    6. Rejects names containing characters other than lowercase letters,
       numbers, hyphens, and underscores.

    Args:
        name: The workspace name from the HTTP header, may be None.

    Returns:
        The sanitized workspace name, or empty string for the default workspace.

    Raises:
        WorkspaceNameError: If the workspace name is invalid.
    """
    # Rule 1: None or empty returns empty string (default workspace)
    if name is None or name == "":
        return ""

    # Rule 2: Strip whitespace
    name = name.strip()

    # Rule 3: Convert to lowercase
    name = name.lower()

    # Rule 4: Reject path traversal attempts
    if ".." in name or "/" in name or "\\" in name:
        raise WorkspaceNameError("Invalid workspace name: path traversal detected")

    # Rule 5: Limit length to 64 characters
    if len(name) > 64:
        raise WorkspaceNameError("Workspace name too long (max 64 characters)")

    # Rule 6: Only allow alphanumeric, hyphens, and underscores
    if not re.match(r"^[a-z0-9_-]+$", name):
        raise WorkspaceNameError(
            "Invalid workspace name: only lowercase letters, numbers, hyphens, and underscores allowed"
        )

    return name


def extract_workspace_from_header(request: Request) -> str:
    """Extract and sanitize workspace from LIGHTRAG-WORKSPACE header.

    Returns empty string for default workspace.
    Raises HTTPException(400) on invalid workspace name.
    """
    raw = request.headers.get("LIGHTRAG-WORKSPACE", "").strip()
    if raw:
        try:
            return sanitize_workspace_name(raw)
        except WorkspaceNameError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return ""

