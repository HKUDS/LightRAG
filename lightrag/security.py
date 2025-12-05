"""Security utilities for multi-tenant validation and sanitization."""

from pathlib import Path
from typing import Tuple
import uuid
from fastapi import HTTPException, status
import logging

logger = logging.getLogger(__name__)


def validate_uuid(value: str, param_name: str = "id") -> str:
    """Validate that a string is a valid UUID format.

    Args:
        value: String to validate
        param_name: Name of parameter (for error messages)

    Returns:
        The validated UUID string

    Raises:
        HTTPException: If value is not a valid UUID
    """
    if not value or not value.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {param_name}: empty value",
        )

    try:
        # Validate UUID format
        uuid.UUID(value.strip())
        return value.strip()
    except ValueError:
        logger.warning(f"Invalid UUID format for {param_name}: {value}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {param_name} format",
        )


def validate_identifier(value: str, param_name: str = "id") -> str:
    """Validate that a string is a safe identifier (UUID or slug).

    Accepts both UUID format and string slugs (alphanumeric with hyphens/underscores).
    Used for tenant_id and kb_id which can be either UUIDs or human-readable strings.

    Args:
        value: String to validate
        param_name: Name of parameter (for error messages)

    Returns:
        The validated identifier string

    Raises:
        HTTPException: If value is empty or contains unsafe characters
    """
    if not value or not value.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {param_name}: empty value",
        )

    value = value.strip()

    # Try UUID first
    try:
        uuid.UUID(value)
        return value
    except ValueError:
        pass  # Not a UUID, check if it's a valid slug

    # Validate as slug: alphanumeric, hyphens, underscores only
    # This prevents path traversal (no slashes) and injection attacks
    import re

    if not re.match(r"^[a-zA-Z0-9_-]+$", value):
        logger.warning(f"Invalid identifier format for {param_name}: {value}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {param_name} format. Must be UUID or alphanumeric with hyphens/underscores only.",
        )

    # Limit length to prevent abuse
    if len(value) > 255:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {param_name}: too long (max 255 characters)",
        )

    return value


def validate_working_directory(
    base_dir: str, tenant_id: str, kb_id: str
) -> Tuple[Path, str]:
    """Validate and create safe working directory for tenant/KB.

    This function does the following:
    1. Validates tenant_id and kb_id are safe identifiers (UUID or slug)
    2. Creates the directory path safely
    3. Verifies the resolved path stays within base_dir (prevents path traversal)
    4. Returns both the path and composite workspace identifier

    Args:
        base_dir: Base working directory for all tenants
        tenant_id: Tenant identifier (UUID or slug like 'acme-corp')
        kb_id: Knowledge base identifier (UUID or slug like 'kb-main')

    Returns:
        Tuple of (validated_path, composite_workspace_id)

    Raises:
        HTTPException: If validation fails or path traversal detected
    """
    # Validate identifiers (accepts both UUIDs and slugs)
    tenant_id = validate_identifier(tenant_id, "tenant_id")
    kb_id = validate_identifier(kb_id, "kb_id")

    try:
        # Create and resolve paths
        base_path = Path(base_dir).resolve()
        tenant_path = (base_path / tenant_id / kb_id).resolve()

        # Critical security check: verify path stays within base_dir
        if not tenant_path.is_relative_to(base_path):
            logger.error(
                f"Path traversal attempt detected: base={base_path}, "
                f"requested={tenant_path}, tenant={tenant_id}, kb={kb_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid path detected"
            )

        # Create composite workspace identifier
        composite_workspace = f"{tenant_id}:{kb_id}"

        logger.debug(
            f"Validated working directory: path={tenant_path}, workspace={composite_workspace}"
        )

        return tenant_path, composite_workspace

    except (OSError, RuntimeError) as e:
        logger.error(f"Error validating working directory: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate working directory",
        )


def sanitize_identifier(value: str, max_length: int = 255) -> str:
    """Sanitize an identifier string for safe use.

    Removes or replaces potentially dangerous characters while preserving
    readability. Used for names, descriptions, etc. - not UUIDs.

    Args:
        value: String to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized string

    Raises:
        HTTPException: If value is empty or too long after sanitization
    """
    if not value or not value.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Value cannot be empty"
        )

    # Remove null bytes and control characters
    sanitized = "".join(c for c in value if ord(c) >= 32 and c != "\x7f")

    # Remove path separators
    sanitized = sanitized.replace("/", "").replace("\\", "")

    # Strip and limit length
    sanitized = sanitized.strip()[:max_length]

    if not sanitized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid value after sanitization",
        )

    return sanitized
