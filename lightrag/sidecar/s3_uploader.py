"""S3 asset uploader for sidecar image assets.

Uploads local image assets to S3 when enabled via environment variables.
Returns ``{ref: remote_url}`` mapping.  Remote URLs are stored in the
``remote_url`` field of ``drawings.json`` and used in ``blocks.jsonl``
placeholders; the local ``path`` is always preserved for VLM disk access.
"""

from __future__ import annotations

import os
from pathlib import Path

from lightrag.utils import logger


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_BOOL_TRUTHY = {"true", "1", "yes"}


def _is_enabled() -> bool:
    return os.environ.get("LIGHTRAG_ASSET_UPLOAD_ENABLED", "").lower().strip() in _BOOL_TRUTHY


def _get_config() -> dict[str, str | None]:
    """Read all S3 upload config from environment variables."""
    return {
        "endpoint": os.environ.get("LIGHTRAG_ASSET_UPLOAD_ENDPOINT"),
        "bucket": os.environ.get("LIGHTRAG_ASSET_UPLOAD_BUCKET"),
        "prefix": os.environ.get("LIGHTRAG_ASSET_UPLOAD_PREFIX", "lightrag/"),
        "region": os.environ.get("LIGHTRAG_ASSET_UPLOAD_REGION", "us-east-1"),
        "access_key": os.environ.get("LIGHTRAG_ASSET_UPLOAD_ACCESS_KEY")
        or os.environ.get("AWS_ACCESS_KEY_ID"),
        "secret_key": os.environ.get("LIGHTRAG_ASSET_UPLOAD_SECRET_KEY")
        or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "public_url_prefix": os.environ.get("LIGHTRAG_ASSET_UPLOAD_PUBLIC_URL_PREFIX"),
    }


# ---------------------------------------------------------------------------
# Upload logic
# ---------------------------------------------------------------------------


def upload_assets_to_s3(
    assets_dir: Path,
    asset_paths: dict[str, str],
    doc_id: str | None = None,
) -> dict[str, str] | None:
    """Upload all materialized assets to S3 and return a mapping of remote URLs.

    Args:
        assets_dir: The local directory containing asset files
            (e.g. ``<base>.blocks.assets/``).
        asset_paths: The ``{ref: filename_inside_assets_dir}`` mapping returned
            by ``_materialize_assets``.
        doc_id: Optional document ID used to create a per-document sub-prefix
            in S3 so that assets from different documents are isolated.

    Returns:
        ``{ref: remote_url}`` if upload succeeded, or ``None`` if the feature
        is disabled or misconfigured (a warning is logged in the latter case).
    """
    if not _is_enabled():
        return None

    config = _get_config()

    # Validate required config
    missing = [
        k for k in ("endpoint", "bucket", "access_key", "secret_key")
        if not config[k]
    ]
    if missing:
        logger.warning(
            "[s3_upload] LIGHTRAG_ASSET_UPLOAD_ENABLED is set but required "
            "config missing: %s. Skipping upload.",
            ", ".join(
                f"LIGHTRAG_ASSET_UPLOAD_{k.upper()}" for k in missing
            ),
        )
        return None

    try:
        import boto3
    except ImportError:
        logger.warning(
            "[s3_upload] boto3 is not installed. Install it with: "
            "pip install boto3  — skipping S3 upload."
        )
        return None

    endpoint: str = config["endpoint"]  # type: ignore[assignment]
    bucket: str = config["bucket"]  # type: ignore[assignment]
    prefix: str = config["prefix"] or ""  # type: ignore[assignment]
    region: str = config["region"] or "us-east-1"  # type: ignore[assignment]
    access_key: str = config["access_key"]  # type: ignore[assignment]
    secret_key: str = config["secret_key"]  # type: ignore[assignment]
    public_url_prefix: str | None = config["public_url_prefix"]  # type: ignore[assignment]

    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        verify=False
    )

    remote_urls: dict[str, str] = {}
    uploaded = 0
    failed = 0

    for ref, filename in asset_paths.items():
        local_path = assets_dir / filename
        if not local_path.exists():
            logger.warning(
                "[s3_upload] local asset %s (ref=%s) not found, skipping",
                local_path,
                ref,
            )
            failed += 1
            continue

        key = f"{prefix}{doc_id}/{filename}" if doc_id else f"{prefix}{filename}"
        try:
            s3_client.upload_file(
                str(local_path),
                bucket,
                key,
            )
        except Exception as exc:
            logger.warning(
                "[s3_upload] failed to upload %s to %s/%s/%s: %s",
                local_path,
                endpoint.rstrip("/"),
                bucket,
                key,
                exc,
            )
            failed += 1
            continue

        # Build the public URL
        if public_url_prefix:
            url = f"{public_url_prefix.rstrip('/')}/{key}"
        else:
            url = f"{endpoint.rstrip('/')}/{bucket}/{key}"
        remote_urls[ref] = url
        uploaded += 1

    logger.info(
        "[s3_upload] uploaded %d/%d assets to %s/%s/%s (%d failed)",
        uploaded,
        len(asset_paths),
        endpoint.rstrip("/"),
        bucket,
        prefix,
        failed,
    )

    return remote_urls
