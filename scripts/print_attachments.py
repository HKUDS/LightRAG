#!/usr/bin/env python
"""Print query ``attachments`` and optional sidecar asset URLs from a running server.

Keep ``lightrag-server`` running in another terminal (use ``.venv\\Scripts\\lightrag-server``), then:

    cd LightRAG
    .venv\\Scripts\\python.exe scripts\\print_attachments.py
    .venv\\Scripts\\python.exe scripts\\print_attachments.py --asset-urls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_QUERY = "部分螺纹螺钉型式尺寸"
DEFAULT_MODE = "hybrid"


def _base_url(host: str | None, port: int | None) -> str:
    host = (host or os.getenv("HOST") or "127.0.0.1").strip()
    port_raw = port if port is not None else os.getenv("PORT", "9621")
    return f"http://{host}:{int(port_raw)}"


def _auth_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = api_key if api_key is not None else os.getenv("LIGHTRAG_API_KEY")
    if key:
        headers["X-API-Key"] = key.strip()
    return headers


def fetch_attachments(
    *,
    base_url: str,
    query: str,
    mode: str,
    timeout: float,
    api_key: str | None = None,
) -> tuple[dict, list | None]:
    """POST /query/data and return the full JSON body plus attachments list."""
    url = f"{base_url.rstrip('/')}/query/data"
    response = requests.post(
        url,
        json={"query": query, "mode": mode},
        headers=_auth_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    attachments = (payload.get("data") or {}).get("attachments")
    return payload, attachments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print data.attachments from POST /query/data (PR-1 verify)."
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help=f"Query text (default: {DEFAULT_QUERY!r})",
    )
    parser.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        choices=["local", "global", "hybrid", "mix", "naive"],
        help=f"Retrieval mode (default: {DEFAULT_MODE})",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Server host (default: HOST from .env or 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: PORT from .env or 9621)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="X-API-Key header (default: LIGHTRAG_API_KEY from .env if set)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print the entire /query/data JSON response",
    )
    parser.add_argument(
        "--asset-urls",
        action="store_true",
        help="Also print GET /documents/sidecar/assets URLs for each attachment",
    )
    args = parser.parse_args(argv)

    load_dotenv(PROJECT_ROOT / ".env")
    base_url = _base_url(args.host, args.port)

    try:
        payload, attachments = fetch_attachments(
            base_url=base_url,
            query=args.query,
            mode=args.mode,
            timeout=args.timeout,
            api_key=args.api_key,
        )
    except requests.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    if args.full:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("status:", payload.get("status"))
    print("message:", payload.get("message"))
    data = payload.get("data") or {}
    if "attachments" not in data:
        print(
            "attachments: (field missing — restart lightrag-server after PR-1 hook)",
            file=sys.stderr,
        )
    print("attachments:")
    print(json.dumps(attachments, ensure_ascii=False, indent=2))
    if isinstance(attachments, list):
        print(f"\n({len(attachments)} item(s))", file=sys.stderr)
        if args.asset_urls:
            print("\nasset_urls:", file=sys.stderr)
            for item in attachments:
                doc_id = item.get("doc_id")
                im_id = item.get("im_id")
                if doc_id and im_id:
                    url = (
                        f"{base_url.rstrip('/')}/documents/sidecar/assets"
                        f"?doc_id={doc_id}&im_id={im_id}"
                    )
                    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
