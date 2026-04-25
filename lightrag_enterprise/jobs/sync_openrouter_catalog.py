from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from lightrag_enterprise.model_gateway.openrouter import sync_openrouter_catalog


async def run_sync_openrouter_catalog(
    output_path: str | Path = "rag_storage/model_catalog/openrouter_catalog.json",
) -> int:
    catalog = await sync_openrouter_catalog(output_path, force=True)
    print(f"synced {len(catalog.entries)} OpenRouter models to {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync OpenRouter runtime model catalog")
    parser.add_argument(
        "--output",
        default="rag_storage/model_catalog/openrouter_catalog.json",
        help="Catalog cache output path",
    )
    args = parser.parse_args()
    return asyncio.run(run_sync_openrouter_catalog(args.output))


if __name__ == "__main__":
    raise SystemExit(main())
