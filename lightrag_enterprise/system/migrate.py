from __future__ import annotations

import asyncio

from .db import get_database_url, run_schema


async def _main() -> int:
    database_url = get_database_url()
    if not database_url:
        print(
            "LIGHTRAG_SYSTEM_DATABASE_URL/DATABASE_URL not set; no Postgres migration was run."
        )
        return 0
    await run_schema(database_url)
    print("Little Bull system Postgres schema is up to date.")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
