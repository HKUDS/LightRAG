from __future__ import annotations

import argparse
import asyncio

from .db import get_database_url, run_schema
from .repositories import (
    PostgresSystemRepository,
    default_tenant_and_workspace,
    membership_for_master,
)
from .auth import SystemAuthService


async def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap the global Little Bull MASTER user."
    )
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--display-name", default=None)
    args = parser.parse_args()

    database_url = get_database_url()
    if not database_url:
        print(
            "LIGHTRAG_SYSTEM_DATABASE_URL/DATABASE_URL not set; bootstrap requires Postgres."
        )
        return 1

    await run_schema(database_url)
    repo = PostgresSystemRepository(database_url)
    tenant, workspace = default_tenant_and_workspace()
    await repo.create_tenant(tenant)
    await repo.create_workspace(workspace)
    auth = SystemAuthService(repo)
    user = await auth.bootstrap_master(
        username=args.username,
        password=args.password,
        display_name=args.display_name,
    )
    await repo.create_membership(membership_for_master(user.user_id))
    print(f"MASTER user bootstrapped: {user.username}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
