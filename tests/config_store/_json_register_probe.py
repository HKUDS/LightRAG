"""One process tree registering its workspace in a JSON configuration group:
run twice at once, for two workspaces, by ``test_json_config_shards.py``.

Each process binds its own real JsonKVStorage. The first pauses during
registration while holding the bind lock and resumes only after the second
has encountered contention on that lock.
"""

from __future__ import annotations

import asyncio
import fcntl
import sys
from pathlib import Path
from unittest.mock import patch

from _rendezvous import arrive_and_wait

from lightrag import config_store as cs
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data


async def _wait_for_marker(path: Path) -> None:
    async with asyncio.timeout(10):
        while not path.exists():
            await asyncio.sleep(0.005)


async def main(working_dir: str, workspace: str, rendezvous: str, role: str) -> None:
    initialize_share_data(workers=1)
    holder_reading = Path(rendezvous) / "holder-reading"
    contender_blocked = Path(rendezvous) / "contender-blocked"
    read_shard_rows = cs.read_shard_rows
    flock = fcntl.flock

    async def read_while_holding(config):
        rows = await read_shard_rows(config)
        holder_reading.touch()
        try:
            await _wait_for_marker(contender_blocked)
        except TimeoutError:
            raise AssertionError(
                "second registrant did not encounter bind-lock contention"
            ) from None
        return rows

    def observe_contention(fd, operation):
        try:
            return flock(fd, operation)
        except BlockingIOError:
            if operation == (fcntl.LOCK_EX | fcntl.LOCK_NB):
                contender_blocked.touch()
            raise

    config = cs.create_configuration_storage(
        JsonKVStorage,
        global_config={"working_dir": working_dir, "workspace": workspace},
        embedding_func=None,
    )
    await config.initialize()
    try:
        arrive_and_wait(rendezvous)
        if role == "holder":
            instrumentation = patch.object(cs, "read_shard_rows", read_while_holding)
        else:
            await _wait_for_marker(holder_reading)
            instrumentation = patch.object(fcntl, "flock", observe_contention)
        with instrumentation:
            binding = await cs.bind_configuration_identity(
                config,
                working_dir=working_dir,
                backend="JsonKVStorage",
                container=f"JsonKVStorage ({workspace})",
                workspace=workspace,
            )
        print(f"ACTION={binding.action}")
        print(f"UUID={binding.storage_uuid}")
    finally:
        await config.finalize()
        finalize_share_data()


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
