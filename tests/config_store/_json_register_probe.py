"""One process tree registering its workspace in a JSON configuration group:
run twice at once, for two workspaces, by ``test_json_config_shards.py``.

Each process opens its own workspace's real ``JsonKVStorage`` snapshot and
binds it. The snapshot read is slowed down so both starts hold the member
list they read at the same moment -- the shape of two servers with
different workspaces sharing one ``WORKING_DIR`` on first use. Only the
anchor bind lock keeps one append from overwriting the other.
"""

from __future__ import annotations

import asyncio
import sys
import time

from lightrag import config_store as cs
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data


async def main(working_dir: str, workspace: str, start_at: float) -> None:
    initialize_share_data(workers=1)
    original = cs.read_shard_rows

    async def _slow_read(config):
        await asyncio.sleep(0.3)
        return await original(config)

    cs.read_shard_rows = _slow_read
    config = cs.create_configuration_storage(
        JsonKVStorage,
        global_config={"working_dir": working_dir, "workspace": workspace},
        embedding_func=None,
    )
    await config.initialize()
    try:
        time.sleep(max(0.0, start_at - time.time()))
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
    asyncio.run(main(sys.argv[1], sys.argv[2], float(sys.argv[3])))
