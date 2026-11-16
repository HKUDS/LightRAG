"""One process tree binding the configuration identity: run twice at once by
``test_bind_across_process_trees.py``. The store is a JSON file shared by
both processes, written through on upsert, with a slow read so both starts
reach the decision together -- the shape of two servers with different
workspaces sharing one ``working_dir`` and one server-backed container."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

from lightrag import config_store as cs
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data


class FileKV:
    supports_strict_point_reads = True

    def __init__(self, path: str) -> None:
        self.path = path

    def _load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    async def get_by_id_strict(self, key):
        await asyncio.sleep(0.3)
        row = self._load().get(key)
        return None if row is None else dict(row)

    async def upsert(self, data):
        rows = self._load()
        rows.update(data)
        tmp = f"{self.path}.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rows, f)
        os.replace(tmp, self.path)

    async def index_done_callback(self):
        return None


async def main(working_dir: str, store: str, start_at: float) -> None:
    initialize_share_data(workers=1)
    try:
        time.sleep(max(0.0, start_at - time.time()))
        binding = await cs.bind_configuration_identity(
            FileKV(store),
            working_dir=working_dir,
            backend="PGKVStorage",
            container="PGKVStorage (_lightrag_config)",
        )
        print(f"ACTION={binding.action}")
        print(f"UUID={binding.storage_uuid}")
    finally:
        finalize_share_data()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], float(sys.argv[3])))
