"""Real-OSS integration test for rebuild-from-snapshot (zero re-embed promotion).

Run ON THE ECS BOX (the internal endpoint is ECS-only):

    LIGHTRAG_RUN_INTEGRATION=true \
    OSS_BUCKET=<bucket> \
    OSS_ENDPOINT=https://oss-cn-hangzhou-internal.aliyuncs.com \
    OSS_REGION=cn-hangzhou \
    OSS_DEFAULT_SECRET_ID=... OSS_DEFAULT_SECRET_KEY=... OSS_DEFAULT_SECRET_TOKEN=... \
    ./scripts/test.sh tests/tools/test_rebuild_from_snapshot_oss_integration.py --run-integration

lancedb 0.41 speaks oss:// natively: ``storage_options_from_env`` emits the
canonical access_key_id/access_key_secret/security_token/endpoint/region keys
(the STS token key is security_token; aws_* keys do NOT work against OSS).
OSS_ACCESS_KEY_ID/SECRET and oss_access_ak/sk names map too. The export half
reuses the N1 job; the restore half is this wave's promotion path read back
over OSS.
"""

import os
import uuid

import lancedb
import pytest

from lightrag.kg.lancedb_impl import _TABLES, export_snapshot, storage_options_from_env
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.tools.snapshot_restore import open_snapshot, restore_workspace_snapshot

from tests.kg.lancedb_impl.conftest import DIM, CountingEmbedder, seed_rag

from .test_rebuild_from_snapshot import (
    assert_restored_matches_seed,
    build_target,
    read_back_vdbs,
)

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


def _oss_env_ready() -> bool:
    if os.getenv("LIGHTRAG_RUN_INTEGRATION", "").lower() != "true":
        return False
    if not os.getenv("OSS_BUCKET") or not os.getenv("OSS_ENDPOINT"):
        return False
    return bool(storage_options_from_env())


@pytest.mark.skipif(
    not _oss_env_ready(),
    reason="needs LIGHTRAG_RUN_INTEGRATION=true + OSS_BUCKET/OSS_ENDPOINT + credentials",
)
async def test_restore_from_oss_snapshot_is_exact_and_idempotent(tmp_path):
    source, embedder = await seed_rag(tmp_path)
    embedder.armed = True
    options = storage_options_from_env()
    prefix = f"lightrag-n2-integration-{uuid.uuid4().hex[:12]}"
    dest = f"oss://{os.environ['OSS_BUCKET']}/{prefix}"
    tag = f"int-{uuid.uuid4().hex[:8]}"

    await export_snapshot(source, dest, tag, batch_size=2, storage_options=options)
    assert embedder.export_calls == 0
    for storage in (source.full_docs, source.text_chunks,
                    source.chunk_entity_relation_graph, source.entities_vdb,
                    source.relationships_vdb, source.chunks_vdb):
        await storage.finalize()

    target_embedder = CountingEmbedder()
    target_embedder.armed = True
    target, gc = await build_target(tmp_path, target_embedder)

    tables = open_snapshot(dest, tag, dim=DIM, storage_options=options)
    await restore_workspace_snapshot(tables, target, gc, batch_size=2)
    assert target_embedder.export_calls == 0
    await assert_restored_matches_seed(target)

    first = await read_back_vdbs(target)
    tables = open_snapshot(dest, tag, dim=DIM, storage_options=options)
    await restore_workspace_snapshot(tables, target, gc, batch_size=2)
    assert target_embedder.export_calls == 0
    assert await read_back_vdbs(target) == first

    db = lancedb.connect(dest, storage_options=options)
    for name in _TABLES:
        try:
            db.drop_table(name)
        except Exception as exc:
            print(f"cleanup: could not drop {dest}/{name}: {exc}")
