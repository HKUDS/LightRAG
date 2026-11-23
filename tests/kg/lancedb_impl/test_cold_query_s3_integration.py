"""Real-OSS integration test for the DuckDB cold query.

WRITTEN, NOT YET RUN against OSS: the bucket policy currently denies List
operations, which lancedb's connect/namespace layer issues before the tag
check. Run it ON THE ECS BOX once the grant lands (the internal endpoint is
ECS-only):

    LIGHTRAG_RUN_INTEGRATION=true \
    OSS_BUCKET=<bucket> \
    OSS_ENDPOINT=oss-cn-hangzhou-internal.aliyuncs.com \
    OSS_REGION=oss-cn-hangzhou \
    OSS_ACCESS_KEY_ID=... OSS_ACCESS_KEY_SECRET=... \
    ./scripts/test.sh tests/kg/lancedb_impl/test_cold_query_s3_integration.py --run-integration

The export uses ``batch_size=500`` on purpose: DuckDB's bundled Lance 2.0.1
decoder panics on a fragment that holds exactly one row, and N1's default test
batch of 2 leaves such a fragment in the three-row chunks table. Whether the
ECS DuckDB 2.0.1 build needs a different S3 secret/URL-style combination is for
the integration run to discover; ``ColdQueryConfig.duckdb_setup`` is the
override point if ``storage_options_from_env`` is not enough.
"""

import os
import shutil
import uuid

import lancedb
import pytest

from lightrag.kg.lancedb_cold_query import ColdQueryConfig, ColdQueryError, cold_search
from lightrag.kg.lancedb_cold_recall import recall_at_k
from lightrag.kg.lancedb_impl import export_snapshot, storage_options_from_env

from .conftest import CHUNKS, expected_vector, seed_rag

pytestmark = pytest.mark.integration

_TABLES = ("docs", "chunks", "entities", "relations")


def _oss_env_ready() -> bool:
    if os.getenv("LIGHTRAG_RUN_INTEGRATION", "").lower() != "true":
        return False
    if not os.getenv("OSS_BUCKET") or not os.getenv("OSS_ENDPOINT"):
        return False
    return bool(storage_options_from_env()) and shutil.which("duckdb") is not None


@pytest.mark.skipif(
    not _oss_env_ready(),
    reason="needs LIGHTRAG_RUN_INTEGRATION=true + OSS_BUCKET/OSS_ENDPOINT + OSS credentials + duckdb CLI",
)
async def test_cold_query_reads_tagged_oss_snapshot(tmp_path):
    rag, embedder = await seed_rag(tmp_path)
    embedder.armed = True
    options = storage_options_from_env()
    prefix = f"lightrag-n3-integration-{uuid.uuid4().hex[:12]}"
    dest = f"s3://{os.environ['OSS_BUCKET']}/{prefix}"
    tag = f"int-{uuid.uuid4().hex[:8]}"
    vector = expected_vector(CHUNKS["chunk-1"]["content"])
    try:
        first = await export_snapshot(rag, dest, tag, batch_size=500, storage_options=options)
        assert embedder.export_calls == 0

        config = ColdQueryConfig(dest=dest, tag=tag, storage_options=options)
        hits = cold_search(config, vector, top_k=10)
        assert embedder.export_calls == 0
        assert hits[0].id == "chunk-1"
        assert hits[0].distance == pytest.approx(0.0, abs=1e-6)

        hot = await rag.chunks_vdb.query("ignored", top_k=10, query_embedding=vector)
        assert recall_at_k([hit.id for hit in hits], [row["id"] for row in hot], 10) == 1.0

        tag2 = f"int-{uuid.uuid4().hex[:8]}"
        await export_snapshot(rag, dest, tag2, batch_size=500, storage_options=options)
        with pytest.raises(ColdQueryError, match="behind the latest version"):
            cold_search(config, vector, top_k=10)
        latest = cold_search(ColdQueryConfig(dest=dest, tag=tag2, storage_options=options), vector, 10)
        assert latest[0].id == "chunk-1"
        assert first["tables"]["chunks"]["rows"] == len(CHUNKS)
    finally:
        # Best-effort cleanup of the disposable prefix; a cleanup failure is
        # reported, never hidden, but must not mask the verified outcome.
        try:
            db = lancedb.connect(dest, storage_options=options)
        except Exception as exc:
            print(f"cleanup: could not connect to {dest}: {exc}")
        else:
            for name in _TABLES:
                try:
                    db.drop_table(name)
                except Exception as exc:
                    print(f"cleanup: could not drop {dest}/{name}: {exc}")