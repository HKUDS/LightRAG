"""Real-OSS integration test for the DuckDB cold query (native oss:// lake).

Run ON THE ECS BOX (the internal endpoint is ECS-only):

    LIGHTRAG_RUN_INTEGRATION=true \
    OSS_BUCKET=<bucket> \
    OSS_ENDPOINT=https://oss-cn-hangzhou-internal.aliyuncs.com \
    OSS_REGION=cn-hangzhou \
    OSS_DEFAULT_SECRET_ID=... OSS_DEFAULT_SECRET_KEY=... OSS_DEFAULT_SECRET_TOKEN=... \
    ./scripts/test.sh tests/kg/lancedb_impl/test_cold_query_oss_integration.py --run-integration

The export uses ``batch_size=500`` on purpose: DuckDB's bundled Lance 2.0.1
decoder panics on a fragment that holds exactly one row, and N1's default test
batch of 2 leaves such a fragment in the three-row chunks table. lancedb
writes through the native oss:// provider; DuckDB reads the same objects
through the S3 protocol (s3. endpoint, vhost style, STS SESSION_TOKEN), which
``_s3_setup`` builds from the native storage options.
``ColdQueryConfig.duckdb_setup`` is the override point if that is not enough.
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
    dest = f"oss://{os.environ['OSS_BUCKET']}/{prefix}"
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