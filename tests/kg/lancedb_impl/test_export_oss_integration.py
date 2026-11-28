"""Real-OSS integration test for the cold-snapshot export job (native oss://).

Run ON THE ECS BOX (the internal endpoint is ECS-only):

    LIGHTRAG_RUN_INTEGRATION=true \
    OSS_BUCKET=<bucket> \
    OSS_ENDPOINT=https://oss-cn-hangzhou-internal.aliyuncs.com \
    OSS_REGION=cn-hangzhou \
    OSS_DEFAULT_SECRET_ID=... OSS_DEFAULT_SECRET_KEY=... OSS_DEFAULT_SECRET_TOKEN=... \
    ./scripts/test.sh tests/kg/lancedb_impl/test_export_oss_integration.py --run-integration

lancedb 0.41 speaks oss:// natively: ``storage_options_from_env`` emits the
canonical access_key_id/access_key_secret/security_token/endpoint/region keys.
The STS token key is ``security_token``; aws_* keys and s3:// virtual-hosted
addressing do NOT work against OSS (the bucket is stripped -> NoSuchBucket).
OSS_ACCESS_KEY_ID/SECRET and oss_access_ak/sk names map too; AWS_* names serve
s3:// destinations only.
"""

import os
import uuid

import lancedb
import pytest

from lightrag.kg.lancedb_impl import export_snapshot, storage_options_from_env

from .conftest import CHUNKS, DOCS, EDGES, NODES, expected_vector, seed_rag

pytestmark = pytest.mark.integration

_TABLES = ("docs", "chunks", "entities", "relations")


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
async def test_export_to_oss_publishes_tagged_snapshot(tmp_path):
    rag, embedder = await seed_rag(tmp_path)
    embedder.armed = True
    options = storage_options_from_env()
    prefix = f"lightrag-n1-integration-{uuid.uuid4().hex[:12]}"
    dest = f"oss://{os.environ['OSS_BUCKET']}/{prefix}"
    tag = f"int-{uuid.uuid4().hex[:8]}"
    expected = {
        "docs": len(DOCS),
        "chunks": len(CHUNKS),
        "entities": len(NODES),
        "relations": len(EDGES),
    }

    report = await export_snapshot(
        rag, dest, tag, batch_size=2, storage_options=options
    )
    assert embedder.export_calls == 0

    # Fresh connection for read-back: the export committed through its own.
    db = lancedb.connect(dest, storage_options=options)
    try:
        for name, count in expected.items():
            table = db.open_table(name)
            assert table.count_rows() == count == report["tables"][name]["rows"]
            assert table.tags.get_version(tag) == table.version
        chunks = {r["id"]: r for r in db.open_table("chunks").to_arrow().to_pylist()}
        assert chunks["chunk-1"]["embedding"] == expected_vector(
            CHUNKS["chunk-1"]["content"]
        )

        tag2 = f"int-{uuid.uuid4().hex[:8]}"
        report2 = await export_snapshot(
            rag, dest, tag2, batch_size=2, storage_options=options
        )
        db = lancedb.connect(dest, storage_options=options)
        for name in _TABLES:
            table = db.open_table(name)
            assert (
                report2["tables"][name]["version"] > report["tables"][name]["version"]
            )
            assert table.tags.get_version(tag) == report["tables"][name]["version"]
            assert table.tags.get_version(tag2) == report2["tables"][name]["version"]
    finally:
        # Best-effort cleanup of the disposable test prefix; a cleanup failure
        # is reported, never hidden, but must not mask the verified outcome.
        for name in _TABLES:
            try:
                db.drop_table(name)
            except Exception as exc:
                print(f"cleanup: could not drop {dest}/{name}: {exc}")
