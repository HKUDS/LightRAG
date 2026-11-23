"""Offline contract tests for the Lance cold-snapshot export job.

Each test pins one clause of the module's caller rules: stored-vector
fidelity with zero embedding calls, versioned re-export with immutable tags,
fail-before-write on empty sources / missing vectors / tag collision, loud
failure on uninitialized or unflushed sources, and the interrupted-export
recovery path. The CLI test drives the real ``main()`` entry point.
"""

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import lancedb
import lancedb.table
import pytest

from lightrag.kg.lancedb_impl import (
    ExportError,
    export_snapshot,
    main,
    storage_options_from_env,
)
from lightrag.kg.shared_storage import finalize_share_data
from lightrag.utils import EmbeddingFunc, compute_mdhash_id

from .conftest import (
    CHUNKS,
    DIM,
    DOCS,
    EDGES,
    MODEL_NAME,
    NODES,
    entity_content,
    expected_vector,
    relation_content,
    seed_rag,
)

pytestmark = pytest.mark.offline

_TABLES = ("docs", "chunks", "entities", "relations")


async def test_export_publishes_tagged_snapshot_with_stored_vectors(tmp_path, seeded):
    rag, embedder = seeded
    embedder.armed = True
    dest = str(tmp_path / "lake")

    report = await export_snapshot(rag, dest, "snap-1", batch_size=2)

    assert embedder.export_calls == 0
    assert report["tag"] == "snap-1"
    db = lancedb.connect(dest)
    assert set(db.list_tables()) == set(_TABLES)

    docs = db.open_table("docs").to_arrow().to_pylist()
    assert {r["id"]: r["content"] for r in docs} == {
        k: v["content"] for k, v in DOCS.items()
    }

    chunks = {r["id"]: r for r in db.open_table("chunks").to_arrow().to_pylist()}
    assert set(chunks) == set(CHUNKS)
    for cid, source in CHUNKS.items():
        assert chunks[cid]["content"] == source["content"]
        assert chunks[cid]["full_doc_id"] == source["full_doc_id"]
        assert chunks[cid]["chunk_order_index"] == source["chunk_order_index"]
        assert chunks[cid]["tokens"] == source["tokens"]
        assert chunks[cid]["file_path"] == source["file_path"]
        assert chunks[cid]["embedding"] == expected_vector(source["content"])

    entities = {r["entity_name"]: r for r in db.open_table("entities").to_arrow().to_pylist()}
    assert set(entities) == set(NODES)
    for name, props in NODES.items():
        assert entities[name]["id"] == compute_mdhash_id(name, prefix="ent-")
        assert entities[name]["entity_type"] == props["entity_type"]
        assert entities[name]["description"] == props["description"]
        assert entities[name]["source_id"] == props["source_id"]
        assert entities[name]["embedding"] == expected_vector(entity_content(name))

    relations = db.open_table("relations").to_arrow().to_pylist()
    assert len(relations) == len(EDGES)
    src, tgt, props = EDGES[0]
    s, t = sorted((src, tgt))
    rel = relations[0]
    assert rel["id"] == compute_mdhash_id(s + t, prefix="rel-")
    assert (rel["src_id"], rel["tgt_id"]) == (s, t)
    assert rel["keywords"] == props["keywords"]
    assert rel["description"] == props["description"]
    assert rel["weight"] == props["weight"]
    assert rel["embedding"] == expected_vector(relation_content(s, t, props))

    for name in _TABLES:
        table = db.open_table(name)
        assert table.tags.get_version("snap-1") == table.version
        assert report["tables"][name] == {"rows": table.count_rows(), "version": table.version}
        assert table.schema.metadata[b"lightrag.embedding_model"] == MODEL_NAME.encode()


async def test_reexport_creates_new_version_and_keeps_old_tag(tmp_path, seeded):
    rag, embedder = seeded
    dest = str(tmp_path / "lake")
    first = await export_snapshot(rag, dest, "v1")

    new_chunk = {"content": "gamma chunk", "full_doc_id": "doc-2",
                 "chunk_order_index": 1, "tokens": 2, "file_path": "b.txt"}
    await rag.text_chunks.upsert({"chunk-4": new_chunk})
    await rag.chunks_vdb.upsert({"chunk-4": {
        "content": new_chunk["content"], "full_doc_id": "doc-2", "file_path": "b.txt"}})
    await rag.text_chunks.index_done_callback()
    await rag.chunks_vdb.index_done_callback()

    embedder.armed = True
    second = await export_snapshot(rag, dest, "v2")
    assert embedder.export_calls == 0

    db = lancedb.connect(dest)
    for name in _TABLES:
        table = db.open_table(name)
        assert second["tables"][name]["version"] > first["tables"][name]["version"]
        assert table.tags.get_version("v1") == first["tables"][name]["version"]
        assert table.tags.get_version("v2") == second["tables"][name]["version"]
    chunks = db.open_table("chunks")
    assert chunks.count_rows() == len(CHUNKS) + 1
    chunks.checkout("v1")
    assert chunks.count_rows() == len(CHUNKS)


async def test_existing_tag_refused_before_any_write(tmp_path, seeded):
    rag, _ = seeded
    dest = str(tmp_path / "lake")
    first = await export_snapshot(rag, dest, "same")

    with pytest.raises(ExportError, match="already exists"):
        await export_snapshot(rag, dest, "same")

    db = lancedb.connect(dest)
    for name, info in first["tables"].items():
        assert db.open_table(name).version == info["version"]


@pytest.mark.parametrize("empty_source", ["docs", "chunks", "entities", "relations"])
async def test_empty_source_refused_before_write(tmp_path, seeded, empty_source):
    rag, _ = seeded
    if empty_source == "docs":
        await rag.full_docs.drop()
    elif empty_source == "chunks":
        await rag.text_chunks.drop()
    elif empty_source == "entities":
        await rag.chunk_entity_relation_graph.drop()
    else:
        await rag.chunk_entity_relation_graph.remove_edges([("Alice", "Bob")])

    dest = str(tmp_path / "lake")
    with pytest.raises(ExportError, match="empty"):
        await export_snapshot(rag, dest, "snap")
    assert list(lancedb.connect(dest).list_tables()) == []


async def test_missing_vector_aborts_before_write(tmp_path, seeded):
    rag, embedder = seeded
    ent_id = compute_mdhash_id("Alice", prefix="ent-")
    await rag.entities_vdb.delete([ent_id])
    await rag.entities_vdb.index_done_callback()
    embedder.armed = True

    dest = str(tmp_path / "lake")
    with pytest.raises(ExportError, match=ent_id):
        await export_snapshot(rag, dest, "snap")
    assert embedder.export_calls == 0
    assert list(lancedb.connect(dest).list_tables()) == []


async def test_unflushed_vector_row_cannot_silently_embed(tmp_path, seeded):
    """A pending (unflushed) row must not be embedded silently by the export.

    nano's read path lazily embeds pending rows; with the fail-loud embedder
    armed, that attempt aborts the export and is recorded. This is why the
    module rule demands flushed sources and the CLI installs a raising stub.
    """
    rag, embedder = seeded
    pending = {"content": "pending chunk", "full_doc_id": "doc-1",
               "chunk_order_index": 9, "tokens": 2, "file_path": "a.txt"}
    await rag.text_chunks.upsert({"chunk-x": pending})
    await rag.chunks_vdb.upsert({"chunk-x": {
        "content": pending["content"], "full_doc_id": "doc-1", "file_path": "a.txt"}})
    await rag.text_chunks.index_done_callback()  # chunks_vdb deliberately NOT flushed
    embedder.armed = True

    with pytest.raises(AssertionError, match="during export"):
        await export_snapshot(rag, str(tmp_path / "lake"), "snap")
    assert embedder.export_calls == 1


async def test_missing_storages_refused(tmp_path):
    rag = SimpleNamespace(
        full_docs=None, text_chunks=None, chunk_entity_relation_graph=None,
        entities_vdb=None, relationships_vdb=None, chunks_vdb=None,
        embedding_func=None,
    )
    with pytest.raises(ExportError, match="initialize_storages"):
        await export_snapshot(rag, str(tmp_path / "lake"), "snap")


async def test_invalid_dest_fails_loud(tmp_path, seeded):
    rag, _ = seeded
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("occupied")
    # lancedb refuses a regular-file destination; the export must propagate it.
    with pytest.raises(FileExistsError):
        await export_snapshot(rag, str(blocker), "snap")


@pytest.mark.parametrize("tag", ["", "   "], ids=["empty", "blank"])
async def test_blank_tag_refused(tmp_path, seeded, tag):
    rag, _ = seeded
    with pytest.raises(ExportError, match="non-empty"):
        await export_snapshot(rag, str(tmp_path / "lake"), tag)


async def test_interrupted_before_tag_leaves_no_tag_and_rerun_publishes(
    tmp_path, seeded, monkeypatch
):
    rag, _ = seeded
    dest = str(tmp_path / "lake")
    original = lancedb.table.Tags.create
    calls = {"n": 0}

    def flaky_create(self, tag, version):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated interruption before tagging")
        return original(self, tag, version)

    with monkeypatch.context() as m:
        m.setattr(lancedb.table.Tags, "create", flaky_create)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            await export_snapshot(rag, dest, "snap")
    assert calls["n"] == 1

    db = lancedb.connect(dest)
    for name in _TABLES:
        assert db.open_table(name).tags.list() == {}

    report = await export_snapshot(rag, dest, "snap")
    db = lancedb.connect(dest)
    for name, info in report["tables"].items():
        table = db.open_table(name)
        assert table.tags.get_version("snap") == info["version"]
        assert table.count_rows() == info["rows"]


def test_cli_exports_seeded_workspace(tmp_path, monkeypatch):
    """Drive the real main() entry point over a seeded working dir.

    Seeding runs in its own event loop; shared storage is then reset so the
    CLI's fresh loop builds its own locks (asyncio primitives cannot cross
    loops). The CLI's raising embedding stub is the enforcement: any embed
    attempt during the run would fail the test.
    """
    asyncio.run(seed_rag(tmp_path))
    finalize_share_data()
    dest = str(tmp_path / "cli-lake")
    monkeypatch.setattr(
        "sys.argv",
        ["lancedb_impl", "--dest", dest, "--tag", "cli-snap", "--batch-size", "2"],
    )

    main()

    db = lancedb.connect(dest)
    expected = {"docs": len(DOCS), "chunks": len(CHUNKS),
                "entities": len(NODES), "relations": len(EDGES)}
    for name, count in expected.items():
        table = db.open_table(name)
        assert table.count_rows() == count
        assert table.tags.get_version("cli-snap") == table.version
    chunks = {r["id"]: r for r in db.open_table("chunks").to_arrow().to_pylist()}
    assert chunks["chunk-1"]["embedding"] == expected_vector(CHUNKS["chunk-1"]["content"])


_CREDS_ENV = (
    "OSS_DEFAULT_SECRET_ID", "OSS_DEFAULT_SECRET_KEY", "OSS_DEFAULT_SECRET_TOKEN",
    "OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET", "oss_access_ak", "oss_access_sk",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "OSS_ENDPOINT", "oss_endpoint", "AWS_ENDPOINT_URL", "AWS_ENDPOINT",
    "OSS_REGION", "AWS_REGION",
)
_STS_ENV = {
    "OSS_DEFAULT_SECRET_ID": "STS.test-id", "OSS_DEFAULT_SECRET_KEY": "test-secret",
    "OSS_DEFAULT_SECRET_TOKEN": "test-token",
    "OSS_ENDPOINT": "https://oss-cn-hangzhou-internal.aliyuncs.com",
}


@pytest.mark.parametrize(
    ("env", "dest", "expected"),
    [
        pytest.param(
            _STS_ENV, "oss://bucket/prefix",
            {"access_key_id": "STS.test-id", "access_key_secret": "test-secret",
             "security_token": "test-token",
             "endpoint": "https://oss-cn-hangzhou-internal.aliyuncs.com",
             "region": "cn-hangzhou"},
            id="native-oss-sts",
        ),
        pytest.param(
            _STS_ENV, "s3://bucket/prefix",
            {"aws_access_key_id": "STS.test-id", "aws_secret_access_key": "test-secret",
             "aws_session_token": "test-token", "virtual_hosted_style_request": "true",
             "aws_endpoint": "https://oss-cn-hangzhou-internal.aliyuncs.com",
             "aws_allow_http": "false", "aws_region": "us-east-1"},
            id="s3-compat-sts",
        ),
        pytest.param(
            {"OSS_DEFAULT_SECRET_ID": "STS.test-id", "OSS_DEFAULT_SECRET_KEY": "test-secret",
             "OSS_REGION": "cn-beijing", "AWS_SESSION_TOKEN": "aws-token",
             "AWS_ENDPOINT_URL": "https://s3.us-west-2.amazonaws.com", "AWS_REGION": "us-west-2"},
            "oss://bucket/prefix",
            {"access_key_id": "STS.test-id", "access_key_secret": "test-secret",
             "region": "cn-beijing"},
            id="native-oss-ignores-aws-fallbacks",
        ),
        pytest.param(
            {"AWS_ACCESS_KEY_ID": "AKIA-test", "AWS_SECRET_ACCESS_KEY": "aws-secret",
             "AWS_SESSION_TOKEN": "aws-token", "AWS_ENDPOINT_URL": "http://localhost:9000",
             "AWS_REGION": "eu-west-1"},
            "s3://bucket/prefix",
            {"aws_access_key_id": "AKIA-test", "aws_secret_access_key": "aws-secret",
             "aws_session_token": "aws-token", "virtual_hosted_style_request": "true",
             "aws_endpoint": "http://localhost:9000", "aws_allow_http": "true",
             "aws_region": "eu-west-1"},
            id="aws-only-s3",
        ),
        pytest.param({}, "oss://bucket/prefix", {}, id="no-creds"),
    ],
)
def test_storage_options_shape_follows_dest_scheme(monkeypatch, env, dest, expected):
    """Native oss:// needs the canonical OSS keys (security_token, endpoint
    region derived from the endpoint host); s3:// keeps the aws_* shape."""
    for name in _CREDS_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert storage_options_from_env(dest) == expected


async def test_concat_collision_edges_both_archived(tmp_path, seeded):
    """Edges ('A','BC') and ('AB','C') concatenate to the same 'ABC' hash;
    dedup must key on the endpoint tuple so both survive as distinct rows."""
    rag, embedder = seeded
    graph = rag.chunk_entity_relation_graph
    names = ("A", "BC", "AB", "C")
    for name in names:
        await graph.upsert_node(name, {"entity_type": "person", "description": "x",
                                       "source_id": "chunk-1", "file_path": "a.txt"})
    await graph.upsert_edge("A", "BC", {"keywords": "k1", "description": "d1",
                                        "source_id": "chunk-1", "weight": 1.0, "file_path": "a.txt"})
    await graph.upsert_edge("AB", "C", {"keywords": "k2", "description": "d2",
                                        "source_id": "chunk-1", "weight": 1.0, "file_path": "a.txt"})
    await rag.entities_vdb.upsert({
        compute_mdhash_id(name, prefix="ent-"): {
            "entity_name": name, "content": f"{name}\nx",
            "source_id": "chunk-1", "file_path": "a.txt"}
        for name in names
    })
    await rag.relationships_vdb.upsert({
        compute_mdhash_id("ABC", prefix="rel-"): {
            "src_id": "A", "tgt_id": "BC", "content": "k1\tA\nBC\nd1",
            "source_id": "chunk-1", "file_path": "a.txt"}})
    for storage in (graph, rag.entities_vdb, rag.relationships_vdb):
        await storage.index_done_callback()
    embedder.armed = True

    dest = str(tmp_path / "lake")
    report = await export_snapshot(rag, dest, "snap")

    assert embedder.export_calls == 0
    relations = lancedb.connect(dest).open_table("relations").to_arrow().to_pylist()
    assert len(relations) == len(EDGES) + 2
    pairs = {(r["src_id"], r["tgt_id"]) for r in relations}
    assert {("A", "BC"), ("AB", "C")} <= pairs
    colliding_ids = {r["id"] for r in relations if r["src_id"] in ("A", "AB")}
    assert colliding_ids == {compute_mdhash_id("ABC", prefix="rel-")}
    assert report["tables"]["relations"]["rows"] == len(EDGES) + 2


def test_cli_resolves_provider_default_dim_and_model(tmp_path, monkeypatch):
    """With EMBEDDING_DIM/MODEL unset the CLI must resolve the binding's
    provider defaults (dim != 1024 here), not a hard-coded dimension."""
    asyncio.run(seed_rag(tmp_path))
    finalize_share_data()
    calls = []

    async def _unused_provider(texts, **kwargs):
        calls.append(texts)
        raise AssertionError("export must never call the provider embedder")

    fake = ModuleType("lightrag.llm.fakebind")
    fake.fakebind_embed = EmbeddingFunc(
        embedding_dim=DIM, func=_unused_provider, model_name=MODEL_NAME)
    monkeypatch.setitem(sys.modules, "lightrag.llm.fakebind", fake)
    monkeypatch.setenv("EMBEDDING_BINDING", "fakebind")
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    dest = str(tmp_path / "cli-lake")
    monkeypatch.setattr(
        "sys.argv",
        ["lancedb_impl", "--dest", dest, "--tag", "cli-snap", "--batch-size", "2"],
    )

    main()

    assert calls == []
    db = lancedb.connect(dest)
    for name in _TABLES:
        table = db.open_table(name)
        assert table.tags.get_version("cli-snap") == table.version
        assert table.schema.metadata[b"lightrag.embedding_model"] == MODEL_NAME.encode()
    assert db.open_table("chunks").count_rows() == len(CHUNKS)


@pytest.mark.parametrize(
    ("binding", "extra_env", "model"),
    [
        pytest.param("openai", {"EMBEDDING_MODEL": "custom-embed-model"},
                     "custom-embed-model", id="openai-custom-model"),
        pytest.param("azure_openai", {"AZURE_EMBEDDING_DEPLOYMENT": "my-deployment"},
                     "my-deployment", id="azure-deployment"),
    ],
)
def test_cli_custom_model_without_dim_refused(tmp_path, monkeypatch, binding, extra_env, model):
    """A non-default model with no EMBEDDING_DIM must fail loud like the
    server's dimension guard, not silently export under the provider default."""
    monkeypatch.setenv("EMBEDDING_BINDING", binding)
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("AZURE_EMBEDDING_DEPLOYMENT", raising=False)
    for name, value in extra_env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        "sys.argv",
        ["lancedb_impl", "--dest", str(tmp_path / "lake"), "--tag", "snap"],
    )

    with pytest.raises(ExportError, match=rf"EMBEDDING_DIM must be set for model '{model}'"):
        main()
    assert list(lancedb.connect(str(tmp_path / "lake")).list_tables()) == []


async def test_partial_tag_failure_recovers_with_new_tag(tmp_path, seeded, monkeypatch):
    """A tag loop that fails after >= 1 tag leaves a partial tag set: a
    same-tag retry is refused (tags immutable) and a NEW tag publishes cleanly."""
    rag, _ = seeded
    dest = str(tmp_path / "lake")
    original = lancedb.table.Tags.create
    calls = {"n": 0}

    def flaky_create(self, tag, version):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated interruption mid-tagging")
        return original(self, tag, version)

    with monkeypatch.context() as m:
        m.setattr(lancedb.table.Tags, "create", flaky_create)
        with pytest.raises(RuntimeError, match="mid-tagging"):
            await export_snapshot(rag, dest, "snap")
    assert calls["n"] == 2

    db = lancedb.connect(dest)
    assert "snap" in db.open_table("docs").tags.list()
    assert "snap" not in db.open_table("chunks").tags.list()

    with pytest.raises(ExportError, match="already exists"):
        await export_snapshot(rag, dest, "snap")

    report = await export_snapshot(rag, dest, "snap-2")
    db = lancedb.connect(dest)
    for name, info in report["tables"].items():
        table = db.open_table(name)
        assert table.tags.get_version("snap-2") == info["version"]
        assert table.count_rows() == info["rows"]