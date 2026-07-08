"""Bm25KeywordStorage: build/search/persist/degrade."""

import pytest

pytest.importorskip("bm25s")

from lightrag.kg.bm25s_keyword_impl import Bm25KeywordStorage  # noqa: E402


def make_storage(tmp_path, workspace=""):
    return Bm25KeywordStorage(
        namespace="entity_keywords",
        workspace=workspace,
        global_config={"working_dir": str(tmp_path)},
    )


@pytest.mark.asyncio
async def test_index_and_exact_term_search(tmp_path):
    st = make_storage(tmp_path)
    await st.initialize()
    await st.index_entities(["NVLink", "PCIe", "Apple Inc.", "苹果公司"])
    hits = await st.search("what is the bandwidth of NVLink?", top_k=3)
    assert hits, "expected at least one hit"
    assert hits[0][0] == "NVLink"


@pytest.mark.asyncio
async def test_cjk_entity_search(tmp_path):
    st = make_storage(tmp_path)
    await st.initialize()
    await st.index_entities(["苹果公司", "富士康", "NVIDIA"])
    hits = await st.search("苹果公司的供应链", top_k=2)
    assert hits[0][0] == "苹果公司"


@pytest.mark.asyncio
async def test_empty_index_returns_empty(tmp_path):
    st = make_storage(tmp_path)
    await st.initialize()
    assert await st.search("anything", top_k=5) == []


@pytest.mark.asyncio
async def test_persistence_round_trip(tmp_path):
    st = make_storage(tmp_path)
    await st.initialize()
    await st.index_entities(["NVLink", "PCIe"])
    st2 = make_storage(tmp_path)
    await st2.initialize()
    hits = await st2.search("NVLink", top_k=1)
    assert hits and hits[0][0] == "NVLink"


@pytest.mark.asyncio
async def test_workspace_isolation(tmp_path):
    a = make_storage(tmp_path, workspace="tenant_a")
    b = make_storage(tmp_path, workspace="tenant_b")
    await a.initialize()
    await b.initialize()
    await a.index_entities(["OnlyInA"])
    assert await b.search("OnlyInA", top_k=1) == []


@pytest.mark.asyncio
async def test_top_k_larger_than_corpus(tmp_path):
    st = make_storage(tmp_path)
    await st.initialize()
    await st.index_entities(["NVLink"])
    hits = await st.search("NVLink", top_k=50)
    assert len(hits) == 1


@pytest.mark.asyncio
async def test_rebuild_replaces_corpus(tmp_path):
    st = make_storage(tmp_path)
    await st.initialize()
    await st.index_entities(["AlphaZero"])
    await st.index_entities(["NVLink"])
    hits = await st.search("AlphaZero", top_k=5)
    assert all(name != "AlphaZero" for name, _ in hits)
    hits = await st.search("NVLink", top_k=5)
    assert hits and hits[0][0] == "NVLink"


@pytest.mark.asyncio
async def test_drop_clears_everything(tmp_path):
    st = make_storage(tmp_path)
    await st.initialize()
    await st.index_entities(["NVLink"])
    await st.drop()
    assert await st.search("NVLink", top_k=5) == []


@pytest.mark.asyncio
async def test_unavailable_when_bm25s_missing(tmp_path, monkeypatch):
    import lightrag.kg.bm25s_keyword_impl as mod

    monkeypatch.setattr(mod, "_import_bm25s", lambda: None)
    st = make_storage(tmp_path)
    await st.initialize()
    assert st.available is False
    await st.index_entities(["NVLink"])
    assert await st.search("NVLink", top_k=5) == []
