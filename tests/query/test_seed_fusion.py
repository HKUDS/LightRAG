"""RRF fusion of vector and BM25 seed rankings."""

from lightrag.operate import fuse_seed_rankings


def test_consensus_entity_ranks_first():
    fused = fuse_seed_rankings(
        vector_names=["a", "shared", "b"],
        bm25_names=["shared", "c"],
        top_k=10,
    )
    names = [n for n, _ in fused]
    assert names[0] == "shared"
    assert dict(fused)["shared"] == "both"


def test_single_list_high_rank_survives():
    fused = fuse_seed_rankings(
        vector_names=[f"v{i}" for i in range(30)],
        bm25_names=["jargon"],
        top_k=5,
    )
    names = [n for n, _ in fused]
    assert "jargon" in names


def test_dedup_and_top_k_truncation():
    fused = fuse_seed_rankings(
        vector_names=["a", "b", "c"],
        bm25_names=["b", "d"],
        top_k=3,
    )
    names = [n for n, _ in fused]
    assert len(names) == 3
    assert len(set(names)) == 3


def test_empty_bm25_returns_vector_order():
    fused = fuse_seed_rankings(["a", "b"], [], top_k=10)
    assert [n for n, _ in fused] == ["a", "b"]
    assert all(src == "vector" for _, src in fused)


def test_empty_both_returns_empty():
    assert fuse_seed_rankings([], [], top_k=10) == []
