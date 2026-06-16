"""End-to-end pipeline tests with mock LLM + embed (see conftest.py).

Covers the bugs fixed in this round:
  #1 event description carries FE roles
  #2 frame names normalized (no spaces) end-to-end
  #3/#4 event_hints actually drive event-VDB seeding
  Direction B mention-level nodes + retrieval correctness
"""
from __future__ import annotations

async def test_index_produces_expected_counts(indexed_rag):
    stats = await indexed_rag.get_stats()
    assert stats["chunks"] >= 1
    assert stats["entity_mentions"] == 3        # Holmes, Watson, violin
    assert stats["events"] == 2                 # play, undisturbed
    assert stats["frame_instances"] == 2


async def test_bug1_event_description_has_fe_roles(indexed_rag):
    hg = indexed_rag._hg
    descs = []
    for eid in hg._event_ids:
        e = await hg.events.get_by_id(eid)
        descs.append(e["event_description"])
    joined = " ".join(descs)
    # Roles must be embedded into the event description.
    assert "Agent: Sherlock Holmes" in joined
    assert "Instrument: violin" in joined


async def test_bug2_frame_names_have_no_spaces(indexed_rag):
    hg = indexed_rag._hg
    names = []
    for fid in hg._fi_ids:
        f = await hg.frame_instances.get_by_id(fid)
        names.append(f["frame_name"])
    assert "Music_Performance" in names          # space → underscore
    assert all(" " not in n for n in names)


async def test_mention_level_nodes_in_matrices(indexed_rag):
    # Direction B: node layer must be mentions + info nodes, never canonicals.
    matrices = await indexed_rag._hg.build_matrices()
    node_ids = set(matrices["node_ids"])
    assert hg_mentions(indexed_rag) <= node_ids
    # canonical ids must NOT appear as graph nodes
    assert not (set(indexed_rag._hg._canonical_ids) & node_ids)


def hg_mentions(rag):
    return set(rag._hg._mention_ids)


async def test_acoref_matrix_present(indexed_rag):
    matrices = await indexed_rag._hg.build_matrices()
    assert "A_coref" in matrices
    n = len(matrices["node_ids"])
    assert matrices["A_coref"].shape == (n, n)


async def test_retrieval_surfaces_music_frame(indexed_rag):
    ctx = await indexed_rag.aretrieve_context(
        "What instrument did the detective play to relax?"
    )
    assert ctx["chunk_hits"], "should retrieve at least one chunk"
    assert ctx["frame_hits"], "should retrieve at least one frame instance"

    hg = indexed_rag._hg
    music_fi = None
    for fid in hg._fi_ids:
        f = await hg.frame_instances.get_by_id(fid)
        if f["frame_name"].startswith("Music"):
            music_fi = fid
    assert any(h["id"] == music_fi for h in ctx["frame_hits"])


async def test_event_hint_seeding_changes_scores(indexed_rag):
    """#3/#4: with event_hints active, the event VDB is queried and the matched
    event's frame instance gets extra weight. Compare event_hint_weight=0 vs >0."""
    rag = indexed_rag
    q = "What instrument did the detective play to relax?"

    rag._event_hint_weight = 0.0
    base = await rag.aretrieve_context(q)
    rag._event_hint_weight = 2.0
    boosted = await rag.aretrieve_context(q)

    def score_of(ctx, prefix):
        for h in ctx["frame_hits"]:
            return h  # top hit
        return None

    # The top frame hit's score should differ once event seeding is on.
    base_scores = {h["id"]: h["score"] for h in base["frame_hits"]}
    boosted_scores = {h["id"]: h["score"] for h in boosted["frame_hits"]}
    assert base_scores != boosted_scores


async def test_answer_generation_runs(indexed_rag):
    ans = await indexed_rag.aquery("What did Holmes play?")
    assert isinstance(ans, str) and len(ans) > 0
