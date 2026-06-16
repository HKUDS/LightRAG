"""Tests for entity coreference type-guard (#6) and Direction-B diffusion."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from framerag.diffusion import HypergraphDiffusion
from framerag.coreference.entity_coref import EntityCoreferenceResolver
from framerag.types import EntityMentionSchema
from .conftest import mock_embed, mock_llm


# ── #6: coreference resolves co-referent mentions, keeps distinct ones apart ──

async def test_coref_merges_same_entity():
    resolver = EntityCoreferenceResolver(mock_embed, mock_llm, enable_llm_verify=False)
    mentions = [
        # "Holmes" appears as an alias on the full-name mention, so blocking
        # places both in the same "holmes" block for comparison.
        EntityMentionSchema("m1", "c1", "Sherlock Holmes", "PERSON",
                            "the detective at baker street", ["Holmes"], "HIGH"),
        EntityMentionSchema("m2", "c2", "Holmes", "PERSON",
                            "the detective at baker street", [], "HIGH"),
    ]
    canons = await resolver.resolve_with_llm_verify(mentions)
    # same description → same embedding → should merge into one canonical
    assert len(canons) == 1
    assert {m.canonical_id for m in mentions} == {canons[0].canonical_id}


async def test_coref_keeps_different_entities_apart():
    resolver = EntityCoreferenceResolver(mock_embed, mock_llm, enable_llm_verify=False)
    mentions = [
        EntityMentionSchema("m1", "c1", "Holmes", "PERSON",
                            "a brilliant consulting detective", [], "HIGH"),
        EntityMentionSchema("m2", "c2", "Moriarty", "PERSON",
                            "a criminal mastermind professor", [], "HIGH"),
    ]
    canons = await resolver.resolve_with_llm_verify(mentions)
    assert len(canons) == 2


# ── Direction B: A_coref propagates score to co-referent siblings ─────────────

def _toy_matrices(with_coref: bool):
    # 2 chunks, 2 events, 3 FIs, nodes = [m0, m1, m2, info0]
    n_c, n_e, n_f, n_n = 2, 2, 3, 4
    H_ce = sparse.csr_matrix(([1, 1], ([0, 1], [0, 1])), shape=(n_c, n_e))
    H_ef = sparse.csr_matrix(([1, 1, 1], ([0, 0, 1], [0, 1, 2])), shape=(n_e, n_f))
    H_fe = sparse.csr_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=(n_f, n_n))
    A_cau = sparse.csr_matrix((n_e, n_e))
    if with_coref:
        # m0 <-> m1 are co-referent
        A_coref = sparse.csr_matrix(([1, 1], ([0, 1], [1, 0])), shape=(n_n, n_n))
    else:
        A_coref = sparse.csr_matrix((n_n, n_n))
    return {
        "H_ce": H_ce, "H_ef": H_ef, "H_fe": H_fe,
        "A_cau": A_cau, "A_coref": A_coref,
        "chunk_ids": ["c0", "c1"], "event_ids": ["e0", "e1"],
        "fi_ids": ["f0", "f1", "f2"], "node_ids": ["m0", "m1", "m2", "i0"],
        "chunk_idx": {"c0": 0, "c1": 1}, "event_idx": {"e0": 0, "e1": 1},
        "fi_idx": {"f0": 0, "f1": 1, "f2": 2},
        "node_idx": {"m0": 0, "m1": 1, "m2": 2, "i0": 3},
    }


def test_acoref_lifts_sibling_score():
    y_node = np.array([1.0, 0.0, 0.0, 0.0])   # seed only m0
    y_fi = np.zeros(3)
    y_chunk = np.zeros(2)

    no = HypergraphDiffusion(_toy_matrices(with_coref=False)).run(
        y_node.copy(), y_fi.copy(), y_chunk.copy(), warm_up_steps=3, coref_weight=0.3)
    yes = HypergraphDiffusion(_toy_matrices(with_coref=True)).run(
        y_node.copy(), y_fi.copy(), y_chunk.copy(), warm_up_steps=3, coref_weight=0.3)

    m1_no = no[0][1]
    m1_yes = yes[0][1]
    # With co-ref edges, the un-seeded sibling m1 must end up with MORE score.
    assert m1_yes > m1_no


def test_diffusion_outputs_are_finite_and_nonneg():
    y_node = np.array([0.5, 0.0, 0.5, 0.0])
    f_node, f_fi, f_chunk = HypergraphDiffusion(_toy_matrices(True)).run(
        y_node, np.zeros(3), np.zeros(2), warm_up_steps=3)
    for v in (f_node, f_fi, f_chunk):
        assert np.all(np.isfinite(v))
        assert np.all(v >= -1e-9)
