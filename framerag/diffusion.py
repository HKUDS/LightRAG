"""Hypergraph Diffusion for FrameRAG retrieval.

Implements role-aware, multi-granularity diffusion across:
  ChunkNodes ↔ EventNodes ↔ FrameInstanceNodes ↔ EntityMentionNodes / InfoNodes

The algorithm is a Personalized PageRank variant adapted for heterogeneous
hypergraphs with FE-weighted edges.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse


def _row_normalize(M: sparse.csr_matrix) -> sparse.csr_matrix:
    """Row-normalize a sparse matrix: each row divided by its sum."""
    row_sums = np.array(M.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    D_inv = sparse.diags(1.0 / row_sums)
    return D_inv @ M


def _l1_normalize(v: np.ndarray) -> np.ndarray:
    s = v.sum()
    return v / s if s > 1e-12 else v


class HypergraphDiffusion:
    """Multi-granularity diffusion over the FrameRAG hypergraph.

    Usage:
        diffusion = HypergraphDiffusion(matrices)
        f_node, f_fi, f_chunk = diffusion.run(
            y_node, y_fi, y_chunk, fe_focus_mask, T=3
        )
    """

    def __init__(self, matrices: dict):
        """
        matrices: output of HypergraphStore.build_matrices()
          H_ce  [n_chunks × n_events]
          H_ef  [n_events × n_frames]
          H_fe  [n_frames × n_nodes]   already FE-weighted
          A_cau [n_events × n_events]
          + index lists: chunk_ids, event_ids, fi_ids, node_ids
          + index dicts: chunk_idx, event_idx, fi_idx, node_idx
        """
        self.matrices = matrices

        H_ce: sparse.csr_matrix = matrices["H_ce"]
        H_ef: sparse.csr_matrix = matrices["H_ef"]
        H_fe: sparse.csr_matrix = matrices["H_fe"]
        A_cau: sparse.csr_matrix = matrices["A_cau"]
        A_coref: sparse.csr_matrix = matrices.get(
            "A_coref", sparse.csr_matrix((H_fe.shape[1], H_fe.shape[1]))
        )

        # Row-normalized propagation matrices (source-row convention)
        self.H_fe_n   = _row_normalize(H_fe)              # [n_frames × n_nodes]  frame→node
        self.H_fe_T_n = _row_normalize(H_fe.T.tocsr())   # [n_nodes  × n_frames] node→frame
        self.H_ef_n   = _row_normalize(H_ef)              # [n_events × n_frames] event←frame
        self.H_ce_n   = _row_normalize(H_ce)              # [n_chunks × n_events] chunk←event
        self.A_cau_n  = _row_normalize(A_cau) if A_cau.nnz > 0 else A_cau
        self.A_coref_n = _row_normalize(A_coref) if A_coref.nnz > 0 else A_coref

        self.n_chunks = H_ce.shape[0]
        self.n_events = H_ce.shape[1]
        self.n_frames = H_ef.shape[1]
        self.n_nodes  = H_fe.shape[1]

    def _step_warmup(
        self,
        f_node: np.ndarray,
        f_fi: np.ndarray,
        f_chunk: np.ndarray,
        y_node: np.ndarray,
        y_fi: np.ndarray,
        y_chunk: np.ndarray,
        alpha: float,
        causal_weight: float,
        coref_weight: float = 0.3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fully bidirectional PPR step with state accumulation.

        Signal flows both ways (Node↔FI) AND each layer adds incoming signal
        to its current state before normalising — this is genuine multi-hop
        accumulation (unlike a plain overwrite).

        Co-reference expansion (Direction B): after the FI→Node backward pass,
        a fraction (coref_weight) of each mention's current score is shared with
        co-referent siblings via A_coref.  This lets a seeded mention of
        "Mr. Holmes" pull in signal from "Holmes" / "the detective" without
        completely collapsing them into one node.
        """
        f_fi_from_node  = self.H_fe_n   @ f_node   # Node  → FI  (forward)
        f_node_from_fi  = self.H_fe_T_n @ f_fi     # FI   → Node (backward)

        # Co-reference expansion: co-referent mentions share score proportionally
        if self.A_coref_n.nnz > 0:
            f_node_from_fi = f_node_from_fi + coref_weight * (self.A_coref_n @ f_node)

        f_event = self.H_ef_n @ f_fi               # FI   → Event
        if self.A_cau_n.nnz > 0:
            f_event = f_event + causal_weight * (self.A_cau_n @ f_event)
        f_chunk_from_ev = self.H_ce_n @ f_event    # Event → Chunk

        # PPR: new = (1-α)·l1(incoming + current) + α·seed
        # Adding current state to incoming ensures prior hops accumulate.
        f_node  = (1 - alpha) * _l1_normalize(f_node_from_fi  + f_node)  + alpha * y_node
        f_fi    = (1 - alpha) * _l1_normalize(f_fi_from_node  + f_fi)    + alpha * y_fi
        f_chunk = (1 - alpha) * _l1_normalize(f_chunk_from_ev + f_chunk) + alpha * y_chunk
        return f_node, f_fi, f_chunk

    def _step_cool(
        self,
        f_node_frozen: np.ndarray,
        f_fi: np.ndarray,
        f_chunk: np.ndarray,
        y_fi: np.ndarray,
        y_chunk: np.ndarray,
        w_back: float,
        alpha: float,
        causal_weight: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cooling step: node contribution decays, signal drains forward to chunks.

        The backward channel is gradually closed by scaling the *node addend*
        (not the result of l1-normalisation) — this is the correct fix so that
        w_back genuinely reduces node influence in the mixed signal:

            l1( w_back * node_signal + f_fi )   ← node shrinks relative to f_fi
        vs the previous (broken) form:
            l1( node_signal * w_back )           ← l1 cancels the scalar entirely

        Entity nodes are frozen at their warm-up values.  As w_back → 0 the
        FI layer decouples from entities and only propagates accumulated FI
        signal forward through Event → Chunk.
        """
        # Decaying node→FI contribution: scale the addend, not the normalised result
        f_fi_from_node = w_back * (self.H_fe_n @ f_node_frozen)

        f_event = self.H_ef_n @ f_fi
        if self.A_cau_n.nnz > 0:
            f_event = f_event + causal_weight * (self.A_cau_n @ f_event)
        f_chunk_from_ev = self.H_ce_n @ f_event

        # FI: node addend shrinks each step → node "freezes out" of the mixture
        f_fi    = (1 - alpha) * _l1_normalize(f_fi_from_node  + f_fi)    + alpha * y_fi
        f_chunk = (1 - alpha) * _l1_normalize(f_chunk_from_ev + f_chunk) + alpha * y_chunk
        return f_fi, f_chunk

    def run(
        self,
        y_node: np.ndarray,
        y_fi: np.ndarray,
        y_chunk: np.ndarray,
        alpha: float = 0.15,
        causal_weight: float = 0.3,
        warm_up_steps: int = 3,
        t_decay: float = 0.7,
        epsilon: float = 0.01,
        coref_weight: float = 0.3,
        T: int = 0,  # unused, kept for call-site compatibility
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Two-phase diffusion: warm-up (explore) → cooling (drain to chunks).

        Phase 1 — Warm-up (``warm_up_steps`` bidirectional steps):
            Entity ↔ FI ↔ Event ↔ Chunk all exchange signal freely.
            Each layer *accumulates* incoming + current state before
            normalising, so signal from 2-hop entity neighbourhoods builds
            up inside FI nodes before the backward channel closes.

        Phase 2 — Cooling (dynamic, terminates when w_back < epsilon):
            Entity nodes are frozen at their warm-up values.
            The node→FI addend is scaled by w_back = t_decay^k each step:

                f_fi ← l1( w_back·(H_fe @ f_node_frozen) + f_fi ) * (1-α) + α·y_fi

            Because w_back multiplies an *addend* (not the whole normalised
            vector), it genuinely reduces node influence relative to the
            existing f_fi signal — the effect that the previous implementation
            accidentally cancelled via l1(·).  As w_back → 0:
              • FI decouples from entities
              • accumulated FI signal drains forward: FI → Event → Chunk
              • chunks accumulate all signal that explored during warm-up

            With t_decay=0.7, epsilon=0.01: ~13 cooling steps.
            With t_decay=0.5, epsilon=0.01: ~ 7 cooling steps.

        Returns:
            f_node  [n_nodes]   entity+info scores (frozen after warm-up)
            f_fi    [n_frames]  frame instance scores
            f_chunk [n_chunks]  chunk scores — primary retrieval signal
        """
        f_node  = y_node.copy().astype(np.float64)
        f_fi    = y_fi.copy().astype(np.float64)
        f_chunk = y_chunk.copy().astype(np.float64)

        # ── Phase 1: Warm-up — fully bidirectional, accumulating ─────────────
        for _ in range(warm_up_steps):
            f_node, f_fi, f_chunk = self._step_warmup(
                f_node, f_fi, f_chunk, y_node, y_fi, y_chunk,
                alpha=alpha, causal_weight=causal_weight,
                coref_weight=coref_weight,
            )

        # ── Phase 2: Cooling — entity frozen, signal drains to chunks ────────
        f_node_frozen = f_node.copy()
        cool_step = 0
        while True:
            w_back = t_decay ** cool_step
            if w_back < epsilon:
                break
            f_fi, f_chunk = self._step_cool(
                f_node_frozen, f_fi, f_chunk, y_fi, y_chunk,
                w_back=w_back, alpha=alpha, causal_weight=causal_weight,
            )
            cool_step += 1

        return f_node, f_fi, f_chunk

    def top_results(
        self,
        f_node: np.ndarray,
        f_fi: np.ndarray,
        f_chunk: np.ndarray,
        top_chunks: int = 20,
        top_frames: int = 10,
        top_nodes: int = 15,
    ) -> dict:
        """Extract top-k indices and scores from diffusion output."""

        def _topk(scores: np.ndarray, k: int, ids: list[str]) -> list[dict]:
            k = min(k, len(ids))
            if k == 0:
                return []
            idxs = np.argpartition(scores, -k)[-k:]
            idxs = idxs[np.argsort(scores[idxs])[::-1]]
            return [{"id": ids[i], "score": float(scores[i])} for i in idxs
                    if scores[i] > 1e-10]

        return {
            "chunk_hits":  _topk(f_chunk, top_chunks, self.matrices["chunk_ids"]),
            "frame_hits":  _topk(f_fi,    top_frames, self.matrices["fi_ids"]),
            "node_hits":   _topk(f_node,  top_nodes,  self.matrices["node_ids"]),
        }
