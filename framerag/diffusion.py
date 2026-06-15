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

        # Row-normalized propagation matrices (source-row convention)
        self.H_fe_n   = _row_normalize(H_fe)              # [n_frames × n_nodes]  frame→node
        self.H_fe_T_n = _row_normalize(H_fe.T.tocsr())   # [n_nodes  × n_frames] node→frame
        self.H_ef_n   = _row_normalize(H_ef)              # [n_events × n_frames] event←frame
        self.H_ce_n   = _row_normalize(H_ce)              # [n_chunks × n_events] chunk←event
        self.A_cau_n  = _row_normalize(A_cau) if A_cau.nnz > 0 else A_cau

        self.n_chunks = H_ce.shape[0]
        self.n_events = H_ce.shape[1]
        self.n_frames = H_ef.shape[1]
        self.n_nodes  = H_fe.shape[1]

    def _step(
        self,
        f_node: np.ndarray,
        f_fi: np.ndarray,
        f_chunk: np.ndarray,
        y_node: np.ndarray,
        y_fi: np.ndarray,
        y_chunk: np.ndarray,
        alpha: float,
        w_back: float,
        causal_weight: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Single PPR propagation step."""
        # Node → FI (forward)
        f_fi_from_node = self.H_fe_n @ f_node
        # FI → Node (backward, weighted by w_back)
        f_node_from_fi = self.H_fe_T_n @ f_fi * w_back
        # FI → Event → Chunk (forward)
        f_event = self.H_ef_n @ f_fi
        if self.A_cau_n.nnz > 0:
            f_event = f_event + causal_weight * (self.A_cau_n @ f_event)
        f_chunk_from_ev = self.H_ce_n @ f_event

        # PPR restart (node restart also scales with w_back — less anchoring as we cool)
        f_node  = (1 - alpha) * _l1_normalize(f_node_from_fi)  + alpha * w_back * y_node
        f_fi    = (1 - alpha) * _l1_normalize(f_fi_from_node)  + alpha * y_fi
        f_chunk = (1 - alpha) * _l1_normalize(f_chunk_from_ev) + alpha * y_chunk
        return f_node, f_fi, f_chunk

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
        # legacy param — ignored; steps now determined dynamically
        T: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Two-phase dynamic diffusion over the hypergraph.

        Phase 1 — Warm-up (``warm_up_steps`` steps, w_back = 1.0):
            Fully bidirectional. Entity ↔ FI ↔ Event ↔ Chunk exchange signal
            freely. Weak entity seeds (low initial cosine similarity) get a
            chance to propagate through the FI neighborhood and accumulate score
            before the backward channel closes. Analogous to high temperature
            in simulated annealing — exploration dominates.

        Phase 2 — Cooling (dynamic, w_back = t_decay^k until w_back < epsilon):
            Backward weight decays geometrically: w_back(k) = t_decay^k.
            The loop terminates as soon as w_back falls below ``epsilon``.
            Entity nodes "freeze" while accumulated signal drains forward
            through FI → Event → Chunk. Terminates automatically — no fixed T.

            With t_decay=0.7, epsilon=0.01: ceil(log(0.01)/log(0.7)) ≈ 13 steps.
            With t_decay=0.5, epsilon=0.01: ceil(log(0.01)/log(0.5)) ≈  7 steps.

        Total steps = warm_up_steps + cooling_steps (both determined by params).

        Args:
            warm_up_steps: Number of fully-bidirectional exploration steps.
                           Suggested: 3 — enough for 2-hop entity neighborhoods
                           to contribute before cooling.
            t_decay:  Geometric cooling rate (0 < t_decay < 1).
            epsilon:  Cooling stops when w_back < epsilon.

        Returns:
            f_node  [n_nodes]  — entity+info scores (near 0 after cooling)
            f_fi    [n_frames] — frame instance scores
            f_chunk [n_chunks] — chunk scores (primary retrieval signal)
        """
        f_node  = y_node.copy().astype(np.float64)
        f_fi    = y_fi.copy().astype(np.float64)
        f_chunk = y_chunk.copy().astype(np.float64)

        # ── Phase 1: Warm-up (full bidirectional) ────────────────────────────
        for _ in range(warm_up_steps):
            f_node, f_fi, f_chunk = self._step(
                f_node, f_fi, f_chunk, y_node, y_fi, y_chunk,
                alpha=alpha, w_back=1.0, causal_weight=causal_weight,
            )

        # ── Phase 2: Cooling (dynamic — stop when w_back < epsilon) ──────────
        cool_step = 0
        while True:
            w_back = t_decay ** cool_step
            if w_back < epsilon:
                break
            f_node, f_fi, f_chunk = self._step(
                f_node, f_fi, f_chunk, y_node, y_fi, y_chunk,
                alpha=alpha, w_back=w_back, causal_weight=causal_weight,
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
