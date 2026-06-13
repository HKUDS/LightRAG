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

        # Row-normalized versions for propagation
        # Direction: "source → target" means row is source, col is target
        # H_ce row=chunk, col=event  → chunk to event: H_ce
        # event to chunk: H_ce.T
        self.H_ce_T_n = _row_normalize(H_ce.T.tocsr())   # [n_events × n_chunks]
        self.H_ef_T_n = _row_normalize(H_ef.T.tocsr())   # [n_frames × n_events]
        self.H_fe_n   = _row_normalize(H_fe)              # [n_frames × n_nodes]
        self.H_fe_T_n = _row_normalize(H_fe.T.tocsr())   # [n_nodes  × n_frames]
        self.H_ef_n   = _row_normalize(H_ef)              # [n_events × n_frames]
        self.H_ce_n   = _row_normalize(H_ce)              # [n_chunks × n_events]
        self.A_cau_n  = _row_normalize(A_cau) if A_cau.nnz > 0 else A_cau

        self.n_chunks = H_ce.shape[0]
        self.n_events = H_ce.shape[1]
        self.n_frames = H_ef.shape[1]
        self.n_nodes  = H_fe.shape[1]

    def run(
        self,
        y_node: np.ndarray,    # [n_nodes]   initial seed scores for entity+info nodes
        y_fi: np.ndarray,      # [n_frames]  initial seed scores for frame instances
        y_chunk: np.ndarray,   # [n_chunks]  initial seed scores for chunks
        alpha: float = 0.15,
        T: int = 3,
        causal_weight: float = 0.3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run T iterations of personalized diffusion.

        Returns:
            f_node  [n_nodes]   final scores for entity+info nodes
            f_fi    [n_frames]  final scores for frame instances
            f_chunk [n_chunks]  final scores for chunks
        """
        f_node  = y_node.copy().astype(np.float64)
        f_fi    = y_fi.copy().astype(np.float64)
        f_chunk = y_chunk.copy().astype(np.float64)

        for _ in range(T):
            # ── Step A: Node → FrameInstance ──
            # Each frame instance collects scores from its entity/info fillers
            # f_fi_new[fi] = sum_e( H_fe[fi,e] * f_node[e] )
            # H_fe is already FE-weighted (core > noncore, focus-boosted)
            f_fi_from_node = self.H_fe_n @ f_node          # [n_frames]

            # ── Step B: FrameInstance → Node ──
            # Each entity/info node collects from frame instances it participates in
            # f_node_new[e] = sum_fi( H_fe_T[e,fi] * f_fi[fi] )
            f_node_from_fi = self.H_fe_T_n @ f_fi          # [n_nodes]

            # ── Step C: FrameInstance → Event → Chunk ──
            # Frame → Event
            f_event = self.H_ef_T_n @ f_fi                 # [n_events]

            # Causal propagation: if ev_A scores high, ev_B (consequence) benefits
            if self.A_cau_n.nnz > 0:
                f_event_causal = self.A_cau_n @ f_event    # [n_events]
                f_event = f_event + causal_weight * f_event_causal

            # Event → Chunk
            f_chunk_from_ev = self.H_ce_T_n @ f_event      # [n_chunks]

            # ── Step D: Combine ──
            f_fi_new    = f_fi_from_node + f_fi             # merge signals
            f_node_new  = f_node_from_fi + f_node
            f_chunk_new = f_chunk_from_ev + f_chunk

            # ── Step E: Personalized PageRank restart ──
            # Keep alpha fraction anchored to seed — prevents drift from query
            f_node  = (1 - alpha) * _l1_normalize(f_node_new)  + alpha * y_node
            f_fi    = (1 - alpha) * _l1_normalize(f_fi_new)    + alpha * y_fi
            f_chunk = (1 - alpha) * _l1_normalize(f_chunk_new) + alpha * y_chunk

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
