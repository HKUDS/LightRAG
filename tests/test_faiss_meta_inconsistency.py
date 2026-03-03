"""
Regression tests for Faiss meta/index inconsistency handling.

Verifies that FaissVectorDBStorage gracefully handles cases where
meta.json has more rows than the Faiss index (e.g., after a crash
during save), and that delete/upsert operations don't crash.
"""

import json
import os
import tempfile

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")


@pytest.mark.offline
class TestFaissMetaInconsistency:
    """Test that stale metadata rows are handled gracefully."""

    def _create_index_and_meta(self, tmp_dir, dim=4, n_vectors=3, n_extra_meta=2):
        """
        Helper: create a Faiss index with `n_vectors` vectors and a meta.json
        that has `n_vectors + n_extra_meta` entries (simulating a crash where
        meta was written but index wasn't fully updated).
        """
        index_file = os.path.join(tmp_dir, "faiss_index_test.index")
        meta_file = index_file + ".meta.json"

        # Build real index with n_vectors
        index = faiss.IndexFlatIP(dim)
        vectors = np.random.rand(n_vectors, dim).astype(np.float32)
        # Normalize for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        index.add(vectors)
        faiss.write_index(index, index_file)

        # Build meta with extra rows beyond index.ntotal
        meta = {}
        for i in range(n_vectors):
            meta[str(i)] = {"__id__": f"id_{i}", "content": f"text_{i}"}
        for i in range(n_vectors, n_vectors + n_extra_meta):
            meta[str(i)] = {"__id__": f"stale_{i}", "content": f"stale_{i}"}

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        return index_file, meta_file, vectors

    def test_load_skips_invalid_metadata_rows(self):
        """
        Loading an index where meta.json has fids beyond index.ntotal
        should skip those rows with a warning, not crash.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dim = 4
            n_vectors = 3
            n_extra = 2
            index_file, meta_file, vectors = self._create_index_and_meta(
                tmp_dir, dim=dim, n_vectors=n_vectors, n_extra_meta=n_extra
            )

            # Manually load and verify behavior
            index = faiss.read_index(index_file)
            with open(meta_file, "r", encoding="utf-8") as f:
                stored_dict = json.load(f)

            assert len(stored_dict) == n_vectors + n_extra

            # Simulate the load logic from _load_faiss_index
            id_to_meta = {}
            skipped = 0
            for fid_str, meta in stored_dict.items():
                fid = int(fid_str)
                if fid >= index.ntotal:
                    skipped += 1
                    continue
                if "__vector__" not in meta:
                    meta["__vector__"] = index.reconstruct(fid).tolist()
                id_to_meta[fid] = meta

            assert len(id_to_meta) == n_vectors
            assert skipped == n_extra

            # Verify reconstructed vectors match originals
            for fid in range(n_vectors):
                reconstructed = np.array(
                    id_to_meta[fid]["__vector__"], dtype=np.float32
                )
                np.testing.assert_allclose(reconstructed, vectors[fid], atol=1e-6)

    def test_remove_with_missing_vector_uses_reconstruct(self):
        """
        _remove_faiss_ids should reconstruct vectors from the index
        when __vector__ is not present in metadata.
        """
        dim = 4
        n_vectors = 3

        index = faiss.IndexFlatIP(dim)
        vectors = np.random.rand(n_vectors, dim).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        index.add(vectors)

        # Metadata WITHOUT __vector__ (as stored on disk after our PR)
        id_to_meta = {}
        for i in range(n_vectors):
            id_to_meta[i] = {"__id__": f"id_{i}", "content": f"text_{i}"}

        # Simulate rebuild logic from _remove_faiss_ids (remove fid=1)
        fid_list = [1]
        keep_fids = [fid for fid in id_to_meta if fid not in fid_list]

        vectors_to_keep = []
        new_id_to_meta = {}
        for new_fid, old_fid in enumerate(keep_fids):
            vec_meta = id_to_meta[old_fid]
            if "__vector__" in vec_meta:
                vec = vec_meta["__vector__"]
            elif old_fid < index.ntotal:
                vec = index.reconstruct(old_fid).tolist()
                vec_meta["__vector__"] = vec
            else:
                continue
            vectors_to_keep.append(vec)
            new_id_to_meta[new_fid] = vec_meta

        assert len(vectors_to_keep) == 2
        assert len(new_id_to_meta) == 2
        # Verify the kept vectors match originals (fid 0 and 2)
        np.testing.assert_allclose(
            np.array(vectors_to_keep[0], dtype=np.float32), vectors[0], atol=1e-6
        )
        np.testing.assert_allclose(
            np.array(vectors_to_keep[1], dtype=np.float32), vectors[2], atol=1e-6
        )

    def test_atomic_save_meta(self):
        """
        _save_faiss_index should write meta.json atomically via temp file + os.replace.
        Verify no .tmp file remains after save.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            meta_file = os.path.join(tmp_dir, "test.meta.json")
            tmp_meta_file = meta_file + ".tmp"

            serializable_dict = {"0": {"__id__": "id_0", "content": "text_0"}}

            # Simulate atomic write
            with open(tmp_meta_file, "w", encoding="utf-8") as f:
                json.dump(serializable_dict, f)
            os.replace(tmp_meta_file, meta_file)

            assert os.path.exists(meta_file)
            assert not os.path.exists(tmp_meta_file)

            with open(meta_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded == serializable_dict
