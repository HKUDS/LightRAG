"""Resolve toolbox tests: drawings.json load, asset path, single-drawing resolve.

Shared by Index (JSON filter, no verify) and Enrich (resolve + verify_file).
Uses synthetic sidecar under ``tmp_path`` (see ``conftest_query_attachments``).
"""

from __future__ import annotations

import concurrent.futures
import json

import pytest

from lightrag.sidecar.query_attachments import (
    clear_drawings_index_cache,
    load_drawings_index,
    resolve_asset_path,
    resolve_single_drawing,
)
from tests.sidecar.conftest_query_attachments import (
    SYNTH_DOC_ID,
    SYNTH_IM_0001,
    SYNTH_IM_0002,
    SYNTH_IM_0002_TITLE,
    SYNTH_JPG_1,
    SYNTH_JPG_2,
    corrupt_drawings_json,
)


@pytest.fixture(autouse=True)
def _clear_drawings_cache():
    """Isolate process-global drawings LRU cache between tests (mtime / corrupt cases)."""
    clear_drawings_index_cache()
    yield
    clear_drawings_index_cache()


@pytest.mark.offline
class TestDrawingsResolve:
    """Unit coverage for load / path / resolve helpers (no Index or Enrich orchestration)."""

    def test_load_drawings_index(self, synthetic_sidecar_uri):
        """Happy path: load ``drawings`` map and see both synthetic im-ids + paths."""
        index = load_drawings_index(synthetic_sidecar_uri)
        assert SYNTH_IM_0001 in index
        assert SYNTH_IM_0002 in index
        assert SYNTH_JPG_2 in index[SYNTH_IM_0002]["path"]

    def test_resolve_path(self, synthetic_sidecar_uri):
        """Relative path from drawings.json resolves to an on-disk jpg under sidecar root."""
        index = load_drawings_index(synthetic_sidecar_uri)
        rel = index[SYNTH_IM_0002]["path"]
        asset = resolve_asset_path(synthetic_sidecar_uri, rel)
        assert asset is not None
        assert asset.name == SYNTH_JPG_2
        assert asset.is_file()

    @pytest.mark.asyncio
    async def test_resolve_single_drawing(self, synthetic_sidecar_uri):
        """Build one attachment dict (im_id/doc_id/type/path + title/heading) with verify."""
        index = load_drawings_index(synthetic_sidecar_uri)
        item = await resolve_single_drawing(
            im_id=SYNTH_IM_0002,
            doc_id=SYNTH_DOC_ID,
            sidecar_uri=synthetic_sidecar_uri,
            drawings_index=index,
        )
        assert item is not None
        assert item["im_id"] == SYNTH_IM_0002
        assert item["doc_id"] == SYNTH_DOC_ID
        assert item["type"] == "drawing"
        assert SYNTH_JPG_2 in item["path"]
        assert item["title"] == SYNTH_IM_0002_TITLE
        assert item["heading"] == "Section B"

    @pytest.mark.asyncio
    async def test_missing_im_id_skipped(self, synthetic_sidecar_uri):
        """im-id absent from drawings index → resolve returns None (skip that attachment)."""
        index = load_drawings_index(synthetic_sidecar_uri)
        assert (
            await resolve_single_drawing(
                im_id="im-deadbeefdeadbeefdeadbeefdeadbeef-9999",
                doc_id=SYNTH_DOC_ID,
                sidecar_uri=synthetic_sidecar_uri,
                drawings_index=index,
            )
            is None
        )

    def test_missing_sidecar_returns_empty_index(self):
        """No URI or non-local URI → empty drawings map (no throw)."""
        assert load_drawings_index(None) == {}
        assert load_drawings_index("s3://bucket/x/") == {}

    def test_load_drawings_index_bad_json_returns_empty(
        self, synthetic_sidecar, synthetic_sidecar_uri
    ):
        """Corrupt drawings.json → ``{}`` (same degrade Index uses for empty whitelist)."""
        corrupt_drawings_json(synthetic_sidecar)
        assert load_drawings_index(synthetic_sidecar_uri) == {}

    def test_load_drawings_index_concurrent(self, synthetic_sidecar_uri):
        """Thread-safe cache: many concurrent loads return consistent indexes."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(
                    lambda _: load_drawings_index(synthetic_sidecar_uri),
                    range(32),
                )
            )
        assert len(results) == 32
        assert all(SYNTH_IM_0001 in index for index in results)
        assert all(SYNTH_IM_0002 in index for index in results)

    def test_cache_invalidates_when_drawings_mtime_changes(
        self, synthetic_sidecar, synthetic_sidecar_uri
    ):
        """Rewriting drawings.json (mtime change) must not keep the old cached map."""
        parsed_dir = synthetic_sidecar["parsed_dir"]
        stem = synthetic_sidecar["doc_id"]
        drawings_path = parsed_dir / f"{stem}.drawings.json"

        first = load_drawings_index(synthetic_sidecar_uri)
        assert SYNTH_IM_0001 in first

        # Empty drawings map on disk → next load must refresh, not return first hit.
        payload = {"drawings": {}}
        drawings_path.write_text(json.dumps(payload), encoding="utf-8")
        second = load_drawings_index(synthetic_sidecar_uri)
        assert second == {}

    @pytest.mark.asyncio
    async def test_missing_jpg_skipped(self, synthetic_sidecar_uri):
        """Enrich verify_file: entry has path but file missing → None (Index would still whitelist)."""
        # In-memory index only — path points at a non-existent asset under the sidecar.
        index = {
            SYNTH_IM_0001: {
                "id": SYNTH_IM_0001,
                "format": "jpg",
                "path": "assets/no-such-file.jpg",
            }
        }
        assert (
            await resolve_single_drawing(
                im_id=SYNTH_IM_0001,
                doc_id=SYNTH_DOC_ID,
                sidecar_uri=synthetic_sidecar_uri,
                drawings_index=index,
            )
            is None
        )

    def test_resolve_asset_path_embedded_nul_returns_none(self, synthetic_sidecar_uri):
        """Malformed drawings.json path with NUL must not raise (Enrich fail-loud seam)."""
        assert (
            resolve_asset_path(synthetic_sidecar_uri, f"assets/{SYNTH_JPG_2}\x00evil")
            is None
        )
        assert resolve_asset_path(synthetic_sidecar_uri, "\x00") is None

    @pytest.mark.asyncio
    async def test_resolve_single_drawing_embedded_nul_path_skipped(
        self, synthetic_sidecar_uri
    ):
        """verify_file: path with embedded NUL → None (skip attachment, no raise)."""
        index = {
            SYNTH_IM_0001: {
                "id": SYNTH_IM_0001,
                "format": "jpg",
                "path": f"assets/{SYNTH_JPG_1}\x00evil",
            }
        }
        assert (
            await resolve_single_drawing(
                im_id=SYNTH_IM_0001,
                doc_id=SYNTH_DOC_ID,
                sidecar_uri=synthetic_sidecar_uri,
                drawings_index=index,
            )
            is None
        )
