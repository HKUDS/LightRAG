"""PR-1 L2 tests: resolve im-ids via sidecar ``drawings.json`` on disk.

After L1 collects im-ids, L2 loads the sidecar index and maps each id to a
local asset path plus attachment metadata (title, heading, format). These tests
use the GB/T 70.3 parsed sidecar fixture when present.

Coverage:
- ``load_drawings_index`` — build im-id → metadata map from ``*.drawings.json``
- ``resolve_asset_path`` — join relative sidecar paths to on-disk jpg files
- ``resolve_single_drawing`` — produce a full attachment dict for one drawing
- ``resolve_drawing_asset_file`` — async entry that looks up ``full_docs.sidecar_location``

Golden target: ``GB_T_IM_ID_0002`` (counterbore head dimension figure).
"""

from __future__ import annotations

import json

import pytest

from lightrag.sidecar.query_attachments import (
    clear_drawings_index_cache,
    load_drawings_index,
    resolve_asset_path,
    resolve_single_drawing,
)
from tests.sidecar.conftest_query_attachments import (
    FakeFullDocs,
    GB_T_DOC_ID,
    GB_T_IM_ID_0002,
    GB_T_IM_0002_TITLE,
    GB_T_JPG_SUFFIX,
    GB_T_PARSED_DIR,
)

# Same doc hash as collect/conftest; _IM_0001 also exists in the GB/T drawings index.
_DOC_HASH = "116da71bde652ed78cf28934d0712aa6"
_IM_0001 = f"im-{_DOC_HASH}-0001"


@pytest.fixture(autouse=True)
def _clear_drawings_cache():
    """Clear the process-local drawings.json cache before and after each case."""
    clear_drawings_index_cache()
    yield
    clear_drawings_index_cache()


@pytest.mark.offline
class TestDrawingsResolve:
    """Sidecar index loading and per-drawing attachment resolution."""

    def test_load_drawings_index(self, gb_t_sidecar_uri):
        """Return an im-id keyed dict; golden entry path ends with the expected jpg."""
        index = load_drawings_index(gb_t_sidecar_uri)
        assert _IM_0001 in index
        assert GB_T_IM_ID_0002 in index
        assert GB_T_JPG_SUFFIX in index[GB_T_IM_ID_0002]["path"]

    def test_resolve_path_ffda008de9ce(self, gb_t_sidecar_uri):
        """Resolve a relative sidecar path to an existing local asset file."""
        index = load_drawings_index(gb_t_sidecar_uri)
        rel = index[GB_T_IM_ID_0002]["path"]
        asset = resolve_asset_path(gb_t_sidecar_uri, rel)
        assert asset is not None
        assert asset.name == GB_T_JPG_SUFFIX
        assert asset.is_file()

    def test_resolve_single_drawing_golden(self, gb_t_sidecar_uri):
        """Golden im-0002 resolves with full attachment fields including heading."""
        index = load_drawings_index(gb_t_sidecar_uri)
        item = resolve_single_drawing(
            im_id=GB_T_IM_ID_0002,
            doc_id=GB_T_DOC_ID,
            sidecar_uri=gb_t_sidecar_uri,
            drawings_index=index,
        )
        assert item is not None
        assert item["im_id"] == GB_T_IM_ID_0002
        assert item["doc_id"] == GB_T_DOC_ID
        assert item["type"] == "drawing"
        assert GB_T_JPG_SUFFIX in item["path"]
        assert item["title"] == GB_T_IM_0002_TITLE
        assert item["heading"] == "4.1 型式尺寸"

    @pytest.mark.asyncio
    async def test_resolve_drawing_asset_file_golden(self, gb_t_sidecar_uri):
        """Async helper returns (Path, media_type) via FakeFullDocs sidecar lookup."""
        from lightrag.sidecar.query_attachments import resolve_drawing_asset_file

        resolved = await resolve_drawing_asset_file(
            GB_T_DOC_ID,
            GB_T_IM_ID_0002,
            FakeFullDocs(gb_t_sidecar_uri),
        )
        assert resolved is not None
        asset_path, media_type = resolved
        assert asset_path.name == GB_T_JPG_SUFFIX
        assert media_type == "image/jpeg"

    def test_missing_im_id_skipped(self, gb_t_sidecar_uri):
        """Unknown im-ids return None instead of raising."""
        index = load_drawings_index(gb_t_sidecar_uri)
        assert (
            resolve_single_drawing(
                im_id="im-deadbeefdeadbeefdeadbeefdeadbeef-9999",
                doc_id=GB_T_DOC_ID,
                sidecar_uri=gb_t_sidecar_uri,
                drawings_index=index,
            )
            is None
        )

    def test_missing_sidecar_returns_empty_index(self):
        """None or remote-only URIs yield an empty index without errors."""
        assert load_drawings_index(None) == {}
        assert load_drawings_index("s3://bucket/x/") == {}

    def test_missing_jpg_skipped(self, gb_t_sidecar_uri, tmp_path):
        """When verify_file=True and the jpg is absent, resolution returns None."""
        # Synthetic index entry pointing at a path that does not exist on disk.
        index = {
            _IM_0001: {
                "id": _IM_0001,
                "format": "jpg",
                "path": "missing.blocks.assets/no-such-file.jpg",
            }
        }
        assert (
            resolve_single_drawing(
                im_id=_IM_0001,
                doc_id=GB_T_DOC_ID,
                sidecar_uri=gb_t_sidecar_uri,
                drawings_index=index,
            )
            is None
        )

    def test_drawings_json_readable(self):
        """Self-check: GB/T fixture directory contains a readable drawings.json payload."""
        if not GB_T_PARSED_DIR.is_dir():
            pytest.skip("GB/T fixture missing")
        drawings_files = list(GB_T_PARSED_DIR.glob("*.drawings.json"))
        assert drawings_files
        payload = json.loads(drawings_files[0].read_text(encoding="utf-8"))
        assert "drawings" in payload
