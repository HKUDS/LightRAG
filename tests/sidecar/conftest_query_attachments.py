"""Shared fixtures for query attachment resolver tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from lightrag.utils_pipeline import sidecar_uri_for

# Repo root: tests/sidecar/ -> LightRAG/ -> contributor/
_CONTRIBUTOR_ROOT = Path(__file__).resolve().parents[3]
GB_T_PARSED_DIR = (
    _CONTRIBUTOR_ROOT
    / "inputs"
    / "__parsed__"
    / "MinerU_markdown_GB_T_70.3-2023_3374_2089563653659709440.md.parsed"
)

# GB/T 70.3 fixture document id (stable hash from parsed sidecar tree).
GB_T_DOC_ID = "doc-116da71bde652ed78cf28934d0712aa6"
GB_T_IM_ID_0002 = "im-116da71bde652ed78cf28934d0712aa6-0002"
GB_T_IM_0002_TITLE = "内六角沉头螺钉头部尺寸标注图"
GB_T_JPG_SUFFIX = "image-ffda008de9ce.jpg"


@pytest.fixture
def gb_t_sidecar_uri() -> str:
    if not GB_T_PARSED_DIR.is_dir():
        pytest.skip(f"GB/T sidecar fixture missing: {GB_T_PARSED_DIR}")
    return sidecar_uri_for(GB_T_PARSED_DIR)


class FakeFullDocs:
    """Minimal full_docs stub for sidecar resolution tests."""

    def __init__(self, sidecar_uri: str):
        self._uri = sidecar_uri

    async def get_by_id(self, doc_id: str):
        if doc_id == GB_T_DOC_ID:
            return {"sidecar_location": self._uri}
        return None


REQUIRED_ATTACHMENT_KEYS = frozenset(
    {"im_id", "doc_id", "type", "format", "path"}
)


def assert_valid_attachment(item: dict) -> None:
    assert REQUIRED_ATTACHMENT_KEYS <= item.keys()
    assert item["type"] == "drawing"
    assert str(item["im_id"]).startswith("im-")
    assert str(item["doc_id"]).startswith("doc-")

