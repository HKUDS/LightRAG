"""Regression tests: html_url derivation for paper-lib file paths.

Entities/relations extracted before html_url was added to the extraction
pipeline have no stored html_url. convert_to_user_format must derive it
from file_path at query time so existing data links back to the original
paper without reprocessing.
"""
import pytest

from lightrag import utils
from lightrag.utils import (
    convert_to_user_format,
    derive_paper_html_url,
    normalize_paper_html_url,
)

pytestmark = pytest.mark.offline

PAPER_FILE = "papers_for_rag/R_D_Strategies_sha256_ea73d268fd4a7.md"
PAPER_URL = "https://paper-lib.huamedicine.com/papers/sha256_ea73d268fd4a7.html"
FULL_ID = "sha256_ea73d268fd4a75488b9e8fdc15e5fae09b3c67d7a5793f5c097f8a4237971cce"
FULL_URL = f"https://paper-lib.huamedicine.com/papers/{FULL_ID}.html"


@pytest.fixture
def paper_id_lookup():
    """Register a short->full paper_id lookup; reset after the test."""
    utils.register_paper_id_lookup({"sha256_ea73d268fd4a7": FULL_ID})
    yield
    utils.register_paper_id_lookup({})


def test_derive_paper_html_url_matches_sha256_marker():
    assert derive_paper_html_url(PAPER_FILE) == PAPER_URL


def test_derive_paper_html_url_no_marker_returns_empty():
    assert derive_paper_html_url("random_doc.md") == ""
    assert derive_paper_html_url("") == ""


def test_derive_expands_short_id_via_lookup(paper_id_lookup):
    """Truncated filename markers resolve to the full paper_id when mapped."""
    assert derive_paper_html_url(PAPER_FILE) == FULL_URL


def test_derive_unknown_short_id_falls_back_to_marker(paper_id_lookup):
    """Unmapped markers keep their raw (short) form — same as before."""
    assert derive_paper_html_url("T_sha256_deadbeef12345.md").endswith(
        "sha256_deadbeef12345.html"
    )


def test_normalize_repairs_stored_short_url(paper_id_lookup):
    assert normalize_paper_html_url(PAPER_URL) == FULL_URL


def test_normalize_passes_through_full_or_foreign_urls(paper_id_lookup):
    assert normalize_paper_html_url(FULL_URL) == FULL_URL
    assert normalize_paper_html_url("https://example.com/x.html") == (
        "https://example.com/x.html"
    )
    assert normalize_paper_html_url("") == ""


def test_normalize_without_lookup_is_identity():
    assert normalize_paper_html_url(PAPER_URL) == PAPER_URL


def test_convert_entities_derive_html_url_from_file_path():
    """Entity with empty stored html_url falls back to file_path derivation."""
    result = convert_to_user_format(
        entities_context=[
            {
                "entity": "Glucokinase",
                "type": "PROTEIN",
                "description": "desc",
                "source_id": "doc-x-chunk-000",
                "file_path": PAPER_FILE,
            }
        ],
        relations_context=[],
        chunks=[],
        references=[],
        query_mode="mix",
    )
    assert result["data"]["entities"][0]["html_url"] == PAPER_URL


def test_convert_relationships_derive_html_url_from_file_path():
    result = convert_to_user_format(
        entities_context=[],
        relations_context=[
            {
                "entity1": "A",
                "entity2": "B",
                "description": "desc",
                "file_path": PAPER_FILE,
            }
        ],
        chunks=[],
        references=[],
        query_mode="mix",
    )
    assert result["data"]["relationships"][0]["html_url"] == PAPER_URL


def test_convert_chunks_derive_html_url_from_file_path():
    result = convert_to_user_format(
        entities_context=[],
        relations_context=[],
        chunks=[{"content": "text", "file_path": PAPER_FILE, "chunk_id": "c1"}],
        references=[],
        query_mode="mix",
    )
    assert result["data"]["chunks"][0]["html_url"] == PAPER_URL


def test_stored_html_url_wins_over_derivation():
    """A stored html_url (post-fix extraction) is never overwritten."""
    stored = "https://paper-lib.huamedicine.com/papers/sha256_stored999.html"
    result = convert_to_user_format(
        entities_context=[
            {"entity": "E", "file_path": PAPER_FILE, "html_url": stored}
        ],
        relations_context=[],
        chunks=[],
        references=[],
        query_mode="mix",
    )
    assert result["data"]["entities"][0]["html_url"] == stored


def test_convert_chunks_normalize_stored_short_url(paper_id_lookup):
    """Chunks ingested with short-form html_url get repaired at query time."""
    result = convert_to_user_format(
        entities_context=[],
        relations_context=[],
        chunks=[{"content": "text", "file_path": PAPER_FILE, "html_url": PAPER_URL}],
        references=[],
        query_mode="mix",
    )
    assert result["data"]["chunks"][0]["html_url"] == FULL_URL


def test_non_paper_file_path_yields_empty_html_url():
    result = convert_to_user_format(
        entities_context=[{"entity": "E", "file_path": "notes/plain.md"}],
        relations_context=[],
        chunks=[],
        references=[],
        query_mode="mix",
    )
    assert result["data"]["entities"][0]["html_url"] == ""
