"""The truncation summary is operator-facing, not internal bookkeeping.

``DocStatusResponse`` strips ``_INTERNAL_METADATA_KEYS`` from the metadata it
returns, so anything a client needs to SEE has to stay out of that set. The
whole point of recording token-limit truncation on the document (issue #3601)
is that ``/documents`` and the WebUI can show why a knowledge graph came out
short, so this guards the key against being swept in later.
"""

import importlib
import sys

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

from lightrag.base import DocStatus  # noqa: E402
from lightrag.utils import LLM_TRUNCATION_METADATA_KEY  # noqa: E402

pytestmark = pytest.mark.offline

DocStatusResponse = _document_routes.DocStatusResponse


def _response(metadata: dict) -> "DocStatusResponse":
    return DocStatusResponse(
        id="doc-1",
        content_summary="s",
        content_length=1,
        status=DocStatus.PROCESSED,
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
        file_path="doc-1.txt",
        metadata=metadata,
    )


def test_truncation_summary_survives_the_internal_metadata_strip():
    response = _response(
        {
            LLM_TRUNCATION_METADATA_KEY: {
                "events": 2,
                "affected": 1,
                "stages": {"initial": 1, "gleaning": 1},
                "samples": ["chunk-001"],
            },
            # Stripped for contrast: a purge-time bookkeeping list.
            "smartheading_llm_cache_ids": ["default:smartheading:abc"],
        }
    )

    assert response.metadata[LLM_TRUNCATION_METADATA_KEY]["events"] == 2
    assert response.metadata[LLM_TRUNCATION_METADATA_KEY]["samples"] == ["chunk-001"]
    assert "smartheading_llm_cache_ids" not in response.metadata
