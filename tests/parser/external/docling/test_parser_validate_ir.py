"""``DoclingParser.validate_ir``: the zero-block failure and its VLM hint.

The VLM hint must appear only under ``DOCLING_PIPELINE=vlm`` — a standard
pipeline user with an empty conversion has no preset to check — and it must not
point at ``DOCLING_SERVE_ALLOWED_VLM_PRESETS``: a disallowed preset fails the
serve task before any result exists, so it never produces an empty IR.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lightrag.parser.external.docling.parser import DoclingParser
from lightrag.sidecar.ir import IRDoc

pytestmark = pytest.mark.offline


def _empty_ir() -> IRDoc:
    return IRDoc(
        document_name="scan.pdf",
        document_format="pdf",
        doc_title="",
        split_option={},
        blocks=[],
    )


def _zero_block_message(tmp_path: Path) -> str:
    with pytest.raises(ValueError, match="zero blocks") as excinfo:
        DoclingParser().validate_ir(_empty_ir(), file_path="scan.pdf", raw_dir=tmp_path)
    return str(excinfo.value)


def test_zero_blocks_standard_pipeline_has_no_vlm_hint(monkeypatch, tmp_path):
    monkeypatch.delenv("DOCLING_PIPELINE", raising=False)
    message = _zero_block_message(tmp_path)
    assert "VLM" not in message
    assert "DOCLING_SERVE_ALLOWED_VLM_PRESETS" not in message


def test_zero_blocks_vlm_pipeline_names_the_model(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCLING_PIPELINE", "vlm")
    monkeypatch.setenv("DOCLING_VLM_PRESET", "minicpm_ocr")
    message = _zero_block_message(tmp_path)
    assert "DOCLING_PIPELINE=vlm" in message
    assert "DOCLING_SERVE_ALLOWED_VLM_PRESETS" not in message


def test_zero_blocks_invalid_pipeline_still_reports_zero_blocks(monkeypatch, tmp_path):
    # The pipeline value is validated elsewhere; here it must not mask the
    # zero-block error with a different exception.
    monkeypatch.setenv("DOCLING_PIPELINE", "fast")
    message = _zero_block_message(tmp_path)
    assert "VLM" not in message
