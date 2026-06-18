from pathlib import Path

import pytest

from lightrag.kb_iteration.patches import (
    PatchCandidate,
    validate_patch_candidate,
    write_patch_candidate,
)


def test_write_patch_candidate_writes_patch_candidates_file(tmp_path: Path) -> None:
    diff_text = """diff --git a/lightrag/medical_kg/hierarchy.py b/lightrag/medical_kg/hierarchy.py
--- a/lightrag/medical_kg/hierarchy.py
+++ b/lightrag/medical_kg/hierarchy.py
@@ -1 +1 @@
-old
+candidate change
"""
    candidate = PatchCandidate(
        proposal_id="proposal-20260618-001",
        target_path="lightrag/medical_kg/hierarchy.py",
        diff_text=diff_text,
    )

    written_path = write_patch_candidate(candidate, tmp_path)

    assert written_path == tmp_path / "patch_candidates" / "proposal-20260618-001.patch"
    assert written_path.read_text(encoding="utf-8") == diff_text


def test_write_patch_candidate_accepts_string_output_dir(tmp_path: Path) -> None:
    candidate = PatchCandidate(
        proposal_id="proposal-20260618-002",
        target_path="docs/demo.md",
        diff_text="diff --git a/docs/demo.md b/docs/demo.md\n+candidate change\n",
    )

    written_path = write_patch_candidate(candidate, str(tmp_path))

    assert written_path == tmp_path / "patch_candidates" / "proposal-20260618-002.patch"


@pytest.mark.parametrize(
    "target_path",
    [
        ".env",
        "",
        "   ",
        "data/rag_storage/demo/graph_chunk_entity_relation.graphml",
        "work/kb-iteration/demo/approval_queue.md",
        "uv.lock",
        "../outside.py",
        "C:/Users/secret/file.py",
    ],
)
def test_validate_patch_candidate_rejects_unsafe_targets(target_path: str) -> None:
    candidate = PatchCandidate(
        proposal_id="safe-proposal",
        target_path=target_path,
        diff_text="diff --git a/file b/file\n",
    )

    with pytest.raises(ValueError):
        validate_patch_candidate(candidate)


def test_validate_patch_candidate_normalizes_target_whitespace_and_backslashes() -> None:
    candidate = PatchCandidate(
        proposal_id="safe-proposal",
        target_path=" lightrag\\medical_kg\\hierarchy.py ",
        diff_text="diff --git a/file b/file\n",
    )

    validate_patch_candidate(candidate)


def test_validate_patch_candidate_rejects_whitespace_only_diff() -> None:
    candidate = PatchCandidate(
        proposal_id="safe-proposal",
        target_path="docs/demo.md",
        diff_text=" \n\t ",
    )

    with pytest.raises(ValueError):
        validate_patch_candidate(candidate)


def test_validate_patch_candidate_allows_whitelisted_webui_target() -> None:
    candidate = PatchCandidate(
        proposal_id="safe-proposal",
        target_path="lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.tsx",
        diff_text="diff --git a/file b/file\n",
    )

    validate_patch_candidate(candidate)
