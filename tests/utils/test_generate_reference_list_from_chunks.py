"""Regression tests for ``generate_reference_list_from_chunks``.

This function assigns each chunk's ``reference_id`` from its ``file_path``'s
occurrence frequency across the surviving chunk set, so the numbering callers
show the user (and that ``render_chunks_context_text`` sends to the LLM)
shifts whenever truncation changes which chunks survive. The ordering rule --
sort by count descending, then by first-appearance index ascending -- and the
``unknown_source`` / missing ``file_path`` sentinel handling are both load
bearing and were previously only exercised incidentally, through a bigger
truncation-pipeline test that never varied them.
"""

from lightrag.utils import generate_reference_list_from_chunks

import pytest


pytestmark = pytest.mark.offline


def test_empty_chunks_returns_empty_lists():
    assert generate_reference_list_from_chunks([]) == ([], [])


def test_single_chunk_gets_reference_id_one():
    chunks = [{"content": "hello", "file_path": "a.txt"}]

    reference_list, updated = generate_reference_list_from_chunks(chunks)

    assert reference_list == [{"reference_id": "1", "file_path": "a.txt"}]
    assert updated == [{"content": "hello", "file_path": "a.txt", "reference_id": "1"}]


def test_higher_frequency_file_path_gets_lower_reference_id():
    # "common.txt" appears twice, "rare.txt" once -- despite rare.txt
    # appearing first, the more frequent path must be numbered "1".
    chunks = [
        {"content": "c1", "file_path": "rare.txt"},
        {"content": "c2", "file_path": "common.txt"},
        {"content": "c3", "file_path": "common.txt"},
    ]

    reference_list, updated = generate_reference_list_from_chunks(chunks)

    assert reference_list == [
        {"reference_id": "1", "file_path": "common.txt"},
        {"reference_id": "2", "file_path": "rare.txt"},
    ]
    assert [c["reference_id"] for c in updated] == ["2", "1", "1"]


def test_tied_frequency_breaks_by_first_appearance_order():
    # Both file paths occur exactly once each, so the tie must break by which
    # one appeared first in the chunk list, not by any other ordering (e.g.
    # alphabetical, which would put "a.txt" first here too, so the content
    # order is reversed to make sure first-appearance -- not alphabetical --
    # is what decides it).
    chunks = [
        {"content": "c1", "file_path": "z.txt"},
        {"content": "c2", "file_path": "a.txt"},
    ]

    reference_list, _ = generate_reference_list_from_chunks(chunks)

    assert [entry["file_path"] for entry in reference_list] == ["z.txt", "a.txt"]


def test_repeated_file_path_keeps_first_appearance_index_for_the_tie_break():
    # Two file paths both appear twice (a genuine tie on count); the first
    # one to appear in the chunk list must win the tie, confirming the first
    # occurrence's index -- not the count alone -- settles equal-frequency
    # ordering.
    chunks = [
        {"content": "c1", "file_path": "second.txt"},
        {"content": "c2", "file_path": "first.txt"},
        {"content": "c3", "file_path": "first.txt"},
        {"content": "c4", "file_path": "second.txt"},
    ]

    reference_list, _ = generate_reference_list_from_chunks(chunks)

    assert [entry["file_path"] for entry in reference_list] == [
        "second.txt",
        "first.txt",
    ]


def test_unknown_source_sentinel_is_excluded_from_numbering():
    chunks = [
        {"content": "c1", "file_path": "unknown_source"},
        {"content": "c2", "file_path": "a.txt"},
    ]

    reference_list, updated = generate_reference_list_from_chunks(chunks)

    assert reference_list == [{"reference_id": "1", "file_path": "a.txt"}]
    assert updated[0]["reference_id"] == ""
    assert updated[1]["reference_id"] == "1"


def test_missing_and_empty_file_path_get_blank_reference_id():
    chunks = [
        {"content": "no key"},
        {"content": "empty", "file_path": ""},
        {"content": "real", "file_path": "a.txt"},
    ]

    reference_list, updated = generate_reference_list_from_chunks(chunks)

    assert reference_list == [{"reference_id": "1", "file_path": "a.txt"}]
    assert [c["reference_id"] for c in updated] == ["", "", "1"]


def test_does_not_mutate_the_input_chunks():
    original = {"content": "c1", "file_path": "a.txt"}
    chunks = [original]

    _, updated = generate_reference_list_from_chunks(chunks)

    assert "reference_id" not in original
    assert updated[0] is not original
    assert updated[0]["reference_id"] == "1"


def test_preserves_other_chunk_fields_in_the_same_order_as_input():
    chunks = [
        {"content": "c1", "file_path": "b.txt", "chunk_id": "x1"},
        {"content": "c2", "file_path": "a.txt", "chunk_id": "x2"},
    ]

    _, updated = generate_reference_list_from_chunks(chunks)

    assert [c["chunk_id"] for c in updated] == ["x1", "x2"]
    assert [c["content"] for c in updated] == ["c1", "c2"]
