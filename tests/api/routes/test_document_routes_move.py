"""Tests for the `/documents/move` folder-relocation helpers and request model.

Moving a document only rewrites its stored ``file_path``; ids, chunks, vectors
and graph data are untouched. The correctness of a move therefore reduces to
three pure functions plus the request model's validation:

1. **``_encode_folder_prefix``**: a '/' separated folder path becomes the
   ``__`` separated ``file_path`` prefix the webui expects. Must agree with
   the webui's ``encodeFolderPrefix`` — a mismatch silently files documents
   into a folder the UI cannot see.

2. **``_doc_basename``**: recovers the on-disk basename from a stored
   ``file_path``, so a move re-prefixes the name instead of nesting the old
   prefix inside the new one.

3. **``_conflict_free_basename``**: resolves collisions in the target folder,
   either by suffixing (``rename_on_conflict=True``) or by signalling a skip.

4. **``MoveDocRequest``**: rejects empty/blank/duplicate ids up front, so a
   malformed batch never reaches the background task.
"""

import importlib
import sys

import pytest
from pydantic import ValidationError

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_dr = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

_encode_folder_prefix = _dr._encode_folder_prefix
_doc_basename = _dr._doc_basename
_conflict_free_basename = _dr._conflict_free_basename
_MOVE_FOLDER_SEPARATOR = _dr._MOVE_FOLDER_SEPARATOR
MoveDocRequest = _dr.MoveDocRequest

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# 1. _encode_folder_prefix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("folder", ["", "   ", "/", "///"])
def test_root_folder_encodes_to_empty_prefix(folder):
    """Root (however it is spelled) carries no prefix at all.

    A stray separator here would push every root-level document into a
    folder literally named "".
    """
    assert _encode_folder_prefix(folder) == ""


def test_single_segment_keeps_trailing_separator():
    """The prefix is concatenated directly with the basename, so it must
    already end with the separator."""
    assert _encode_folder_prefix("A") == "A__"


@pytest.mark.parametrize(
    "folder",
    ["A/B", "/A/B", "A/B/", "/A/B/", "A//B", "  A/B  "],
)
def test_nested_paths_normalize_to_same_prefix(folder):
    """Leading/trailing/duplicated slashes and surrounding whitespace are all
    normalized away, so equivalent spellings cannot create sibling folders."""
    assert _encode_folder_prefix(folder) == "A__B__"


def test_deeply_nested_path_joins_every_segment():
    """Regression guard: the join must produce a separator *between* segments.

    Getting the join backwards (a JS-style ``list.join(sep)``) raises
    ``AttributeError`` and breaks every non-root move.
    """
    assert _encode_folder_prefix("A/B/C/D") == "A__B__C__D__"


def test_encoded_prefix_uses_the_shared_separator():
    """The separator is a contract shared with the webui, not a local detail."""
    assert (
        _encode_folder_prefix("A/B")
        == f"A{_MOVE_FOLDER_SEPARATOR}B{_MOVE_FOLDER_SEPARATOR}"
    )


# ---------------------------------------------------------------------------
# 2. _doc_basename
# ---------------------------------------------------------------------------


def test_unprefixed_path_is_its_own_basename():
    assert _doc_basename("report.pdf") == "report.pdf"


@pytest.mark.parametrize(
    "stored,expected",
    [
        ("A__report.pdf", "report.pdf"),
        ("A__B__report.pdf", "report.pdf"),
        ("A__B__C__report.pdf", "report.pdf"),
    ],
)
def test_folder_prefix_is_stripped_at_any_depth(stored, expected):
    """Only the trailing segment survives, so re-prefixing on move cannot
    accumulate the previous folder path."""
    assert _doc_basename(stored) == expected


def test_empty_path_yields_empty_basename():
    """The caller treats "" as unmovable and reports it as a failure rather
    than fabricating a name."""
    assert _doc_basename("") == ""


def test_none_path_is_tolerated():
    """``file_path`` is read from stored doc_status and may be missing."""
    assert _doc_basename(None) == ""


def test_basename_keeps_internal_spaces_and_dots():
    assert (
        _doc_basename("A__B__2024 annual.report.v2.pdf") == "2024 annual.report.v2.pdf"
    )


# ---------------------------------------------------------------------------
# 3. _conflict_free_basename
# ---------------------------------------------------------------------------


def test_free_name_is_returned_unchanged():
    assert _conflict_free_basename("r.pdf", set(), True) == "r.pdf"


def test_first_collision_gets_suffix_one():
    assert _conflict_free_basename("r.pdf", {"r.pdf"}, True) == "r (1).pdf"


def test_suffix_increments_until_free():
    """Consecutive moves of the same name must not overwrite each other."""
    occupied = {"r.pdf", "r (1).pdf", "r (2).pdf"}
    assert _conflict_free_basename("r.pdf", occupied, True) == "r (3).pdf"


def test_suffix_is_inserted_before_the_extension():
    """Appending after the extension would break type detection downstream."""
    result = _conflict_free_basename("annual report.docx", {"annual report.docx"}, True)
    assert result == "annual report (1).docx"


def test_extensionless_name_gets_a_bare_suffix():
    assert _conflict_free_basename("README", {"README"}, True) == "README (1)"


def test_multi_dot_name_only_splits_the_last_extension():
    result = _conflict_free_basename("archive.tar.gz", {"archive.tar.gz"}, True)
    assert result == "archive.tar (1).gz"


def test_dotfile_is_not_treated_as_pure_extension():
    """``.env``'s rpartition yields an empty stem; the fallback must keep the
    name usable rather than emitting a bare " (1)"."""
    result = _conflict_free_basename(".env", {".env"}, True)
    assert result == ".env (1)"


def test_collision_returns_none_when_renaming_is_disabled():
    """None is the caller's signal to skip the document, not to overwrite."""
    assert _conflict_free_basename("r.pdf", {"r.pdf"}, False) is None


def test_no_collision_still_succeeds_when_renaming_is_disabled():
    """``rename_on_conflict=False`` only affects actual collisions."""
    assert _conflict_free_basename("r.pdf", {"other.pdf"}, False) == "r.pdf"


# ---------------------------------------------------------------------------
# 4. MoveDocRequest validation
# ---------------------------------------------------------------------------


def test_valid_request_defaults_to_root_and_renaming():
    """Defaults matter: an omitted target means "move to root", and renaming
    is on so a batch move cannot silently drop documents."""
    req = MoveDocRequest(doc_ids=["doc-1"])
    assert req.doc_ids == ["doc-1"]
    assert req.target_folder == ""
    assert req.rename_on_conflict is True


def test_doc_ids_are_stripped():
    assert MoveDocRequest(doc_ids=["  doc-1  "]).doc_ids == ["doc-1"]


def test_empty_doc_ids_list_is_rejected():
    with pytest.raises(ValidationError):
        MoveDocRequest(doc_ids=[])


@pytest.mark.parametrize("bad", ["", "   "])
def test_blank_doc_id_is_rejected(bad):
    with pytest.raises(ValidationError):
        MoveDocRequest(doc_ids=["doc-1", bad])


def test_duplicate_doc_ids_are_rejected():
    """A duplicate would be moved twice and collide with its own new name."""
    with pytest.raises(ValidationError):
        MoveDocRequest(doc_ids=["doc-1", "doc-1"])


def test_duplicates_are_detected_after_stripping():
    """Whitespace must not be a way to smuggle the same id in twice."""
    with pytest.raises(ValidationError):
        MoveDocRequest(doc_ids=["doc-1", "  doc-1  "])


def test_target_folder_and_flag_are_preserved():
    req = MoveDocRequest(
        doc_ids=["doc-1"], target_folder="A/B", rename_on_conflict=False
    )
    assert req.target_folder == "A/B"
    assert req.rename_on_conflict is False
