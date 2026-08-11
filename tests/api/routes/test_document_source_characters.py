"""A document source must not carry a character the graph cannot store.

``file_path`` is the one extracted attribute that reaches graph storage with no
character cleaning at all: the extraction handlers pass it through untouched,
while ``description`` / ``entity_type`` / ``keywords`` all go through
``sanitize_text_for_encoding``. And it is stamped onto *every* entity and
relation extracted from the document.

That made the two ingress paths disagree. ``/documents/upload`` refused control
characters in a filename; ``/documents/text`` accepted the same bytes in
``file_source``, which then reached ``upsert_node`` and -- on the default
NetworkX backend -- poisoned the in-memory graph, stopping all persistence for
the life of the process. Same defect class as GHSA-c922-pw4m-4wcv, through an
ingress the manual-edit field allowlist does not cover.

Both paths now share one character rule. The rejection lives at the boundary,
never in ``normalize_file_path``: that helper also runs on *read* paths (it
normalizes ``doc_status.file_path`` when building document listings), so a
document already holding a bad source has to stay listable and deletable rather
than 500 the listing.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from pydantic import ValidationError

# Importing this module resolves the server config, which parses sys.argv and
# would choke on pytest's own flags. Same guard the other document_routes tests
# use.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

InsertTextRequest = _document_routes.InsertTextRequest
InsertTextsRequest = _document_routes.InsertTextsRequest
find_unsafe_document_source_character = (
    _document_routes.find_unsafe_document_source_character
)
normalize_file_path = _document_routes.normalize_file_path

pytestmark = pytest.mark.offline

UNSAFE = [
    pytest.param("a\x00b.txt", "U+0000", id="nul"),
    pytest.param("a\x0bb.txt", "U+000B", id="vertical-tab"),
    pytest.param("a\x1fb.txt", "U+001F", id="unit-separator"),
    pytest.param("a\tb.txt", "U+0009", id="tab"),
    pytest.param("a\nb.txt", "U+000A", id="newline"),
    pytest.param("a\x7fb.txt", "U+007F", id="del"),
    # Reaches a filename through surrogateescape decoding of a non-UTF-8
    # multipart header, and is exactly as unserializable as a control character.
    pytest.param("a\ud800b.txt", "U+D800", id="lone-surrogate"),
    pytest.param("a￾b.txt", "U+FFFE", id="non-character"),
]

SAFE = [
    "report.pdf",
    "报告 2026.docx",
    "a-b_c.d.txt",
    "emoji \U0001f600.md",
    "spaces are fine.txt",
]


class TestCharacterRule:
    @pytest.mark.parametrize("value, code", UNSAFE)
    def test_unsafe_characters_are_found(self, value, code):
        offender = find_unsafe_document_source_character(value)
        assert offender is not None
        assert f"U+{ord(offender):04X}" == code

    @pytest.mark.parametrize("value", SAFE)
    def test_ordinary_sources_are_accepted(self, value):
        assert find_unsafe_document_source_character(value) is None


class TestTextEndpointRequests:
    """The one place /documents/text and /documents/texts both pass through."""

    @pytest.mark.parametrize("value, code", UNSAFE)
    def test_text_request_rejects_unsafe_file_source(self, value, code):
        with pytest.raises(ValidationError) as excinfo:
            InsertTextRequest(text="hello", file_source=value)

        assert code in str(excinfo.value)

    @pytest.mark.parametrize("value, code", UNSAFE)
    def test_texts_request_rejects_unsafe_file_source(self, value, code):
        with pytest.raises(ValidationError) as excinfo:
            InsertTextsRequest(texts=["hello"], file_sources=[value])

        assert code in str(excinfo.value)

    def test_texts_request_rejects_an_offender_in_any_position(self):
        with pytest.raises(ValidationError, match=r"U\+000B"):
            InsertTextsRequest(
                texts=["a", "b", "c"],
                file_sources=["ok.txt", "bad\x0b.txt", "also-ok.txt"],
            )

    @pytest.mark.parametrize("value", SAFE)
    def test_ordinary_file_source_still_accepted(self, value):
        request = InsertTextRequest(text="hello", file_source=value)

        assert request.file_source == normalize_file_path(value)

    def test_absent_file_source_still_accepted(self):
        """An omitted field keeps its default and is not rejected.

        Pydantic skips a ``mode="before"`` validator for a field that was never
        supplied, so the default ``None`` reaches the endpoint unchanged -- which
        is where ``is_valid_file_source`` turns it into the existing 400.
        """
        request = InsertTextRequest(text="hello")

        assert request.file_source is None

    def test_explicit_null_file_source_still_accepted(self):
        """An explicit null does run the validator, and must survive it."""
        request = InsertTextRequest(text="hello", file_source=None)

        assert request.file_source == _document_routes.UNKNOWN_FILE_SOURCE

    def test_absent_file_sources_still_accepted(self):
        request = InsertTextsRequest(texts=["hello"])

        assert request.file_sources is None


class TestUploadPath:
    """``sanitize_filename`` now shares the rule; its 400 contract is unchanged."""

    @pytest.mark.parametrize("value, _code", UNSAFE)
    def test_unsafe_filename_is_rejected(self, tmp_path, value, _code):
        with pytest.raises(_document_routes.HTTPException) as excinfo:
            _document_routes.sanitize_filename(value, tmp_path)

        assert excinfo.value.status_code == 400

    @pytest.mark.parametrize("value", SAFE)
    def test_ordinary_filename_is_accepted(self, tmp_path, value):
        assert _document_routes.sanitize_filename(value, tmp_path) == value


class TestReadPathsStayNonRaising:
    @pytest.mark.parametrize("value, _code", UNSAFE)
    def test_normalize_file_path_never_raises(self, value, _code):
        """A document already holding a bad source must stay listable.

        ``normalize_file_path`` runs on the response-building path for
        ``doc_status.file_path``, so making *it* raise would turn one poisoned
        legacy row into a 500 for the whole document listing -- and leave no way
        to delete the row.
        """
        assert isinstance(normalize_file_path(value), str)
