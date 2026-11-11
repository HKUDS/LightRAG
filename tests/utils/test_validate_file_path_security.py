"""Tests for ``validate_file_path_security`` path-traversal containment.

The helper is the containment boundary for two callers that turn
externally-influenced strings into filesystem paths — the ``/documents`` upload
and scan routes (``lightrag/api/routers/document_routes.py``) and sidecar
artifact resolution (``lightrag/pipeline.py``, which delegates containment to it
outright). Both feed it names that ultimately came from a client, so a hole here
is a read outside the input directory. It had no direct tests.

Assertions compare ``Path`` objects and use ``relative_to`` rather than string
matching, because separators and resolution differ between POSIX and Windows.
Inputs whose meaning is platform-specific — trailing-dot names such as ``"..."``,
which Windows strips and POSIX keeps as a literal directory name — are
deliberately not asserted on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lightrag.utils import validate_file_path_security

pytestmark = pytest.mark.offline


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    return tmp_path


class TestAcceptedPaths:
    def test_a_plain_filename_resolves_inside_the_base(self, base_dir):
        result = validate_file_path_security("report.pdf", base_dir)

        assert result is not None
        assert result.relative_to(base_dir.resolve()) == Path("report.pdf")

    def test_a_nested_relative_path_is_accepted(self, base_dir):
        result = validate_file_path_security("sub/dir/report.pdf", base_dir)

        assert result is not None
        assert result.relative_to(base_dir.resolve()) == Path("sub/dir/report.pdf")

    def test_a_leading_dot_segment_is_normalized_away(self, base_dir):
        result = validate_file_path_security("./report.pdf", base_dir)

        assert result is not None
        assert result.relative_to(base_dir.resolve()) == Path("report.pdf")

    def test_an_interior_dot_segment_is_normalized_away(self, base_dir):
        result = validate_file_path_security("a/./b.pdf", base_dir)

        assert result is not None
        assert result.relative_to(base_dir.resolve()) == Path("a/b.pdf")

    def test_backslashes_are_normalized_into_separators(self, base_dir):
        # Windows-style input reaching a POSIX server must still land in the
        # same place, so the helper rewrites separators before resolving.
        result = validate_file_path_security(r"sub\dir\report.pdf", base_dir)

        assert result is not None
        assert result.relative_to(base_dir.resolve()) == Path("sub/dir/report.pdf")

    def test_surrounding_whitespace_is_stripped(self, base_dir):
        result = validate_file_path_security("  report.pdf  ", base_dir)

        assert result is not None
        assert result.relative_to(base_dir.resolve()) == Path("report.pdf")

    def test_traversal_that_stays_inside_the_base_is_allowed(self, base_dir):
        """``a/../b.pdf`` resolves to ``b.pdf``, which is still contained.

        Containment is decided on the *resolved* path, not on the presence of a
        ``..`` segment, so a path that walks up and back down without leaving
        the base is legitimate. Pinned because tightening this into a blanket
        ``".." in path`` rejection would look like hardening while actually
        breaking ordinary inputs.
        """
        result = validate_file_path_security("a/../b.pdf", base_dir)

        assert result is not None
        assert result.relative_to(base_dir.resolve()) == Path("b.pdf")

    def test_existence_is_not_checked(self, base_dir):
        """A contained but absent path is returned, per the documented contract.

        The caller decides what "inside but absent" means; conflating it with
        "unsafe" here would make the two indistinguishable to callers.
        """
        result = validate_file_path_security("nope/missing.pdf", base_dir)

        assert result is not None
        assert not result.exists()


class TestRejectedPaths:
    @pytest.mark.parametrize(
        "escaping_path",
        [
            "../etc/passwd",
            "../../etc/passwd",
            "a/../../escape.pdf",
            "sub/../../escape.pdf",
        ],
    )
    def test_paths_escaping_the_base_are_rejected(self, base_dir, escaping_path):
        assert validate_file_path_security(escaping_path, base_dir) is None

    @pytest.mark.parametrize(
        "escaping_path",
        [
            r"..\windows\system32",
            r"..\..\secret.txt",
            r"sub\..\..\escape.pdf",
        ],
    )
    def test_windows_style_traversal_is_rejected(self, base_dir, escaping_path):
        assert validate_file_path_security(escaping_path, base_dir) is None

    def test_a_bare_parent_reference_is_rejected(self, base_dir):
        assert validate_file_path_security("..", base_dir) is None

    def test_an_absolute_path_is_rejected(self, base_dir):
        # An absolute operand makes `base_dir / value` discard the base
        # entirely, so containment is what catches this rather than syntax.
        assert validate_file_path_security("/etc/passwd", base_dir) is None

    @pytest.mark.parametrize("blank", ["", "   ", "\t", "\n"])
    def test_blank_input_is_rejected(self, base_dir, blank):
        assert validate_file_path_security(blank, base_dir) is None

    def test_none_is_rejected(self, base_dir):
        assert validate_file_path_security(None, base_dir) is None


class TestContainmentIsDecidedOnTheResolvedPath:
    def test_a_sibling_directory_sharing_a_name_prefix_is_rejected(self, tmp_path):
        """``/base_evil`` must not pass containment against base ``/base``.

        A prefix comparison on the path *string* would accept it; the helper
        uses ``Path.is_relative_to``, which compares whole components.
        """
        base = tmp_path / "base"
        base.mkdir()
        (tmp_path / "base_evil").mkdir()

        assert validate_file_path_security("../base_evil/loot.txt", base) is None

    def test_a_path_landing_exactly_on_the_base_is_contained(self, base_dir):
        # `.` resolves to the base itself, which is relative to the base.
        result = validate_file_path_security(".", base_dir)

        assert result == base_dir.resolve()
