"""Unit tests for ``lightrag.utils.validate_workspace``.

File-based storages build a per-workspace subdirectory under ``working_dir``
via ``os.path.join(working_dir, workspace)``. ``validate_workspace`` guards that
join against path traversal by rejecting any name that is not a single path
component, while leaving legitimate names (including dotted ones) untouched.
"""

import pytest

from lightrag.utils import validate_workspace

pytestmark = pytest.mark.offline


class TestValidWorkspaces:
    """Names that are valid single path components pass through unchanged."""

    @pytest.mark.parametrize(
        "workspace",
        [
            "my_workspace",
            "workspace-123",
            "MyWorkSpace",
            "12345",
            "_",
            "v1.0",  # dots are safe in a single path component
            "team.alpha",
            "a..b",  # ".." embedded in a name is not a traversal
            "工作区_test",  # non-ASCII is fine for a directory name
        ],
    )
    def test_returns_input_unchanged(self, workspace):
        assert validate_workspace(workspace) == workspace


class TestPathTraversalRejected:
    """Names that could escape ``working_dir`` raise ``ValueError``."""

    @pytest.mark.parametrize(
        "workspace",
        [
            "..",
            ".",
            "../etc",
            "../../../etc/passwd",
            "foo/../../etc",
            "foo/bar",  # nested path, not a single component
            "/etc/passwd",  # absolute path would discard working_dir on join
            "..\\..\\windows",  # Windows-style separator
            "foo\\bar",
        ],
    )
    def test_raises_value_error(self, workspace):
        with pytest.raises(ValueError):
            validate_workspace(workspace)
