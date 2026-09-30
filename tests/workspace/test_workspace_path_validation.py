"""Unit tests for ``lightrag.utils.validate_workspace``.

File-based storages build a per-workspace subdirectory under ``working_dir``
via ``os.path.join(working_dir, workspace)``. ``validate_workspace`` guards that
join against path traversal by rejecting any name that is not a single path
component, while leaving legitimate names (including dotted ones) untouched.
"""

from pathlib import Path

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


class TestStorageRejectsTraversal:
    """The check is wired into storage construction, not just the helper."""

    def test_json_kv_storage_rejects_traversal(self, tmp_path):
        from lightrag.kg.json_kv_impl import JsonKVStorage

        cfg = {"working_dir": str(tmp_path)}
        with pytest.raises(ValueError):
            JsonKVStorage(
                namespace="ns",
                workspace="../../../etc",
                global_config=cfg,
                embedding_func=None,
            )

    def test_json_kv_storage_accepts_dotted_name(self, tmp_path):
        from lightrag.kg.json_kv_impl import JsonKVStorage

        cfg = {"working_dir": str(tmp_path)}
        storage = JsonKVStorage(
            namespace="ns",
            workspace="v1.0",
            global_config=cfg,
            embedding_func=None,
        )
        assert storage.workspace == "v1.0"


def _import_document_manager():
    """Import DocumentManager with a clean argv.

    Importing ``lightrag.api`` modules triggers the server's argparse against
    ``sys.argv`` at import time, which would otherwise see pytest's arguments
    (same workaround as tests/api/test_path_prefixes.py).
    """
    import sys

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        from lightrag.api.routers.document_routes import DocumentManager

        return DocumentManager
    finally:
        sys.argv = original_argv


class TestUploadPath:
    """DocumentManager builds the upload dir from the workspace; the same
    validation must guard it, and the API server's own sanitizer
    (``re.sub(r"[^a-zA-Z0-9_]", "_", ...)`` in api/config.py) must never
    produce a value our validator rejects."""

    def test_document_manager_rejects_traversal(self, tmp_path):
        DocumentManager = _import_document_manager()

        with pytest.raises(ValueError):
            DocumentManager(str(tmp_path), workspace="../../outside")
        assert not (tmp_path.parent.parent / "outside").exists()

    def test_document_manager_creates_workspace_subdir(self, tmp_path):
        DocumentManager = _import_document_manager()

        dm = DocumentManager(str(tmp_path), workspace="space1")
        assert dm.input_dir == tmp_path / "space1"
        assert dm.input_dir.is_dir()

    def test_document_manager_empty_workspace_uses_base_dir(self, tmp_path):
        DocumentManager = _import_document_manager()

        dm = DocumentManager(str(tmp_path), workspace="")
        assert dm.input_dir == tmp_path

    def test_api_sanitizer_output_always_accepted(self):
        import re

        for raw in [
            "my workspace",
            "v1.0",
            "../../../etc",
            "a/b",
            "..\\win",
            "工作区",
            "..",
            ".",
        ]:
            sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", raw)
            assert validate_workspace(sanitized) == sanitized

    def test_upload_dir_matches_storage_dir(self, tmp_path):
        """Files uploaded to <input_dir>/<ws>/ must be resolvable against
        storage data in <working_dir>/<ws>/ — same subdirectory name."""
        DocumentManager = _import_document_manager()
        from lightrag.kg.json_kv_impl import JsonKVStorage

        inputs = tmp_path / "inputs"
        working = tmp_path / "rag_storage"
        inputs.mkdir()
        working.mkdir()

        dm = DocumentManager(str(inputs), workspace="space1")
        kv = JsonKVStorage(
            namespace="ns",
            workspace="space1",
            global_config={"working_dir": str(working)},
            embedding_func=None,
        )
        upload_subdir = dm.input_dir.relative_to(inputs)
        storage_subdir = Path(kv._file_name).parent.relative_to(working)
        assert upload_subdir == storage_subdir


@pytest.mark.parametrize("workspace", ["$", "$meta", "$default", "$custom"])
def test_reserved_prefix_rejected_before_server_normalization(workspace):
    from lightrag.utils import normalize_server_workspace, validate_workspace_override

    with pytest.raises(ValueError, match="reserved"):
        validate_workspace(workspace)
    with pytest.raises(ValueError, match="reserved"):
        normalize_server_workspace(workspace)
    with pytest.raises(ValueError, match="reserved"):
        validate_workspace_override("POSTGRES_WORKSPACE", workspace)


@pytest.mark.parametrize("workspace", ["", "meta", "default", "_lightrag_server"])
def test_ordinary_names_and_empty_workspace_remain_valid(workspace):
    from lightrag.utils import normalize_server_workspace

    assert validate_workspace(workspace) == workspace
    assert normalize_server_workspace(workspace) == workspace


@pytest.mark.parametrize("workspace", ["$meta", "$default"])
def test_sdk_rejects_reserved_workspace_before_creating_storage(tmp_path, workspace):
    from lightrag import LightRAG

    working_dir = tmp_path / "not-created"
    with pytest.raises(ValueError, match="reserved"):
        LightRAG(working_dir=str(working_dir), workspace=workspace)
    assert not working_dir.exists()


@pytest.mark.parametrize("workspace", ["$meta", "$default"])
def test_server_rejects_reserved_workspace_from_environment(monkeypatch, workspace):
    import sys
    from lightrag.api.config import parse_args

    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("WORKSPACE", workspace)
    with pytest.raises(ValueError, match="reserved"):
        parse_args()
