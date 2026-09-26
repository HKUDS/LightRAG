"""The anchor file: a fixed path, three fields, a strict read, and two write
modes that never report a failed write as a success.

See *The anchor and the container identity* in
docs/design/ConfigurationStorageContract.md.
"""

from __future__ import annotations

import errno
import json
import os

import pytest

from lightrag import config_anchor as ca
from lightrag.exceptions import ConfigurationIdentityError
from lightrag.namespace import CONFIG_CONTAINER_TAG

pytestmark = pytest.mark.offline

UUID_A = "3f2b8c1e-6a4d-4e2f-9b7a-1c2d3e4f5a6b"
UUID_B = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"


def _write(tmp_path, payload) -> str:
    path = ca.anchor_path(str(tmp_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(payload, str):
            f.write(payload)
        else:
            json.dump(payload, f)
    return path


def _valid(**overrides):
    payload = {
        "schema_version": 1,
        "backend": "PGKVStorage",
        "storage_uuid": UUID_A,
    }
    payload.update(overrides)
    return payload


class TestPath:
    def test_the_anchor_lives_under_the_default_config_dir(self, tmp_path):
        assert ca.anchor_path(str(tmp_path)) == str(
            tmp_path / CONFIG_CONTAINER_TAG / "config_storage_anchor.json"
        )

    def test_the_path_depends_only_on_the_working_dir(self, tmp_path, monkeypatch):
        """``LIGHTRAG_CONFIG_DIR`` moves the JSON data, never the anchor: the
        check and the checked data must not move together."""
        before = ca.anchor_path(str(tmp_path))
        monkeypatch.setenv("LIGHTRAG_CONFIG_DIR", str(tmp_path / "elsewhere"))
        assert ca.anchor_path(str(tmp_path)) == before

    def test_a_relative_working_dir_resolves_to_an_absolute_path(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        assert os.path.isabs(ca.anchor_path("rag_storage"))


class TestStrictRead:
    def test_no_file_is_the_only_absent_answer(self, tmp_path):
        assert ca.read_anchor(str(tmp_path)) is None

    def test_a_valid_anchor_parses(self, tmp_path):
        _write(tmp_path, _valid())
        assert ca.read_anchor(str(tmp_path)) == ca.StorageAnchor(
            backend="PGKVStorage", storage_uuid=UUID_A
        )

    @pytest.mark.parametrize(
        "payload",
        [
            "",  # truncated to nothing
            '{"schema_version": 1, "backend": "PGKVSt',  # truncated mid-write
            "not json at all",
            "[]",
            "null",
            json.dumps(_valid(schema_version=2)),
            json.dumps(_valid(schema_version="1")),
            json.dumps(_valid(schema_version=True)),
            json.dumps(_valid(backend="RedisKVStorage")),
            json.dumps(_valid(backend="NanoVectorDBStorage")),
            json.dumps(_valid(backend=None)),
            json.dumps(_valid(storage_uuid="not-a-uuid")),
            json.dumps(_valid(storage_uuid=UUID_A.upper())),
            json.dumps(_valid(storage_uuid=UUID_A.replace("-", ""))),
            json.dumps(_valid(storage_uuid=7)),
            json.dumps({"schema_version": 1, "backend": "PGKVStorage"}),
            json.dumps(_valid(host="db.internal")),
        ],
    )
    def test_anything_else_refuses_and_is_never_absent(self, tmp_path, payload):
        path = _write(tmp_path, payload)
        with pytest.raises(ConfigurationIdentityError) as excinfo:
            ca.read_anchor(str(tmp_path))
        assert excinfo.value.cause == ca.IDENTITY_ANCHOR_UNREADABLE
        assert excinfo.value.anchor_path == path

    def test_undecodable_bytes_refuse(self, tmp_path):
        path = ca.anchor_path(str(tmp_path))
        os.makedirs(os.path.dirname(path))
        with open(path, "wb") as f:
            f.write(b"\xff\xfe\x00garbage")
        with pytest.raises(ConfigurationIdentityError):
            ca.read_anchor(str(tmp_path))

    def test_a_permission_error_refuses(self, tmp_path, monkeypatch):
        _write(tmp_path, _valid())
        real_open = open

        def _denied(path, *args, **kwargs):
            if str(path).endswith(ca.ANCHOR_FILE_NAME):
                raise PermissionError(errno.EACCES, "Permission denied", path)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", _denied)
        with pytest.raises(ConfigurationIdentityError, match="PermissionError"):
            ca.read_anchor(str(tmp_path))

    def test_a_directory_where_the_file_belongs_refuses(self, tmp_path):
        os.makedirs(ca.anchor_path(str(tmp_path)))
        with pytest.raises(ConfigurationIdentityError):
            ca.read_anchor(str(tmp_path))

    def test_every_admitted_backend_is_accepted(self, tmp_path):
        for backend in (
            "JsonKVStorage",
            "PGKVStorage",
            "MongoKVStorage",
            "OpenSearchKVStorage",
        ):
            _write(tmp_path, _valid(backend=backend))
            assert ca.read_anchor(str(tmp_path)).backend == backend


class TestPublish:
    def _anchor(self, backend="PGKVStorage", storage_uuid=UUID_A):
        return ca.StorageAnchor(backend=backend, storage_uuid=storage_uuid)

    def _leftovers(self, tmp_path):
        return [
            name
            for name in os.listdir(tmp_path / CONFIG_CONTAINER_TAG)
            if name != ca.ANCHOR_FILE_NAME
        ]

    def test_a_bind_writes_exactly_the_three_fields(self, tmp_path):
        path = ca.publish_anchor(str(tmp_path), self._anchor(), replace=False)
        with open(path, encoding="utf-8") as f:
            assert json.load(f) == _valid()
        assert ca.read_anchor(str(tmp_path)) == self._anchor()
        assert self._leftovers(tmp_path) == []

    def test_a_bind_never_overwrites_an_existing_anchor(self, tmp_path):
        ca.publish_anchor(str(tmp_path), self._anchor(), replace=False)
        before = open(ca.anchor_path(str(tmp_path)), "rb").read()
        with pytest.raises(ConfigurationIdentityError) as excinfo:
            ca.publish_anchor(
                str(tmp_path),
                self._anchor(backend="MongoKVStorage", storage_uuid=UUID_B),
                replace=False,
            )
        assert excinfo.value.cause == ca.IDENTITY_ANCHOR_APPEARED
        assert open(ca.anchor_path(str(tmp_path)), "rb").read() == before
        assert self._leftovers(tmp_path) == []

    def test_without_hard_links_the_bind_still_never_clobbers(
        self, tmp_path, monkeypatch
    ):
        def _no_links(src, dst):
            raise OSError(errno.EPERM, "hard links not supported")

        monkeypatch.setattr(ca.os, "link", _no_links)
        ca.publish_anchor(str(tmp_path), self._anchor(), replace=False)
        assert ca.read_anchor(str(tmp_path)) == self._anchor()
        with pytest.raises(ConfigurationIdentityError) as excinfo:
            ca.publish_anchor(
                str(tmp_path), self._anchor(storage_uuid=UUID_B), replace=False
            )
        assert excinfo.value.cause == ca.IDENTITY_ANCHOR_APPEARED
        assert ca.read_anchor(str(tmp_path)).storage_uuid == UUID_A

    def test_without_hard_links_a_winner_that_lands_late_is_never_replaced(
        self, tmp_path, monkeypatch
    ):
        """The fallback's no-clobber may not rest on a check made before the
        publish: another process tree's anchor landing in between must win.
        The winner is planted at the last moment the target can still be
        missing, i.e. just before the final replace."""

        def _no_links(src, dst):
            raise OSError(errno.EPERM, "hard links not supported")

        real_replace = os.replace
        winner = self._anchor(backend="MongoKVStorage", storage_uuid=UUID_B)
        planted = []

        def _replace_after_a_rival(src, dst):
            if not os.path.lexists(dst):
                with open(dst, "w", encoding="utf-8") as f:
                    json.dump(winner.to_payload(), f)
                planted.append(dst)
            real_replace(src, dst)

        monkeypatch.setattr(ca.os, "link", _no_links)
        monkeypatch.setattr(ca.os, "replace", _replace_after_a_rival)
        try:
            ca.publish_anchor(str(tmp_path), self._anchor(), replace=False)
        except ConfigurationIdentityError as e:
            assert e.cause == ca.IDENTITY_ANCHOR_APPEARED
        assert planted == [], "the target was still claimable at the replace"
        assert ca.read_anchor(str(tmp_path)) == self._anchor()
        assert self._leftovers(tmp_path) == []

    def test_the_migration_mode_replaces_atomically(self, tmp_path):
        ca.publish_anchor(str(tmp_path), self._anchor(), replace=False)
        ca.publish_anchor(
            str(tmp_path), self._anchor(backend="MongoKVStorage"), replace=True
        )
        assert ca.read_anchor(str(tmp_path)).backend == "MongoKVStorage"
        assert self._leftovers(tmp_path) == []

    def test_a_failed_temp_write_publishes_nothing(self, tmp_path, monkeypatch):
        def _full(fd):
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(ca.os, "fsync", _full)
        with pytest.raises(ConfigurationIdentityError) as excinfo:
            ca.publish_anchor(str(tmp_path), self._anchor(), replace=False)
        assert excinfo.value.cause == ca.IDENTITY_ANCHOR_WRITE_FAILED
        assert "No anchor was published" in str(excinfo.value)
        assert ca.read_anchor(str(tmp_path)) is None
        assert self._leftovers(tmp_path) == []

    def test_a_failed_publish_publishes_nothing(self, tmp_path, monkeypatch):
        def _eio(src, dst):
            raise OSError(errno.EIO, "I/O error")

        monkeypatch.setattr(ca.os, "link", _eio)
        with pytest.raises(ConfigurationIdentityError) as excinfo:
            ca.publish_anchor(str(tmp_path), self._anchor(), replace=False)
        assert excinfo.value.cause == ca.IDENTITY_ANCHOR_WRITE_FAILED
        assert ca.read_anchor(str(tmp_path)) is None
        assert self._leftovers(tmp_path) == []

    def test_a_failed_directory_fsync_is_loud(self, tmp_path, monkeypatch):
        def _dir_eio(directory):
            raise OSError(errno.EIO, "I/O error")

        monkeypatch.setattr(ca, "_fsync_dir", _dir_eio)
        with pytest.raises(ConfigurationIdentityError) as excinfo:
            ca.publish_anchor(str(tmp_path), self._anchor(), replace=False)
        assert excinfo.value.cause == ca.IDENTITY_ANCHOR_WRITE_FAILED
        assert "may not be durable" in str(excinfo.value)

    def test_a_directory_fsync_the_filesystem_cannot_do_passes(
        self, tmp_path, monkeypatch
    ):
        real_fsync = os.fsync
        calls = {"dir": 0}

        def _fsync(fd):
            import stat

            if stat.S_ISDIR(os.fstat(fd).st_mode):
                calls["dir"] += 1
                raise OSError(errno.EINVAL, "Invalid argument")
            return real_fsync(fd)

        monkeypatch.setattr(ca.os, "fsync", _fsync)
        ca.publish_anchor(str(tmp_path), self._anchor(), replace=False)
        assert calls["dir"] == (0 if os.name == "nt" else 1)
        assert ca.read_anchor(str(tmp_path)) == self._anchor()

    def test_an_unwritable_working_dir_fails_loudly(self, tmp_path, monkeypatch):
        def _read_only(path, exist_ok=False):
            raise OSError(errno.EROFS, "Read-only file system", path)

        monkeypatch.setattr(ca.os, "makedirs", _read_only)
        with pytest.raises(ConfigurationIdentityError) as excinfo:
            ca.publish_anchor(str(tmp_path), self._anchor(), replace=False)
        assert excinfo.value.cause == ca.IDENTITY_ANCHOR_WRITE_FAILED
        assert "writable" in str(excinfo.value)

    def test_nothing_but_an_admitted_backend_and_a_canonical_uuid_is_written(
        self, tmp_path
    ):
        with pytest.raises(ValueError):
            ca.publish_anchor(
                str(tmp_path), self._anchor(backend="RedisKVStorage"), replace=False
            )
        with pytest.raises(ValueError):
            ca.publish_anchor(
                str(tmp_path), self._anchor(storage_uuid=UUID_A.upper()), replace=False
            )
        assert not os.path.exists(ca.anchor_path(str(tmp_path)))


class TestUuid:
    def test_a_new_uuid_is_canonical(self):
        value = ca.new_storage_uuid()
        assert ca.canonical_storage_uuid(value) == value

    @pytest.mark.parametrize(
        "value", [None, 1, "", "x", UUID_A.upper(), "{" + UUID_A + "}"]
    )
    def test_non_canonical_spellings_are_not_uuids(self, value):
        assert ca.canonical_storage_uuid(value) is None
