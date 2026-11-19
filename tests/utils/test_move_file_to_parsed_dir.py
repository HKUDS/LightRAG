from types import SimpleNamespace

import pytest

from lightrag import utils

pytestmark = pytest.mark.offline


async def test_move_preserves_existing_timestamp_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "time", SimpleNamespace(time=lambda: 1_700_000_000))
    parsed_dir = tmp_path / utils.PARSED_DIR_NAME
    parsed_dir.mkdir()
    (parsed_dir / "report.txt").touch()
    for slot in range(1, 1000):
        (parsed_dir / f"report_{slot:03d}.txt").touch()
    existing_archive = parsed_dir / "report_1700000000.txt"
    existing_archive.write_bytes(b"previous report")
    source = tmp_path / "report.txt"
    source.write_bytes(b"new report")

    archived = await utils.move_file_to_parsed_dir(source)

    assert existing_archive.read_bytes() == b"previous report"
    assert archived == parsed_dir / "report_1700000001.txt"
    assert archived.read_bytes() == b"new report"
    assert not source.exists()
