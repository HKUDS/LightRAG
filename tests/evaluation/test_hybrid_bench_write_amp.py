"""Write-amp orchestration pins, offline: the driver subprocess receives an
ABSOLUTE corpus path because its cwd is a temp dir, not the caller's.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from lightrag.evaluation.hybrid_bench import write_amp

pytestmark = pytest.mark.offline


def test_run_driver_resolves_relative_corpus_path(tmp_path, monkeypatch):
    recorded = {}

    def fake_run(argv, **kwargs):
        recorded["argv"] = argv
        recorded["cwd"] = kwargs["cwd"]
        Path(argv[6]).write_text(
            json.dumps({"p95_seconds": 0.1, "corpus_sha256": "x"}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(write_amp.subprocess, "run", fake_run)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    result = write_amp._run_driver(
        "python",
        elsewhere,
        Path("relative/corpus.jsonl"),
        "wab_test",
        8,
        4,
        tmp_path / "out.json",
    )
    expected = str(Path("relative/corpus.jsonl").resolve())
    assert recorded["argv"][2] == expected
    assert Path(recorded["argv"][2]).is_absolute()
    assert recorded["cwd"] == str(elsewhere)
    assert result["p95_seconds"] == 0.1
