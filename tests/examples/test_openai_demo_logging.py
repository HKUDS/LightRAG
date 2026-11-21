"""Run each demo's real ``configure_logging`` in an isolated interpreter.

The demo module is imported for real, so the function runs against its own
globals; nothing is copied out of it. The child runs with the temporary
directory as cwd, which keeps the demo's import-time ``./dickens`` directory
and any relative ``LOG_DIR`` out of the repository.
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.offline

REPO_ROOT = Path(__file__).resolve().parents[2]

CHILD = """
import importlib.util, logging, sys
spec = importlib.util.spec_from_file_location("demo_under_test", sys.argv[1])
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)
demo.configure_logging()
demo.logger.info("demo logging is ready")
logging.shutdown()
"""


@pytest.mark.parametrize(
    "demo", ["lightrag_openai_demo.py", "lightrag_openai_compatible_demo.py"]
)
@pytest.mark.parametrize(
    "relative", [True, False], ids=["relative-log-dir", "nested-absolute-log-dir"]
)
def test_log_directory_is_created(demo, relative, tmp_path, monkeypatch):
    log_dir = Path("logs") if relative else tmp_path / "nested" / "logs"
    monkeypatch.setenv("LOG_DIR", str(log_dir))
    # Import this checkout's lightrag, not whichever one is installed: the
    # child's sys.path[0] is the temporary cwd, not the repository.
    pythonpath = os.pathsep.join(
        p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH")) if p
    )
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    result = subprocess.run(
        [sys.executable, "-c", CHILD, str(REPO_ROOT / "examples" / demo)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    files = list((tmp_path / log_dir).glob("*.log"))
    assert len(files) == 1
    assert "demo logging is ready" in files[0].read_text(encoding="utf-8")
