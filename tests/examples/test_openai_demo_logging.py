"""Run each demo's real ``configure_logging`` in an isolated interpreter.

The demo module is imported for real, so the function runs against its own
globals; nothing is copied out of it. Each case runs with a directory of its
own as cwd, which keeps the demo's import-time ``./dickens`` directory and
any relative ``LOG_DIR`` out of the repository.

One child serves every case, importing the demo afresh for each:
``configure_logging`` resets the handlers it installs and ``dictConfig``
closes the previous ones, which is what lets a single interpreter stand in
for one per case.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.offline

REPO_ROOT = Path(__file__).resolve().parents[2]

DEMOS = ("lightrag_openai_demo.py", "lightrag_openai_compatible_demo.py")
RELATIVE = {"relative-log-dir": True, "nested-absolute-log-dir": False}

CHILD = """
import importlib.util, json, logging, os, sys, traceback
out = {}
for case in json.loads(sys.argv[1]):
    try:
        os.chdir(case["cwd"])
        os.environ["LOG_DIR"] = case["log_dir"]
        spec = importlib.util.spec_from_file_location(
            f"demo_under_test_{len(out)}", case["demo"])
        demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo)
        demo.configure_logging()
        demo.logger.info("demo logging is ready")
        for handler in logging.getLogger("lightrag").handlers:
            handler.flush()
        out[case["id"]] = None
    except Exception:
        out[case["id"]] = traceback.format_exc()
logging.shutdown()
print(json.dumps(out))
"""


def _case_id(demo: str, relative_id: str) -> str:
    return f"{relative_id}-{demo}"


@pytest.fixture(scope="module")
def demo_runs(tmp_path_factory):
    """Run every case in one child; map case id to (error, cwd, log dir)."""
    cases = []
    layout = {}
    for demo in DEMOS:
        for relative_id, relative in RELATIVE.items():
            case_id = _case_id(demo, relative_id)
            cwd = tmp_path_factory.mktemp("demo-logging")
            log_dir = Path("logs") if relative else cwd / "nested" / "logs"
            layout[case_id] = (cwd, log_dir)
            cases.append(
                {
                    "id": case_id,
                    "cwd": str(cwd),
                    "log_dir": str(log_dir),
                    "demo": str(REPO_ROOT / "examples" / demo),
                }
            )
    # Import this checkout's lightrag, not whichever one is installed: the
    # child's sys.path[0] is not the repository.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH")) if p
    )
    result = subprocess.run(
        [sys.executable, "-c", CHILD, json.dumps(cases)],
        cwd=cases[0]["cwd"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    errors = json.loads(result.stdout.strip().splitlines()[-1])
    return {case_id: (errors[case_id], *layout[case_id]) for case_id in layout}


@pytest.mark.parametrize("demo", DEMOS)
@pytest.mark.parametrize("relative_id", list(RELATIVE))
def test_log_directory_is_created(demo, relative_id, demo_runs):
    error, cwd, log_dir = demo_runs[_case_id(demo, relative_id)]
    assert error is None, error
    files = list((cwd / log_dir).glob("*.log"))
    assert len(files) == 1
    assert "demo logging is ready" in files[0].read_text(encoding="utf-8")
