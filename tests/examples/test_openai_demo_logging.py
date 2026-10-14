"""Run the actual demo logging setup in an isolated interpreter."""

from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "demo", ["lightrag_openai_demo.py", "lightrag_openai_compatible_demo.py"]
)
@pytest.mark.parametrize("relative", [True, False])
def test_log_directory_is_created(demo, relative, tmp_path, monkeypatch):
    log_dir = Path("logs") if relative else tmp_path / "nested" / "logs"
    monkeypatch.setenv("LOG_DIR", str(log_dir))
    source = Path(__file__).resolve().parents[2] / "examples" / demo
    script = f"""
import ast, logging, logging.config, os
from pathlib import Path
from unittest.mock import Mock
source = Path({str(source)!r})
tree = ast.parse(source.read_text(encoding='utf-8'))
node = next(n for n in tree.body if getattr(n, 'name', None) == 'configure_logging')
logger = logging.getLogger('lightrag')
namespace = dict(os=os, logging=logging, logger=logger, set_verbose_debug=Mock())
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
namespace['configure_logging']()
logger.info('demo logging is ready')
logging.shutdown()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    files = list((tmp_path / log_dir).glob("*.log"))
    assert len(files) == 1
    assert "demo logging is ready" in files[0].read_text(encoding="utf-8")
