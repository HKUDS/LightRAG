"""``lightrag-server``'s ``configure_logging`` creates the directory ``LOG_DIR``
names, not only its parent."""

from contextlib import contextmanager
import logging

import pytest

pytestmark = pytest.mark.offline

CONFIGURED_LOGGERS = ("uvicorn", "uvicorn.access", "uvicorn.error", "lightrag")


@contextmanager
def restored_loggers():
    """Put back the loggers ``configure_logging`` rewrites, closing the file
    handlers it opened."""
    saved = {
        name: (lg.handlers[:], lg.filters[:], lg.level, lg.propagate)
        for name in CONFIGURED_LOGGERS
        for lg in [logging.getLogger(name)]
    }
    try:
        yield
    finally:
        for name, (handlers, filters, level, propagate) in saved.items():
            lg = logging.getLogger(name)
            for handler in lg.handlers:
                if handler not in handlers:
                    handler.close()
            lg.handlers, lg.filters = handlers, filters
            lg.setLevel(level)
            lg.propagate = propagate


@pytest.mark.parametrize(
    "relative", [True, False], ids=["relative-log-dir", "nested-absolute-log-dir"]
)
def test_configure_logging_creates_the_log_dir(relative, tmp_path, monkeypatch):
    # Importing the server module parses sys.argv as server arguments.
    monkeypatch.setattr("sys.argv", ["lightrag-server"])
    from lightrag.api import lightrag_server

    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "logs" if relative else tmp_path / "nested" / "logs"
    monkeypatch.setenv("LOG_DIR", "logs" if relative else str(log_dir))
    with restored_loggers():
        lightrag_server.configure_logging()
        logger = logging.getLogger("lightrag")
        logger.info("server logging is ready")
        for handler in logger.handlers:
            handler.flush()

    log_file = log_dir / lightrag_server.DEFAULT_LOG_FILENAME
    assert "server logging is ready" in log_file.read_text(encoding="utf-8")
