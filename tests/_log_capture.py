"""Route the ``lightrag`` logger's records into caplog, exactly once.

``lightrag.utils`` sets ``propagate = False`` on the ``lightrag`` logger, so
caplog's handler on the root logger never sees its records. Whether caplog
sees them at all then depends on the pytest version:

- pytest < 9.1 attaches caplog's handler to the root logger only, so without
  help it captures nothing from ``lightrag``.
- pytest >= 9.1 also attaches it to every non-propagating logger, so it
  already captures each record once.

Turning ``propagate`` back on — the old workaround — fixes the first case and
breaks the second: the record then reaches the same handler twice, once on
``lightrag`` and once via root, and any exact-count assertion doubles.

Attaching caplog's handler to ``lightrag`` when, and only when, it is not
already there is correct on both. Do not set ``propagate`` in a test that
asserts on caplog; use the ``lightrag_caplog`` fixture (``tests/conftest.py``)
or, inside a helper that receives ``caplog``, :func:`capture_lightrag_logs`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

LIGHTRAG_LOGGER_NAME = "lightrag"


@contextmanager
def capture_lightrag_logs(
    caplog: pytest.LogCaptureFixture,
    logger_name: str = LIGHTRAG_LOGGER_NAME,
) -> Iterator[pytest.LogCaptureFixture]:
    """Attach caplog's handler to ``logger_name`` for the block, if absent.

    Removes only a handler it attached itself, so pytest's own attachment is
    left alone. Records captured inside the block stay in ``caplog.records``
    after it exits.
    """
    logger = logging.getLogger(logger_name)
    attached = caplog.handler not in logger.handlers
    if attached:
        logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        if attached:
            logger.removeHandler(caplog.handler)
