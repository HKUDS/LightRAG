"""Test package markers for shared helper imports.

Also keeps every ``.env`` file out of the test process. LightRAG modules call
``load_dotenv(".env")`` at import time and some dependencies call a bare
``load_dotenv()`` that searches parent directories (pymilvus does), so a
developer's ``.env`` used to leak into tests that pass in CI and fail locally,
or the reverse. This package runs before ``tests/conftest.py`` and before any
test module, so the switch is in place before the first ``load_dotenv`` call.

``setdefault`` keeps an explicit choice: an integration run that keeps its
credentials in ``.env`` can opt back in with ``PYTHON_DOTENV_DISABLED=0``.
A test that needs real ``.env`` loading (e.g. in a subprocess) must drop the
variable from that environment itself.
"""

import os

os.environ.setdefault("PYTHON_DOTENV_DISABLED", "1")
