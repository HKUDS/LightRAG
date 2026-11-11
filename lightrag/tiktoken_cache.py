"""Where tiktoken keeps its downloaded BPE files when nothing is configured.

tiktoken itself caches into ``$TIKTOKEN_CACHE_DIR``, else
``$DATA_GYM_CACHE_DIR``, else ``<tempdir>/data-gym-cache``. The temp-dir
fallback is a poor home for an offline deployment's pre-downloaded cache:
``/tmp`` is commonly a tmpfs, aged out by ``systemd-tmpfiles``, swept by
operators, and writable by every local user. LightRAG therefore documents
``~/.tiktoken_cache`` as the default location, and ``lightrag-download-cache``
writes there when no directory is given.

Rules:

- Call :func:`apply_default_tiktoken_cache_dir` immediately before any
  ``tiktoken.get_encoding`` / ``tiktoken.encoding_for_model`` call in
  ``lightrag/``. tiktoken reads the environment on every cache lookup, not
  at import time, so calling it after ``import tiktoken`` is fine.
- It only ever *adopts* ``~/.tiktoken_cache`` when that directory already
  exists. It never creates it: once ``TIKTOKEN_CACHE_DIR`` is set, tiktoken
  raises on a failed cache write instead of ignoring it, so creating a
  directory under an unwritable or absent home would turn a silent cache
  miss into a hard failure.
- Any explicit setting wins, including ``TIKTOKEN_CACHE_DIR=""`` (tiktoken's
  "disable caching" switch) and ``DATA_GYM_CACHE_DIR``.
"""

from __future__ import annotations

import logging
import os

DEFAULT_TIKTOKEN_CACHE_DIR = "~/.tiktoken_cache"

_ENV_VARS = ("TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR")

logger = logging.getLogger("lightrag")


def default_tiktoken_cache_dir() -> str | None:
    """Return the absolute ``~/.tiktoken_cache`` path, or None if the home
    directory cannot be resolved (``expanduser`` then leaves ``~`` as is)."""
    path = os.path.expanduser(DEFAULT_TIKTOKEN_CACHE_DIR)
    if path.startswith("~"):
        return None
    return os.path.abspath(path)


def configured_tiktoken_cache_dir() -> str | None:
    """Return the cache directory tiktoken is explicitly told to use, if any.

    An empty ``TIKTOKEN_CACHE_DIR`` is returned as ``""`` (caching disabled),
    which is distinct from None (nothing configured).
    """
    for name in _ENV_VARS:
        if name in os.environ:
            return os.environ[name]
    return None


def apply_default_tiktoken_cache_dir() -> str | None:
    """Point tiktoken at ``~/.tiktoken_cache`` when it exists and nothing else
    is configured; return the directory adopted, or None if left untouched."""
    if configured_tiktoken_cache_dir() is not None:
        return None
    path = default_tiktoken_cache_dir()
    if path is None or not os.path.isdir(path):
        return None
    os.environ["TIKTOKEN_CACHE_DIR"] = path
    logger.debug("Using tiktoken cache directory %s", path)
    return path
