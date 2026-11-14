"""Where tiktoken keeps its downloaded BPE files when nothing is configured.

tiktoken itself caches into ``$TIKTOKEN_CACHE_DIR``, else
``$DATA_GYM_CACHE_DIR``, else ``<tempdir>/data-gym-cache``. The temp-dir
fallback is a poor home for an offline deployment's pre-downloaded cache:
``/tmp`` is commonly a tmpfs, aged out by ``systemd-tmpfiles``, swept by
operators, and writable by every local user. LightRAG therefore documents
``~/.tiktoken_cache`` as the default location, and ``lightrag-download-cache``
writes there when no directory is given.

Rules:

- Load every encoding in ``lightrag/`` through :func:`load_tiktoken_encoding`
  rather than calling ``tiktoken.get_encoding`` / ``encoding_for_model``
  directly; otherwise that site neither reads ``~/.tiktoken_cache`` nor gets
  the read-only fallback below.
- ``~/.tiktoken_cache`` is only ever *adopted* when it already exists, never
  created: once ``TIKTOKEN_CACHE_DIR`` is set, tiktoken raises on a failed
  cache write instead of ignoring it.
- An adopted directory that cannot be written still serves the encodings it
  holds. When a missing encoding then fails to be cached there, the load is
  retried with tiktoken's own default (best-effort writes), a warning is
  logged, and the directory is not adopted again in this process. Errors
  from a writable directory, and from a directory the user configured, are
  re-raised unchanged.
- Any explicit setting wins, including ``TIKTOKEN_CACHE_DIR=""`` (tiktoken's
  "disable caching" switch) and ``DATA_GYM_CACHE_DIR``.

See docs/OfflineDeployment.md ("Where the Cache Lives") for the full order.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable, TypeVar

DEFAULT_TIKTOKEN_CACHE_DIR = "~/.tiktoken_cache"

_ENV_VARS = ("TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR")

logger = logging.getLogger("lightrag")

T = TypeVar("T")

_state_lock = threading.Lock()
# The directory this module exported into TIKTOKEN_CACHE_DIR, if any.
_adopted_dir: str | None = None
# Set once an adopted directory proved unwritable; stops re-adoption.
_adoption_disabled = False


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
    global _adopted_dir
    with _state_lock:
        if _adoption_disabled or configured_tiktoken_cache_dir() is not None:
            return None
        path = default_tiktoken_cache_dir()
        if path is None or not os.path.isdir(path):
            return None
        os.environ["TIKTOKEN_CACHE_DIR"] = path
        _adopted_dir = path
    logger.debug("Using tiktoken cache directory %s", path)
    return path


def _drop_unwritable_adoption() -> str | None:
    """Undo our own adoption if it is still in effect and its directory is
    not writable; return the dropped directory, else None."""
    global _adopted_dir, _adoption_disabled
    with _state_lock:
        path = _adopted_dir
        if path is None or os.environ.get("TIKTOKEN_CACHE_DIR") != path:
            return None
        if os.access(path, os.W_OK | os.X_OK):
            return None
        os.environ.pop("TIKTOKEN_CACHE_DIR", None)
        _adopted_dir = None
        _adoption_disabled = True
    return path


def load_tiktoken_encoding(load: Callable[[], T]) -> T:
    """Run ``load`` (a tiktoken encoding lookup) with the default cache dir
    applied, falling back to tiktoken's own default if the implicitly adopted
    directory turns out to be unwritable."""
    apply_default_tiktoken_cache_dir()
    try:
        return load()
    except OSError as exc:
        dropped = _drop_unwritable_adoption()
        if dropped is None:
            raise
        logger.warning(
            "tiktoken cache directory %s is not writable (%s); falling back to "
            "tiktoken's default cache location for this process. Make it "
            "writable, complete it with lightrag-download-cache, or set "
            "TIKTOKEN_CACHE_DIR.",
            dropped,
            exc,
        )
        return load()
