"""Third-party parser engine discovery (``lightrag.parsers`` entry points).

A third-party package exposes parser engines by declaring an entry point in
the ``lightrag.parsers`` group::

    # pyproject.toml of the third-party package
    [project.entry-points."lightrag.parsers"]
    myengine = "my_pkg.lightrag_plugin:register"

Each entry point must resolve to a **zero-argument callable** that performs
its own :func:`lightrag.parser.registry.register_parser` call(s).  The
callable should be import-cheap: defer the parser implementation import to
the ``ParserSpec.impl`` string (the registry already loads it lazily).

:func:`load_third_party_parsers` is invoked once per process by both
entrypoints that drive parsers — the API server (``create_app``, before
routing-rule validation so ``LIGHTRAG_PARSER`` may reference third-party
engine names) and the debug CLI (``lightrag.parser.cli.main``, before
``--engine`` choices are built).  Library users embedding LightRAG directly
can call it themselves before constructing pipelines.

See ``docs/ThirdPartyParser-zh.md`` for the full plugin authoring guide.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

ENTRY_POINT_GROUP = "lightrag.parsers"

logger = logging.getLogger("lightrag")

_loaded = False


def load_third_party_parsers(*, force: bool = False) -> list[str]:
    """Discover and run all ``lightrag.parsers`` entry points.

    Idempotent per process (``force=True`` re-runs, for tests).  Returns the
    names of the entry points that loaded successfully.  A failing plugin is
    logged with its origin and skipped — one broken third-party package must
    not take down server startup or the debug CLI; the built-in engines are
    registered statically and are never affected.
    """
    global _loaded
    if _loaded and not force:
        return []
    _loaded = True

    loaded: list[str] = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            register = ep.load()
            register()
        except Exception as e:
            logger.error(
                "[parser-plugins] failed to load parser plugin %r (%s): %s",
                ep.name,
                ep.value,
                e,
            )
            continue
        loaded.append(ep.name)
        logger.info("[parser-plugins] loaded parser plugin %r (%s)", ep.name, ep.value)
    return loaded
