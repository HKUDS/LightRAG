"""Per-workspace pools for LightRAG and DocumentManager instances.

Enables a single server process to serve multiple workspaces, selected
per-request via the ``LIGHTRAG-WORKSPACE`` HTTP header.

Module placement: ``lightrag/api/workspace_pool.py`` — a dedicated module
so the pool classes can be unit-tested independently and imported without
pulling in the ~2700-line ``lightrag_server.py``.
"""

import asyncio
from collections.abc import Callable

from lightrag import LightRAG
from lightrag.api.document_manager import DocumentManager


class RagPool:
    """Lazily-initialized pool of LightRAG instances keyed by workspace.

    Concurrency model (two-layer lock):

    - Fast path: instance already cached → O(1) dict lookup, no lock.
    - ``_lock_factory``: lightweight lock guarding ``_locks.setdefault()`` (µs).
    - ``_locks[ws]``: per-workspace lock covering the full creation (including
      ``initialize_storages()``, which may take seconds).  Only requests for
      the *same* workspace queue here — which is correct, because they
      depend on this initialization completing.

    Configuration model (config_factory):

    Instead of storing a shared ``base_config`` dict and trying to copy it
    per workspace, the pool accepts a ``config_factory(workspace) -> dict``
    callable.  Each call builds a fresh configuration from scratch:

    - Singletons (LLM functions, clients, server-infos) are captured by the
      factory's closure and shared by reference.
    - Per-workspace values (kwargs dicts, addon_params, role_llm_configs)
      are re-constructed on every call — no deepcopy needed, no risk of
      accidentally sharing mutable state between workspaces.

    Callbacks:

    - *role_llm_builder*: registered on every new instance so that runtime
      ``update_llm_role_config()`` calls work on pooled instances.
    - *on_create*: optional hook invoked immediately after construction and
      builder registration, but before ``initialize_storages()``.  Useful
      for logging (e.g. ``_log_role_provider_options``) or additional setup.

    Lifecycle:

    ``get()`` is valid during normal request and background-task execution.
    ``shutdown_all()`` is called only during server lifespan cleanup, after
    the ASGI server has stopped accepting new requests and in-flight work
    has drained.  There is no public ``evict()``, no ``_closed`` flag, and
    no runtime draining — simplicity over premature sophistication.
    """

    def __init__(
        self,
        config_factory: Callable[[str], dict],
        *,
        role_llm_builder: Callable | None = None,
        on_create: Callable[[LightRAG], None] | None = None,
    ):
        """
        Args:
            config_factory: Called with the workspace name to produce a
                fresh kwargs dict for ``LightRAG(**kwargs)``.  The returned
                dict is NOT retained or mutated by the pool.
            role_llm_builder: Optional callback registered on every new
                ``LightRAG`` instance via ``register_role_llm_builder()``
                so that runtime role-LLM updates work correctly.
            on_create: Optional callback invoked with the new ``LightRAG``
                instance after construction and builder registration, but
                before ``initialize_storages()``.
        """
        self._config_factory = config_factory
        self._role_llm_builder = role_llm_builder
        self._on_create = on_create
        self._rags: dict[str, LightRAG] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._lock_factory = asyncio.Lock()

    async def get(self, workspace: str) -> LightRAG:
        """Return the LightRAG instance for *workspace*, creating it if needed."""
        # ── Fast path: already cached, no lock ──
        rag = self._rags.get(workspace)
        if rag is not None:
            return rag

        # ── Layer 1: safely obtain the per-workspace lock ──
        async with self._lock_factory:
            ws_lock = self._locks.setdefault(workspace, asyncio.Lock())

        # ── Layer 2: create under per-workspace lock ──
        async with ws_lock:
            # Double-check: another request may have created it while we
            # queued for ws_lock.
            rag = self._rags.get(workspace)
            if rag is not None:
                return rag

            # Build fresh config — no shared mutable state between workspaces.
            config = self._config_factory(workspace)
            rag = LightRAG(**config)

            # Register runtime LLM builder (needed for update_llm_role_config).
            if self._role_llm_builder is not None:
                rag.register_role_llm_builder(self._role_llm_builder)

            # Pre-init hook (e.g. logging).
            if self._on_create is not None:
                self._on_create(rag)

            try:
                await rag.initialize_storages()
                # Run data migration on first creation (lazy strategy).
                await rag.check_and_migrate_data()
            except Exception:
                # If storages were partially opened, finalize them to
                # avoid leaking connections (PostgreSQL pools, Neo4j
                # sessions, Redis connections, etc.).  finalize_storages()
                # is idempotent — safe to call even if initialization
                # didn't complete.
                await rag.finalize_storages()
                raise
            self._rags[workspace] = rag
            return rag

    async def _evict(self, workspace: str) -> None:
        """Internal: release resources for *workspace*.  Not exposed publicly
        in the first PR because eviction during active traffic can race with
        in-flight requests (see §14).  Call only from ``shutdown_all()``,
        which runs during server shutdown.
        """
        async with self._lock_factory:
            lock = self._locks.pop(workspace, None)
        if lock is not None:
            async with lock:
                rag = self._rags.pop(workspace, None)
                if rag is not None:
                    await rag.finalize_storages()

    async def shutdown_all(self) -> None:
        """Finalize all cached instances.

        Called during server lifespan cleanup.  Individual finalization
        errors are collected; a single ``RuntimeError`` is raised after
        all workspaces have been attempted so that one failure does not
        skip the rest.
        """
        async with self._lock_factory:
            workspaces = list(self._locks.keys())

        errors: list[tuple[str, Exception]] = []
        for ws in workspaces:
            try:
                await self._evict(ws)
            except Exception as exc:
                errors.append((ws, exc))

        if errors:
            msg = "; ".join(f"{ws}: {exc}" for ws, exc in errors)
            raise RuntimeError(
                f"RagPool.shutdown_all: {len(errors)} workspace(s) failed "
                f"finalization: {msg}"
            ) from errors[0][1]


class DocManagerPool:
    """Per-workspace pool of DocumentManager instances.

    Uses the same two-layer lock structure as RagPool for interface
    consistency.  While ``DocumentManager.__init__`` is currently
    near-instantaneous (path join + mkdir), the two-layer design
    future-proofs against a heavier init and reduces cognitive overhead
    by using one pattern everywhere.
    """

    def __init__(self, base_input_dir: str):
        self._base_input_dir = base_input_dir
        self._managers: dict[str, DocumentManager] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._lock_factory = asyncio.Lock()

    async def get(self, workspace: str) -> DocumentManager:
        """Return the DocumentManager for *workspace*, creating it if needed."""
        mgr = self._managers.get(workspace)
        if mgr is not None:
            return mgr

        async with self._lock_factory:
            ws_lock = self._locks.setdefault(workspace, asyncio.Lock())

        async with ws_lock:
            mgr = self._managers.get(workspace)
            if mgr is not None:
                return mgr
            mgr = DocumentManager(self._base_input_dir, workspace=workspace)
            self._managers[workspace] = mgr
            return mgr

    async def _evict(self, workspace: str) -> None:
        """Internal: see RagPool._evict docstring."""
        async with self._lock_factory:
            lock = self._locks.pop(workspace, None)
        if lock is not None:
            async with lock:
                self._managers.pop(workspace, None)

    async def shutdown_all(self) -> None:
        """Finalize all cached managers.  Mirrors RagPool.shutdown_all()."""
        async with self._lock_factory:
            workspaces = list(self._locks.keys())

        errors: list[tuple[str, Exception]] = []
        for ws in workspaces:
            try:
                await self._evict(ws)
            except Exception as exc:
                errors.append((ws, exc))

        if errors:
            msg = "; ".join(f"{ws}: {exc}" for ws, exc in errors)
            raise RuntimeError(
                f"DocManagerPool.shutdown_all: {len(errors)} workspace(s) "
                f"failed finalization: {msg}"
            ) from errors[0][1]
