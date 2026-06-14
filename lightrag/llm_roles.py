"""LLM role registry, configuration types, and runtime mixin.

LightRAG can route different stages of work (entity extraction, keyword
extraction, query, vlm) to distinct LLM bindings. This module owns the
static role registry (:data:`ROLES`), the per-role configuration
(:class:`RoleLLMConfig`), and the :class:`_RoleLLMMixin` that drives the
runtime: builder registration, wrapper rebuilding, hot config updates,
queue cleanup, and queue-status reporting.
"""

from __future__ import annotations

import asyncio
import inspect
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Mapping

from lightrag.utils import (
    get_env_value,
    logger,
    priority_limit_async_func_call,
)


def _optional_env_int(env_key: str) -> int | None:
    return get_env_value(env_key, None, int, special_none=True)


@dataclass(frozen=True)
class RoleSpec:
    """Static descriptor for a known LLM role.

    Adding a new role anywhere in LightRAG is a single-line edit: append a
    ``RoleSpec`` to :data:`ROLES`. Every other component (env var loop in
    ``api/config.py``, queue observability, role config update flow) iterates
    this registry rather than hard-coding role names.
    """

    name: str
    """Canonical lowercase role key (used in ``role_llm_configs`` dict and CLI/log output)."""

    env_prefix: str
    """Uppercase prefix used by the API env-var layer, e.g. ``"EXTRACT"`` for
    ``EXTRACT_LLM_BINDING`` / ``EXTRACT_MAX_ASYNC_LLM`` / ``EXTRACT_LLM_TIMEOUT``."""

    queue_name: str
    """Display name passed to ``priority_limit_async_func_call`` for log lines."""


ROLES: tuple[RoleSpec, ...] = (
    RoleSpec("extract", "EXTRACT", "extract LLM func"),
    RoleSpec("keyword", "KEYWORD", "keyword LLM func"),
    RoleSpec("query", "QUERY", "query LLM func"),
    RoleSpec("vlm", "VLM", "vlm LLM func"),
)
ROLE_NAMES: frozenset[str] = frozenset(spec.name for spec in ROLES)
ROLES_BY_NAME: dict[str, RoleSpec] = {spec.name: spec for spec in ROLES}


@dataclass
class RoleLLMConfig:
    """Per-role LLM override accepted at :class:`LightRAG` init time.

    Any field left as ``None`` falls back to the corresponding base LLM
    setting (``llm_model_func`` / ``llm_model_kwargs`` / ``llm_model_max_async``
    / ``default_llm_timeout``). When ``max_async`` is None at init and the
    user did not pass a ``role_llm_configs`` entry for the role, the value is
    additionally seeded from ``{ROLE_PREFIX}_MAX_ASYNC_LLM``. ``metadata`` seeds
    runtime observability and role-builder context.
    """

    func: Callable[..., object] | None = None
    kwargs: dict[str, Any] | None = None
    max_async: int | None = None
    timeout: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class _RoleLLMState:
    """Runtime state for one role. Internal — not part of the public API."""

    raw_func: Callable[..., object]
    kwargs: dict[str, Any] | None
    max_async: int | None
    timeout: int | None
    metadata: dict[str, Any] = field(default_factory=dict)
    wrapped: Callable[..., object] | None = None


class _RoleLLMMixin:
    """Mixin that owns the role LLM runtime on :class:`LightRAG`.

    Mixed into LightRAG only. Relies on attributes that the main class
    initializes in ``__post_init__`` (``_role_llm_states``, ``_role_llm_builders``,
    ``llm_model_func``, ``llm_model_kwargs``, ``llm_model_max_async``,
    ``default_llm_timeout``, ``embedding_func``, ``rerank_model_func``).
    """

    _SECRET_MARKERS = (
        "api_key",
        "api-key",
        "apikey",
        "access_key",
        "access-key",
        "secret",
        "token",
        "credential",
        "password",
        "passphrase",
        "pwd",
        "auth",
        "session",
    )

    @staticmethod
    def _normalize_llm_role(role: str) -> str:
        normalized = role.strip().lower()
        if normalized not in ROLE_NAMES:
            raise ValueError(f"Invalid LLM role: {role}")
        return normalized

    def register_role_llm_builder(
        self,
        builder: Callable[
            [str, dict[str, Any]], tuple[Callable[..., object], dict[str, Any] | None]
        ],
    ) -> None:
        """Register a runtime builder used by update_llm_role_config for binding/model updates."""
        self._llm_role_builder = builder

    def set_role_llm_metadata(self, role: str, **metadata: Any) -> None:
        """Store role metadata used when rebuilding a role-specific LLM function."""
        role = self._normalize_llm_role(role)
        state = self._role_llm_states[role]
        for key, value in metadata.items():
            if value is None:
                continue
            state.metadata[key] = value

    @property
    def role_llm_funcs(self) -> Mapping[str, Callable[..., object]]:
        """Read-only mapping of role name → wrapped (queue-managed) LLM func."""
        return {
            name: state.wrapped
            for name, state in self._role_llm_states.items()
            if state.wrapped is not None
        }

    @property
    def role_llm_kwargs(self) -> Mapping[str, dict[str, Any] | None]:
        """Read-only mapping of role name → effective LLM kwargs (None means inherit base)."""
        return {name: state.kwargs for name, state in self._role_llm_states.items()}

    def _get_effective_role_llm_kwargs(self, role: str) -> dict[str, Any]:
        state = self._role_llm_states[self._normalize_llm_role(role)]
        if state.kwargs is not None:
            return state.kwargs
        if state.metadata.get("is_cross_provider"):
            return {}
        return self.llm_model_kwargs

    def _get_effective_role_llm_timeout(self, role: str) -> int:
        state = self._role_llm_states[self._normalize_llm_role(role)]
        return state.timeout if state.timeout is not None else self.default_llm_timeout

    def _get_effective_role_llm_max_async(self, role: str) -> int:
        state = self._role_llm_states[self._normalize_llm_role(role)]
        return (
            state.max_async if state.max_async is not None else self.llm_model_max_async
        )

    def _wrap_llm_role_func(
        self,
        role_name: str,
        raw_func: Callable[..., object],
        max_async: int,
        timeout: int,
        model_kwargs: dict[str, Any],
    ) -> Callable[..., object]:
        spec = ROLES_BY_NAME[role_name]
        return priority_limit_async_func_call(
            max_async,
            llm_timeout=timeout,
            queue_name=spec.queue_name,
            concurrency_group=f"llm:{role_name}",
        )(
            partial(
                raw_func,
                hashing_kv=self.llm_response_cache,
                **model_kwargs,
            )
        )

    def _rebuild_role_llm_funcs(self) -> None:
        """Wrap each role's raw_func with its own priority queue.

        Base ``llm_model_func`` is intentionally NOT wrapped — concurrency
        for the base function is enforced at the role layer (every code path
        that calls an LLM goes through a role wrapper).
        """
        for spec in ROLES:
            self._rebuild_single_role_llm_func(spec.name)

    def _rebuild_single_role_llm_func(self, role: str) -> None:
        role = self._normalize_llm_role(role)
        state = self._role_llm_states[role]
        state.wrapped = self._wrap_llm_role_func(
            role,
            state.raw_func,
            self._get_effective_role_llm_max_async(role),
            self._get_effective_role_llm_timeout(role),
            self._get_effective_role_llm_kwargs(role),
        )

    async def _shutdown_llm_wrapper(self, wrapped_func: Callable[..., object]) -> None:
        shutdown = getattr(wrapped_func, "shutdown", None)
        if callable(shutdown):
            await shutdown(graceful=True)

    def _schedule_retired_llm_queue_cleanup(
        self, wrapped_func: Callable[..., object] | None
    ) -> None:
        if wrapped_func is None or not callable(
            getattr(wrapped_func, "shutdown", None)
        ):
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # The retired wrapper's queue and worker tasks are tied to the
            # event loop that first used them. Spinning up a fresh loop via
            # asyncio.run would either hang on queue.join() or touch
            # primitives bound to a closed loop. Skip cleanup with a warning
            # — call aupdate_llm_role_config() from an async context for
            # deterministic shutdown.
            logger.warning(
                "update_llm_role_config: skipping retired LLM queue cleanup "
                "because no event loop is running; call aupdate_llm_role_config() "
                "from an async context for deterministic shutdown"
            )
            return

        task = loop.create_task(self._shutdown_llm_wrapper(wrapped_func))
        self._retired_llm_queue_cleanup_tasks.add(task)
        task.add_done_callback(self._finalize_retired_llm_queue_cleanup)

    def _finalize_retired_llm_queue_cleanup(self, task: asyncio.Task) -> None:
        self._retired_llm_queue_cleanup_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Retired LLM queue cleanup failed: {e}")

    async def wait_for_retired_llm_queues(self) -> None:
        """Wait until all retired role LLM queues have drained and shut down.

        Cleanup failures are logged by ``_finalize_retired_llm_queue_cleanup``
        and intentionally swallowed here so callers can rely on this method
        always returning once every retired wrapper has finished.
        """
        while self._retired_llm_queue_cleanup_tasks:
            tasks = list(self._retired_llm_queue_cleanup_tasks)
            await asyncio.gather(*tasks, return_exceptions=True)

    def _apply_llm_role_config_update(
        self,
        role: str,
        *,
        model_func: Callable[..., object] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        max_async: int | None = None,
        timeout: int | None = None,
        binding: str | None = None,
        model: str | None = None,
        host: str | None = None,
        api_key: str | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> Callable[..., object] | None:
        role = self._normalize_llm_role(role)
        state = self._role_llm_states[role]
        old_wrapped = state.wrapped

        snapshot = _RoleLLMState(
            raw_func=state.raw_func,
            kwargs=deepcopy(state.kwargs),
            max_async=state.max_async,
            timeout=state.timeout,
            metadata=deepcopy(state.metadata),
            wrapped=state.wrapped,
        )

        try:
            if model_func is not None and not callable(model_func):
                raise TypeError("model_func must be callable")

            if model_kwargs is not None:
                state.kwargs = model_kwargs
            if max_async is not None:
                state.max_async = max_async
            if timeout is not None:
                state.timeout = timeout
            if model_func is not None:
                state.raw_func = model_func

            metadata_updated = any(
                value is not None
                for value in (binding, model, host, api_key, provider_options)
            )
            if binding is not None:
                state.metadata["binding"] = binding
            if model is not None:
                state.metadata["model"] = model
            if host is not None:
                state.metadata["host"] = host
            if api_key is not None:
                state.metadata["api_key"] = api_key
            if provider_options is not None:
                state.metadata["provider_options"] = provider_options
            if "base_binding" in state.metadata and "binding" in state.metadata:
                state.metadata["is_cross_provider"] = (
                    state.metadata["binding"] != state.metadata["base_binding"]
                )

            if metadata_updated:
                builder = getattr(self, "_llm_role_builder", None)
                if builder is None and model_func is None:
                    raise ValueError(
                        "Runtime role builder is not configured; provide model_func or register_role_llm_builder() first"
                    )
                if builder is not None:
                    built_func, built_kwargs = builder(role, state.metadata)
                    state.raw_func = built_func
                    if model_kwargs is None and built_kwargs is not None:
                        state.kwargs = built_kwargs

            self._rebuild_single_role_llm_func(role)
        except Exception:
            state.raw_func = snapshot.raw_func
            state.kwargs = snapshot.kwargs
            state.max_async = snapshot.max_async
            state.timeout = snapshot.timeout
            state.metadata = snapshot.metadata
            state.wrapped = snapshot.wrapped
            raise

        self._log_llm_role_config("updated", role=role)
        return old_wrapped

    def update_llm_role_config(
        self,
        role: str,
        *,
        model_func: Callable[..., object] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        max_async: int | None = None,
        timeout: int | None = None,
        binding: str | None = None,
        model: str | None = None,
        host: str | None = None,
        api_key: str | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> None:
        """
        Update a role-specific LLM configuration at runtime.

        Supports lightweight updates (kwargs/max_async/timeout/model_func) directly.
        For binding/model/host/api_key/provider_options updates, a role builder must
        be registered via register_role_llm_builder().
        """
        old_wrapped = self._apply_llm_role_config_update(
            role,
            model_func=model_func,
            model_kwargs=model_kwargs,
            max_async=max_async,
            timeout=timeout,
            binding=binding,
            model=model,
            host=host,
            api_key=api_key,
            provider_options=provider_options,
        )
        self._schedule_retired_llm_queue_cleanup(old_wrapped)

    async def aupdate_llm_role_config(
        self,
        role: str,
        *,
        model_func: Callable[..., object] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        max_async: int | None = None,
        timeout: int | None = None,
        binding: str | None = None,
        model: str | None = None,
        host: str | None = None,
        api_key: str | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> None:
        """Async variant of update_llm_role_config that waits for queue cleanup.

        Blocking behavior:
            This coroutine awaits a graceful shutdown of the retired role
            wrapper's priority queue. The shutdown blocks on
            ``queue.join()`` until every already-queued LLM call has been
            executed (workers always call ``task_done()`` in ``finally``,
            so in-flight requests are not cut off).

            The wait is bounded by ``max_task_duration`` of the retired
            queue, which is computed as ``llm_timeout * 2 + 15`` seconds
            (default ``180 * 2 + 15 = 375`` seconds, ~6 min 15 s). When
            this bound is reached, the drain times out and the shutdown
            falls through to forced cancellation: pending futures are
            cancelled, the queue is cleared, workers are stopped. So this
            method **never blocks indefinitely**, but with a deep backlog
            of slow LLM calls it can take up to that bound to return, and
            in-flight calls past the bound will be cancelled.

            If you need a non-blocking switch, use the sync
            ``update_llm_role_config()`` (which schedules cleanup as a
            background task) and await ``wait_for_retired_llm_queues()``
            separately when you want to confirm the old queue is gone.
        """
        old_wrapped = self._apply_llm_role_config_update(
            role,
            model_func=model_func,
            model_kwargs=model_kwargs,
            max_async=max_async,
            timeout=timeout,
            binding=binding,
            model=model,
            host=host,
            api_key=api_key,
            provider_options=provider_options,
        )
        if old_wrapped is not None:
            await self._shutdown_llm_wrapper(old_wrapped)

    @classmethod
    def _is_secret_key(cls, key: str) -> bool:
        lowered = key.lower()
        return any(marker in lowered for marker in cls._SECRET_MARKERS)

    def _scrubbed_llm_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Return a deep copy of ``metadata`` with auth-bearing fields removed.

        Auth-bearing fields are stripped entirely — not masked — because a
        masked ``"***"`` carries no information for an external consumer
        (operators already see ``binding`` / ``host`` to confirm a role is
        configured). Stripping makes the invariant simple: anything that
        appears in this output is safe to log, cache, ship over the wire.

        Components that legitimately need the raw secret (the role builder,
        provider clients) read it directly off the private
        ``_role_llm_states[role].metadata`` dict.
        """

        def scrub_value(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {
                    key: scrub_value(inner_value)
                    for key, inner_value in value.items()
                    if not self._is_secret_key(str(key))
                }
            if isinstance(value, list):
                return [scrub_value(item) for item in value]
            if isinstance(value, tuple):
                return tuple(scrub_value(item) for item in value)
            return deepcopy(value)

        return scrub_value(metadata)

    def get_llm_role_config(self, role: str | None = None) -> dict[str, Any]:
        """Return effective role LLM runtime configuration (observability snapshot).

        Each role entry exposes ``binding`` / ``model`` / ``host`` at the top
        level for convenience and again inside ``metadata`` as part of the
        full runtime snapshot (which may contain extra builder-specific
        keys). Auth-bearing fields (``api_key``, ``aws_secret_access_key``,
        ``password``, …) are **stripped entirely** from ``metadata`` — this
        method is intended for ``/health`` / WebUI / audit output and must
        never leak credentials. There is no escape hatch; runtime components
        that legitimately need the raw value read it from
        ``_role_llm_states[role].metadata`` directly.
        """

        def role_config(role_name: str) -> dict[str, Any]:
            state = self._role_llm_states[role_name]
            metadata = self._scrubbed_llm_metadata(state.metadata)
            return {
                "binding": metadata.get("binding"),
                "model": metadata.get("model"),
                "host": metadata.get("host"),
                "is_cross_provider": metadata.get("is_cross_provider", False),
                "max_async": self._get_effective_role_llm_max_async(role_name),
                "timeout": self._get_effective_role_llm_timeout(role_name),
                "has_model_kwargs": state.kwargs is not None,
                "metadata": metadata,
            }

        if role is not None:
            return role_config(self._normalize_llm_role(role))

        return {spec.name: role_config(spec.name) for spec in ROLES}

    def _log_llm_role_config(self, reason: str, role: str | None = None) -> None:
        """Log the sanitized role LLM runtime configuration."""
        if role is None:
            configs = self.get_llm_role_config()
            role_names = [spec.name for spec in ROLES]
            logger.info(f"Role LLM Configuration ({reason}):")
        else:
            normalized_role = self._normalize_llm_role(role)
            configs = {normalized_role: self.get_llm_role_config(normalized_role)}
            role_names = [normalized_role]
            logger.info(f"Role LLM Configuration ({reason}: {normalized_role}):")

        for role_name in role_names:
            cfg = configs[role_name]
            logger.info(
                " - %s: %s/%s, host=%s, max_async=%s, timeout=%s",
                role_name,
                cfg["binding"],
                cfg["model"],
                cfg["host"],
                cfg["max_async"],
                cfg["timeout"],
            )

    async def _queue_status_for_func(
        self, func: Callable[..., object] | None
    ) -> dict[str, Any]:
        if func is None:
            return {"available": False}
        # Prefer the cross-worker aggregated view (sums every gunicorn
        # worker's published snapshot; falls back to the local snapshot
        # internally on any shared-storage failure, so "available" keeps
        # meaning "this wrapper exists", never "aggregation succeeded").
        get_stats = getattr(func, "get_aggregated_queue_stats", None)
        if not callable(get_stats):
            get_stats = getattr(func, "get_queue_stats", None)
        if not callable(get_stats):
            return {"available": False}
        stats = get_stats()
        if inspect.isawaitable(stats):
            stats = await stats
        stats["available"] = True
        return stats

    async def get_llm_queue_status(self, include_base: bool = True) -> dict[str, Any]:
        """Return queue status for each role's wrapped LLM func.

        The base ``llm_model_func`` is no longer queue-wrapped, so it is not
        reported here. ``include_base`` is kept for signature compatibility
        but has no effect.
        """
        del include_base  # base is unwrapped — see docstring

        result: dict[str, Any] = {}
        for spec in ROLES:
            state = self._role_llm_states.get(spec.name)
            result[spec.name] = await self._queue_status_for_func(
                state.wrapped if state else None
            )
        return result

    async def get_embedding_queue_status(self) -> dict[str, Any]:
        """Return queue status for the wrapped embedding function."""
        return await self._queue_status_for_func(
            self.embedding_func.func if self.embedding_func is not None else None
        )

    async def get_rerank_queue_status(self) -> dict[str, Any]:
        """Return queue status for the wrapped rerank function."""
        return await self._queue_status_for_func(self.rerank_model_func)
