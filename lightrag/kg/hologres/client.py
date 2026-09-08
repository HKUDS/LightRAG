"""Restricted asyncpg client for the isolated Hologres backend."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence
from contextlib import AsyncExitStack
from enum import Enum
import re
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

from .config import HologresConfig

if TYPE_CHECKING:
    from .capabilities import CapabilityReport


class HologresClientError(RuntimeError):
    """Base class for sanitized Hologres client failures."""


class HologresSqlError(HologresClientError):
    """Raised before execution when SQL violates the restricted contract."""


class HologresOperationError(HologresClientError):
    """Raised when a database operation fails without exposing server text."""


class HologresCapabilityError(HologresClientError):
    """Raised when an optional operation lacks a proven capability."""


class OperationKind(str, Enum):
    READ = "read"
    WRITE = "write"


class _SetupResetProbeObservation(NamedTuple):
    schema_name: Any
    bound: Any
    same_connection_reacquired: bool
    application_name: Any


class _QuotedIdentifier(NamedTuple):
    value: str


_SqlToken = str | _QuotedIdentifier
_ScannedToken = tuple[_SqlToken, int]


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_DESCRIPTOR = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_DOLLAR_QUOTE = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")
_TRANSACTION_PREFIXES = (
    ("BEGIN",),
    ("START", "TRANSACTION"),
    ("COMMIT",),
    ("END",),
    ("ROLLBACK",),
    ("ABORT",),
    ("SAVEPOINT",),
    ("RELEASE",),
    ("SET", "TRANSACTION"),
    ("SET", "LOCAL", "TRANSACTION"),
    ("SET", "SESSION", "TRANSACTION"),
    ("SET", "SESSION", "CHARACTERISTICS", "AS", "TRANSACTION"),
    ("PREPARE", "TRANSACTION"),
)
_FORBIDDEN_PREFIXES = (*_TRANSACTION_PREFIXES, ("CALL",), ("DO",))
_READ_COMMANDS = frozenset({"SELECT", "VALUES", "TABLE", "SHOW"})
_WITH_READ_COMMANDS = frozenset({"SELECT", "VALUES", "TABLE"})
_EXPLAIN_ANALYZE_TOKENS = frozenset({"ANALYZE", "ANALYSE"})
_RESET_STATEMENTS = (
    "SELECT pg_advisory_unlock_all()",
    "CLOSE ALL",
    "UNLISTEN *",
    "RESET ALL",
)
_T = TypeVar("_T")


def validate_identifier(identifier: str) -> str:
    """Validate one unquoted PostgreSQL identifier component."""

    if not isinstance(identifier, str) or _IDENTIFIER.fullmatch(identifier) is None:
        raise HologresSqlError("Invalid SQL identifier")
    return identifier


def quote_identifier(identifier: str) -> str:
    """Validate and quote one identifier component."""

    return f'"{validate_identifier(identifier)}"'


def quote_qualified_identifier(*components: str) -> str:
    """Validate and quote a qualified identifier component by component."""

    if not components:
        raise HologresSqlError("At least one SQL identifier is required")
    return ".".join(quote_identifier(component) for component in components)


def _consume_single_quoted(
    sql: str, start: int, *, backslash_escapes: bool
) -> int:
    index = start + 1
    while index < len(sql):
        character = sql[index]
        if backslash_escapes and character == "\\" and index + 1 < len(sql):
            index += 2
            continue
        if character == "'":
            if index + 1 < len(sql) and sql[index + 1] == "'":
                index += 2
                continue
            return index + 1
        index += 1
    raise HologresSqlError("Unterminated SQL string")


def _consume_double_quoted(
    sql: str, start: int
) -> tuple[int, _QuotedIdentifier]:
    index = start + 1
    value: list[str] = []
    while index < len(sql):
        character = sql[index]
        if character == '"':
            if index + 1 < len(sql) and sql[index + 1] == '"':
                value.append('"')
                index += 2
                continue
            return index + 1, _QuotedIdentifier("".join(value))
        value.append(character)
        index += 1
    raise HologresSqlError("Unterminated quoted identifier")


def _consume_dollar_quoted(sql: str, start: int, delimiter: str) -> int:
    end = sql.find(delimiter, start + len(delimiter))
    if end < 0:
        raise HologresSqlError("Unterminated dollar-quoted string")
    return end + len(delimiter)


def _consume_block_comment(sql: str, start: int) -> int:
    index = start + 2
    depth = 1
    while index < len(sql):
        if sql.startswith("/*", index):
            depth += 1
            index += 2
            continue
        if sql.startswith("*/", index):
            depth -= 1
            index += 2
            if depth == 0:
                return index
            continue
        index += 1
    raise HologresSqlError("Unterminated block comment")


def _scan_statement_tokens(sql: str) -> tuple[_ScannedToken, ...]:
    """Lex one guarded statement into tokens and parenthesis depths."""

    if not isinstance(sql, str):
        raise HologresSqlError("SQL must be text")

    index = 0
    parenthesis_depth = 0
    tokens: list[_ScannedToken] = []
    while index < len(sql):
        character = sql[index]
        if character == "'":
            has_escape_prefix = (
                index > 0
                and sql[index - 1] in {"e", "E"}
                and (
                    index == 1
                    or not (
                        sql[index - 2].isalnum()
                        or sql[index - 2] in {"_", "$"}
                    )
                )
            )
            index = _consume_single_quoted(
                sql, index, backslash_escapes=has_escape_prefix
            )
            continue
        if character == '"':
            index, quoted_identifier = _consume_double_quoted(sql, index)
            tokens.append((quoted_identifier, parenthesis_depth))
            continue
        if character == "$":
            can_start_dollar_quote = index == 0 or not (
                sql[index - 1].isalnum() or sql[index - 1] in {"_", "$"}
            )
            match = _DOLLAR_QUOTE.match(sql, index) if can_start_dollar_quote else None
            if match is not None:
                index = _consume_dollar_quoted(sql, index, match.group(0))
                continue
        if sql.startswith("--", index):
            newline = sql.find("\n", index + 2)
            index = len(sql) if newline < 0 else newline + 1
            continue
        if sql.startswith("/*", index):
            index = _consume_block_comment(sql, index)
            continue
        if character == "(":
            parenthesis_depth += 1
            index += 1
            continue
        if character == ")":
            parenthesis_depth -= 1
            if parenthesis_depth < 0:
                raise HologresSqlError("Unbalanced SQL parentheses")
            index += 1
            continue
        if character == ";" and parenthesis_depth == 0:
            raise HologresSqlError("Top-level semicolons are not allowed")
        if character.isalpha() or character == "_":
            end = index + 1
            while end < len(sql) and (sql[end].isalnum() or sql[end] == "_"):
                end += 1
            tokens.append((sql[index:end].upper(), parenthesis_depth))
            index = end
            continue
        if character.isdigit():
            end = index + 1
            while end < len(sql) and sql[end].isdigit():
                end += 1
            tokens.append((sql[index:end], parenthesis_depth))
            index = end
            continue
        index += 1

    if parenthesis_depth != 0:
        raise HologresSqlError("Unbalanced SQL parentheses")
    if not any(depth == 0 for _token, depth in tokens):
        raise HologresSqlError("SQL statement is empty")
    return tuple(tokens)


def _top_level_tokens(tokens: tuple[_ScannedToken, ...]) -> tuple[_SqlToken, ...]:
    return tuple(token for token, depth in tokens if depth == 0)


def validate_single_statement(sql: str) -> str:
    """Reject transaction control, CALL, and top-level statement separators.

    The scan is deliberately lexical: quoted strings and identifiers,
    dollar-quoted bodies, comments, and nested parentheses cannot turn their
    contents into top-level control tokens.
    """

    tokens = _top_level_tokens(_scan_statement_tokens(sql))
    for prefix in _FORBIDDEN_PREFIXES:
        if tokens[: len(prefix)] == prefix:
            raise HologresSqlError("Forbidden top-level SQL command")
    return sql


def _statement_end(
    tokens_with_depth: tuple[_ScannedToken, ...], start: int, depth: int
) -> int:
    for index in range(start + 1, len(tokens_with_depth)):
        if tokens_with_depth[index][1] < depth:
            return index
    return len(tokens_with_depth)


def _select_has_top_level_into(
    tokens_with_depth: tuple[_ScannedToken, ...],
    start: int,
    end: int,
    depth: int,
) -> bool:
    return any(
        token == "INTO" and token_depth == depth
        for token, token_depth in tokens_with_depth[start + 1 : end]
    )


def _next_token_at_depth(
    tokens_with_depth: tuple[_ScannedToken, ...],
    start: int,
    end: int,
    depth: int,
) -> int:
    for index in range(start, end):
        if tokens_with_depth[index][1] == depth:
            return index
    return end


def _read_only_with(
    tokens_with_depth: tuple[_ScannedToken, ...],
    start: int,
    end: int,
    depth: int,
) -> bool:
    index = _next_token_at_depth(tokens_with_depth, start + 1, end, depth)
    if index < end and tokens_with_depth[index] == ("RECURSIVE", depth):
        index = _next_token_at_depth(tokens_with_depth, index + 1, end, depth)

    saw_read_only_cte = False
    while index < end:
        token, _token_depth = tokens_with_depth[index]
        if saw_read_only_cte and token in _WITH_READ_COMMANDS:
            return _read_only_statement(tokens_with_depth, index, end, depth)

        if token != "AS":
            index = _next_token_at_depth(tokens_with_depth, index + 1, end, depth)
            if index >= end or tokens_with_depth[index] != ("AS", depth):
                return False

        body_start = index + 1
        while body_start < end and tokens_with_depth[body_start] in {
            ("NOT", depth),
            ("MATERIALIZED", depth),
        }:
            body_start += 1
        if body_start >= end:
            return False
        _body_token, body_depth = tokens_with_depth[body_start]
        if body_depth <= depth:
            return False
        body_end = _statement_end(tokens_with_depth, body_start, body_depth)
        if not _read_only_statement(
            tokens_with_depth, body_start, body_end, body_depth
        ):
            return False
        saw_read_only_cte = True
        index = _next_token_at_depth(tokens_with_depth, body_end, end, depth)
    return False


def _read_only_statement(
    tokens_with_depth: tuple[_ScannedToken, ...],
    start: int,
    end: int,
    depth: int,
) -> bool:
    command, command_depth = tokens_with_depth[start]
    if command_depth != depth:
        return False
    if command == "WITH":
        return _read_only_with(tokens_with_depth, start, end, depth)
    if command not in _READ_COMMANDS:
        return False
    return command != "SELECT" or not _select_has_top_level_into(
        tokens_with_depth, start, end, depth
    )


def _explain_statement_start(
    tokens_with_depth: tuple[_ScannedToken, ...],
) -> int | None:
    for index, (token, depth) in enumerate(tokens_with_depth[1:], start=1):
        if depth != 0 or token in _EXPLAIN_ANALYZE_TOKENS or token == "VERBOSE":
            continue
        return index
    return None


def _is_explain_analyze_option(token: _SqlToken) -> bool:
    if isinstance(token, _QuotedIdentifier):
        return token.value.upper() in _EXPLAIN_ANALYZE_TOKENS
    return token in _EXPLAIN_ANALYZE_TOKENS


def _explain_executes_statement(
    tokens_with_depth: tuple[_ScannedToken, ...], statement_start: int
) -> bool:
    false_values = {"FALSE", "OFF", "NO", "0"}
    options = tokens_with_depth[1:statement_start]
    for index, (token, depth) in enumerate(options):
        if not _is_explain_analyze_option(token):
            continue
        if depth == 0:
            return True
        if index + 1 >= len(options):
            return True
        next_token, next_depth = options[index + 1]
        if next_depth != depth or next_token not in false_values:
            return True
    return False


def _is_lexically_read_only(sql: str) -> bool:
    tokens_with_depth = _scan_statement_tokens(sql)
    top_level = _top_level_tokens(tokens_with_depth)
    if top_level[0] != "EXPLAIN":
        return _read_only_statement(
            tokens_with_depth, 0, len(tokens_with_depth), 0
        )

    statement_start = _explain_statement_start(tokens_with_depth)
    return statement_start is not None and not _explain_executes_statement(
        tokens_with_depth, statement_start
    )


def _row_operation_replay_safe(
    sql: str,
    operation_kind: OperationKind,
    replay_safe: bool | None,
) -> bool:
    validate_single_statement(sql)
    if not isinstance(operation_kind, OperationKind):
        raise HologresSqlError("Invalid row-returning operation kind")
    if replay_safe is not None and not isinstance(replay_safe, bool):
        raise HologresSqlError("replay_safe must be a boolean")
    if operation_kind is OperationKind.READ:
        if not _is_lexically_read_only(sql):
            raise HologresSqlError(
                "Row-returning READ requires lexically proven read-only SQL"
            )
        return True
    if replay_safe is None:
        raise HologresSqlError(
            "Row-returning writes require an explicit replay_safe decision"
        )
    return replay_safe


def _validate_descriptor(descriptor: str) -> str:
    if _DESCRIPTOR.fullmatch(descriptor) is None:
        raise HologresClientError("Invalid operation descriptor")
    return descriptor


def _asyncpg_connection_loss_types() -> tuple[type[BaseException], ...]:
    try:
        import asyncpg
    except ImportError:
        return ()
    return (
        asyncpg.exceptions.PostgresConnectionError,
        asyncpg.exceptions.CannotConnectNowError,
    )


def is_connection_loss(error: BaseException) -> bool:
    """Classify failures that safely indicate a lost database connection."""

    sqlstate = getattr(error, "sqlstate", None)
    if isinstance(sqlstate, str) and sqlstate.startswith("08"):
        return True
    if isinstance(error, (ConnectionError, BrokenPipeError, EOFError)):
        return True
    connection_types = _asyncpg_connection_loss_types()
    return bool(connection_types and isinstance(error, connection_types))


def should_retry_operation(
    kind: OperationKind, error: BaseException, *, replay_safe: bool
) -> bool:
    """Apply the read/write replay contract to a database failure."""

    if not is_connection_loss(error):
        return False
    return kind is OperationKind.READ or replay_safe


def _error_category(error: BaseException) -> str:
    return "connection_loss" if is_connection_loss(error) else "database_error"


def _requires_invalidation(
    kind: OperationKind, error: BaseException, *, replay_safe: bool
) -> bool:
    if is_connection_loss(error):
        return True
    return (
        kind is OperationKind.WRITE
        and not replay_safe
        and isinstance(error, (TimeoutError, asyncio.CancelledError))
    )


async def _default_pool_factory(**kwargs: Any) -> Any:
    try:
        import asyncpg
    except ImportError:
        raise HologresClientError(
            "Hologres requires the asyncpg offline-storage dependency"
        ) from None

    pool = asyncpg.create_pool(**kwargs)
    try:
        return await pool
    except BaseException:
        terminate = getattr(pool, "terminate", None)
        if terminate is not None:
            try:
                terminate()
            except BaseException:
                pass
        raise


class HologresClient:
    """A restricted pool-backed API that never exposes asyncpg connections."""

    def __init__(
        self,
        config: HologresConfig,
        *,
        pool: Any | None = None,
        pool_factory: Callable[..., Awaitable[Any]] | None = None,
        capabilities: "CapabilityReport | None" = None,
    ) -> None:
        self.config = config
        self._pool = pool
        self._pool_factory = pool_factory or _default_pool_factory
        self._capabilities = capabilities
        self._open_lock = asyncio.Lock()

    async def open(self) -> None:
        """Create the asyncpg pool under the shared lifecycle lock."""

        async with self._open_lock:
            if self._pool is not None:
                return

            async def setup(connection: Any) -> None:
                statement = (
                    f"SET search_path TO {quote_identifier(self.config.schema)}"
                )
                validate_single_statement(statement)
                await connection.execute(
                    statement, timeout=self.config.command_timeout
                )

            async def reset(connection: Any) -> None:
                for statement in _RESET_STATEMENTS:
                    validate_single_statement(statement)
                    await connection.execute(
                        statement, timeout=self.config.command_timeout
                    )

            try:
                self._pool = await self._pool_factory(
                    host=self.config.host,
                    port=self.config.port,
                    user=self.config.user,
                    password=self.config.password,
                    database=self.config.database,
                    min_size=self.config.pool_min_size,
                    max_size=self.config.pool_max_size,
                    timeout=self.config.connect_timeout,
                    command_timeout=self.config.command_timeout,
                    statement_cache_size=self.config.statement_cache_size,
                    ssl=self.config.ssl_mode,
                    setup=setup,
                    reset=reset,
                )
            except HologresClientError:
                raise
            except Exception:
                raise HologresOperationError(
                    "Hologres pool initialization failed"
                ) from None

    async def close(self) -> None:
        """Close or terminate the pool under the shared lifecycle lock."""

        async with self._open_lock:
            pool = self._pool
            if pool is None:
                return
            try:
                await asyncio.wait_for(
                    pool.close(), timeout=self.config.pool_close_timeout
                )
            except asyncio.CancelledError:
                self._invalidate(pool)
                raise
            except TimeoutError:
                self._invalidate(pool)
                raise HologresOperationError(
                    "Hologres pool close timed out"
                ) from None
            except Exception:
                self._invalidate(pool)
                raise HologresOperationError("Hologres pool close failed") from None
            finally:
                if self._pool is pool:
                    self._pool = None

    def apply_capabilities(self, capabilities: "CapabilityReport") -> None:
        """Apply an immutable, already-probed capability report."""

        self._capabilities = capabilities

    async def _ensure_pool(self) -> Any:
        if self._pool is None:
            await self.open()
        if self._pool is None:  # defensive for unusual pool factories
            raise HologresOperationError("Hologres pool is unavailable")
        return self._pool

    @staticmethod
    def _invalidate(connection: Any | None) -> None:
        if connection is None:
            return
        terminate = getattr(connection, "terminate", None)
        if terminate is not None:
            try:
                terminate()
            except Exception:
                pass

    async def _run(
        self,
        kind: OperationKind,
        operation: Callable[[Any], Awaitable[_T]],
        *,
        descriptor: str,
        replay_safe: bool,
    ) -> _T:
        _validate_descriptor(descriptor)
        attempt = 0
        while True:
            pool = await self._ensure_pool()
            connection = None
            invalidated = False
            try:
                async with pool.acquire(
                    timeout=self.config.pool_acquire_timeout
                ) as connection:
                    try:
                        return await operation(connection)
                    except asyncio.CancelledError as error:
                        if _requires_invalidation(
                            kind, error, replay_safe=replay_safe
                        ):
                            self._invalidate(connection)
                        raise
                    except Exception as error:
                        if _requires_invalidation(
                            kind, error, replay_safe=replay_safe
                        ):
                            self._invalidate(connection)
                            invalidated = True
                        raise
            except Exception as error:
                retry = should_retry_operation(
                    kind, error, replay_safe=replay_safe
                )
                if (
                    _requires_invalidation(kind, error, replay_safe=replay_safe)
                    and not invalidated
                ):
                    self._invalidate(connection)
                if retry and attempt < self.config.connection_retries:
                    delay = self.config.retry_backoff * (2**attempt)
                    attempt += 1
                    if delay:
                        await asyncio.sleep(delay)
                    continue
                category = _error_category(error)
                raise HologresOperationError(
                    f"Hologres {kind.value} operation failed ({category})"
                ) from None

    async def execute_one(
        self,
        sql: str,
        *values: Any,
        descriptor: str,
        workspace: str | None = None,
        replay_safe: bool = False,
        timeout: float | None = None,
    ) -> str:
        """Execute one guarded parameterized write statement."""

        del workspace
        validate_single_statement(sql)
        command_timeout = self.config.command_timeout if timeout is None else timeout
        return await self._run(
            OperationKind.WRITE,
            lambda connection: connection.execute(
                sql, *values, timeout=command_timeout
            ),
            descriptor=descriptor,
            replay_safe=replay_safe,
        )

    async def fetch_one(
        self,
        sql: str,
        *values: Any,
        descriptor: str,
        workspace: str | None = None,
        operation_kind: OperationKind = OperationKind.READ,
        replay_safe: bool | None = None,
        timeout: float | None = None,
    ) -> Any | None:
        """Fetch one row with explicit semantics for row-returning writes."""

        del workspace
        effective_replay_safe = _row_operation_replay_safe(
            sql, operation_kind, replay_safe
        )
        command_timeout = self.config.command_timeout if timeout is None else timeout
        return await self._run(
            operation_kind,
            lambda connection: connection.fetchrow(
                sql, *values, timeout=command_timeout
            ),
            descriptor=descriptor,
            replay_safe=effective_replay_safe,
        )

    async def fetch_all(
        self,
        sql: str,
        *values: Any,
        descriptor: str,
        workspace: str | None = None,
        operation_kind: OperationKind = OperationKind.READ,
        replay_safe: bool | None = None,
        timeout: float | None = None,
    ) -> Sequence[Any]:
        """Fetch rows with explicit semantics for row-returning writes."""

        del workspace
        effective_replay_safe = _row_operation_replay_safe(
            sql, operation_kind, replay_safe
        )
        command_timeout = self.config.command_timeout if timeout is None else timeout
        return await self._run(
            operation_kind,
            lambda connection: connection.fetch(
                sql, *values, timeout=command_timeout
            ),
            descriptor=descriptor,
            replay_safe=effective_replay_safe,
        )

    async def fetch_value(
        self,
        sql: str,
        *values: Any,
        descriptor: str,
        workspace: str | None = None,
        operation_kind: OperationKind = OperationKind.READ,
        replay_safe: bool | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Fetch a scalar with explicit semantics for row-returning writes."""

        del workspace
        effective_replay_safe = _row_operation_replay_safe(
            sql, operation_kind, replay_safe
        )
        command_timeout = self.config.command_timeout if timeout is None else timeout
        return await self._run(
            operation_kind,
            lambda connection: connection.fetchval(
                sql, *values, timeout=command_timeout
            ),
            descriptor=descriptor,
            replay_safe=effective_replay_safe,
        )

    async def copy_rows(
        self,
        table: str,
        columns: Sequence[str],
        records: Iterable[Sequence[Any]],
        *,
        descriptor: str,
        workspace: str | None = None,
        replay_safe: bool = False,
        timeout: float | None = None,
    ) -> Any:
        """Copy rows only after configuration and capability gates both pass."""

        del workspace
        from .capabilities import ProbeKind

        if not self.config.stream_copy_enabled:
            raise HologresCapabilityError("Stream COPY is disabled by configuration")
        if self._capabilities is None or not self._capabilities.supports(
            ProbeKind.STREAM_COPY
        ):
            raise HologresCapabilityError("Stream COPY capability is not proven")
        table_name = validate_identifier(table)
        column_names = tuple(validate_identifier(column) for column in columns)
        if not column_names:
            raise HologresSqlError("COPY requires at least one column")
        materialized_records = tuple(tuple(record) for record in records)
        command_timeout = self.config.command_timeout if timeout is None else timeout
        return await self._run(
            OperationKind.WRITE,
            lambda connection: connection.copy_records_to_table(
                table_name,
                records=materialized_records,
                columns=column_names,
                schema_name=self.config.schema,
                timeout=command_timeout,
            ),
            descriptor=descriptor,
            replay_safe=replay_safe,
        )

    async def _observe_setup_reset_for_probe(
        self, *, bound: int, marker: str
    ) -> _SetupResetProbeObservation:
        """Observe pool setup and reset without exposing a physical connection."""

        seed_sql = (
            "SELECT current_schema() AS schema_name, "
            "$1::integer AS bound, "
            "pg_backend_pid() AS backend_pid, "
            "set_config('application_name', $2, false) AS seeded_application_name"
        )
        verify_sql = (
            "SELECT pg_backend_pid() AS backend_pid, "
            "current_setting('application_name') AS application_name"
        )
        validate_single_statement(seed_sql)
        validate_single_statement(verify_sql)
        pool = await self._ensure_pool()
        async with pool.acquire(
            timeout=self.config.pool_acquire_timeout
        ) as connection:
            setup_row = await connection.fetchrow(
                seed_sql,
                bound,
                marker,
                timeout=self.config.command_timeout,
            )
        seeded_backend_pid = setup_row["backend_pid"]

        max_candidates = max(1, min(self.config.pool_max_size, 16))
        async with AsyncExitStack() as acquired_connections:
            for _ in range(max_candidates):
                connection = await acquired_connections.enter_async_context(
                    pool.acquire(timeout=self.config.pool_acquire_timeout)
                )
                verify_row = await connection.fetchrow(
                    verify_sql, timeout=self.config.command_timeout
                )
                if (
                    verify_row is not None
                    and verify_row["backend_pid"] == seeded_backend_pid
                ):
                    return _SetupResetProbeObservation(
                        schema_name=setup_row["schema_name"],
                        bound=setup_row["bound"],
                        same_connection_reacquired=True,
                        application_name=verify_row["application_name"],
                    )

        return _SetupResetProbeObservation(
            schema_name=setup_row["schema_name"],
            bound=setup_row["bound"],
            same_connection_reacquired=False,
            application_name=None,
        )

    async def _reconnect_for_probe(self) -> None:
        """Exercise pool reconnect behavior for the isolated probe harness."""

        await self.close()
        await self.open()


class _SharedClientEntry:
    """One private shared-client entry without a credential-bearing repr."""

    __slots__ = ("client", "references")

    def __init__(self, client: HologresClient) -> None:
        self.client = client
        self.references = 1


class HologresClientManager:
    """Reference-count complete configurations onto restricted clients."""

    def __init__(
        self,
        *,
        client_factory: Callable[[HologresConfig], HologresClient] = HologresClient,
    ) -> None:
        self._client_factory = client_factory
        self._entries: dict[HologresConfig, _SharedClientEntry] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, config: HologresConfig) -> HologresClient:
        """Open or reference the one client associated with ``config``."""

        async with self._lock:
            entry = self._entries.get(config)
            if entry is not None:
                entry.references += 1
                return entry.client

            try:
                client = self._client_factory(config)
            except asyncio.CancelledError:
                raise
            except Exception:
                raise HologresOperationError(
                    "Hologres shared client creation failed"
                ) from None
            try:
                await client.open()
            except asyncio.CancelledError:
                await self._discard_failed_open(client)
                raise
            except Exception:
                await self._discard_failed_open(client)
                raise HologresOperationError(
                    "Hologres shared client open failed"
                ) from None

            self._entries[config] = _SharedClientEntry(client)
            return client

    async def release(
        self, config: HologresConfig, client: HologresClient
    ) -> bool:
        """Release one identity-matched reference and close the final client."""

        async with self._lock:
            entry = self._entries.get(config)
            if entry is None or entry.client is not client:
                return False
            entry.references -= 1
            if entry.references:
                return False
            del self._entries[config]

        try:
            await client.close()
        except asyncio.CancelledError:
            raise
        except Exception:
            raise HologresOperationError(
                "Hologres shared client close failed"
            ) from None
        return True

    async def _discard_failed_open(self, client: HologresClient) -> None:
        try:
            await client.close()
        except BaseException:
            pass

    def snapshot_for_tests(self) -> tuple[int, tuple[int, ...]]:
        """Return counts only; configuration values are never exposed."""

        references = tuple(sorted(entry.references for entry in self._entries.values()))
        return len(references), references

    def __repr__(self) -> str:
        return "HologresClientManager(<redacted>)"
